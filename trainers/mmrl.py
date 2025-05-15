import os
import os.path as osp
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.modules.loss import _Loss

from tqdm import tqdm
from trainers.losses import LogitAdjustedLoss
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.'
}


def load_clip_to_cpu(cfg, model_name="MMRL"):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": "MMRL",
                      "rep_tokens_layers": cfg.TRAINER.MMRL.REP_LAYERS,
                      "n_rep_tokens": cfg.TRAINER.MMRL.N_REP_TOKENS,
                      "vision_depth": 0,
                      "language_depth": 0, 
                      "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder_MMRL(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_rep_tokens_text):
        n_rep_tokens = compound_rep_tokens_text[0].shape[0]
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # important: add the batchsize dimension !!!
        #---------------------------------------------------- original
        # x = x.unsqueeze(dim=1).repeat(1,4,1,1) # LCD -> LBCD
        # --------------------------------------------------- new version
    
        
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        eot_index = tokenized_prompts.argmax(dim=-1)
        combined = [x, compound_rep_tokens_text, 0,
                    eot_index]  # third argument is the counter which denotes depth of representation tokens

        outputs = self.transformer(combined)

        x = outputs[0]  # extract the x back from here
        
        
        #--------------------------------------------------------------------------------------- original
        # x = x.permute(2, 0, 1, 3)  # LBCD -> CLBD
        # x = self.ln_final(x).type(self.dtype)
        # x = x[torch.arange(x.shape[0]), eot_index + n_rep_tokens] @ self.text_projection # CBD
        # x = x.permute(1,0,2) # CBD -> BCD
        #---------------------------------------------------------------------------------------
        x = x.permute(1,0,2) # LCD -> CLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), eot_index + n_rep_tokens] @ self.text_projection # CLD
        #---------------------------------------------------------------------------------------
        
        return x

 

# def _get_text_base_features_zero_shot(cfg, classnames, clip_model, text_encoder):
#     device = next(text_encoder.parameters()).device

#     text_encoder = text_encoder.cuda()
#     dataset = cfg.DATASET.NAME
#     template = CUSTOM_TEMPLATES[dataset]

#     with torch.no_grad():
#         tokenized_prompts = []
#         for text in tqdm(classnames, desc="Extracting text features"):
#             tokens = clip.tokenize(template.format(text.replace('_', ' ')))  # (n_tokens)
#             tokens = tokens.to(device)
#             tokenized_prompts.append(tokens)
#         tokenized_prompts = torch.cat(tokenized_prompts)  # (n_classes, n_tokens)

#         embeddings = clip_model.token_embedding(tokenized_prompts).type(
#             clip_model.dtype)  # (n_classes, n_tokens, embed_dim)
#         outputs = text_encoder(embeddings.cuda(), tokenized_prompts.cuda())

#         text_embeddings = outputs

#     text_encoder = text_encoder.to(device)
#     return text_embeddings


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MultiModalRepresentationLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        n_shared_token = cfg.TRAINER.MMRL.N_REP_TOKENS
        self.dtype = clip_model.dtype

        text_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.visual.ln_post.weight.shape[0]

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        n_shared_dim = cfg.TRAINER.MMRL.REP_DIM

        self.rep_layers_length = len(cfg.TRAINER.MMRL.REP_LAYERS)  # max=12
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        dataset = cfg.DATASET.NAME
        template = CUSTOM_TEMPLATES[dataset]
        tokenized_prompts = []
        for text in classnames:
            tokens = clip.tokenize(template.format(text.replace('_', ' ')))  # (n_tokens)
            tokenized_prompts.append(tokens)
        self.tokenized_prompts = torch.cat(tokenized_prompts)  # (n_classes, n_tokens)

        with torch.no_grad():
            self.prompt_embeddings = clip_model.token_embedding(self.tokenized_prompts).type(
                self.dtype)  # (n_classes, n_tokens, embed_dim)

        self.compound_rep_prompts = nn.Parameter(torch.empty(self.rep_layers_length, n_shared_token, n_shared_dim))
        nn.init.normal_(self.compound_rep_prompts, std=0.02)
        single_layer_r2v = nn.Sequential(nn.Linear(n_shared_dim, 32), nn.ReLU(), nn.Linear(32, visual_dim))
        single_layer_r2t = nn.Sequential(nn.Linear(n_shared_dim, 32), nn.ReLU(), nn.Linear(32, text_dim))

        self.compound_rep_tokens_r2vproj = _get_clones(single_layer_r2v, self.rep_layers_length)
        self.compound_rep_tokens_r2tproj = _get_clones(single_layer_r2t, self.rep_layers_length)

        self.extra_visual_prompts = nn.Parameter(torch.empty(self.rep_layers_length, 2, n_shared_dim))
        self.extra_textual_prompts = nn.Parameter(torch.empty(self.rep_layers_length, 2, n_shared_dim))
        nn.init.normal_(self.extra_visual_prompts, std=0.02)
        nn.init.normal_(self.extra_textual_prompts, std=0.02)

    def forward(self):
        compound_rep_tokens_visual = []
        compound_rep_tokens_text = []

        ready_for_use_shared_visual_tokens = []
        ready_for_use_shared_textual_tokens = []

        for index in range(self.rep_layers_length):
            rep_tokens = self.compound_rep_prompts[index]
            rep_mapped_to_text = self.compound_rep_tokens_r2tproj[index](rep_tokens)
            rep_mapped_to_visual = self.compound_rep_tokens_r2vproj[index](rep_tokens)
            compound_rep_tokens_text.append(rep_mapped_to_text.type(self.dtype))
            compound_rep_tokens_visual.append(rep_mapped_to_visual.type(self.dtype))

            extra_vision_tokens = self.extra_visual_prompts[index]
            extra_textual_tokens = self.extra_textual_prompts[index]
            ready_for_use_shared_textual_token_piece = self.compound_rep_tokens_r2tproj[index](extra_textual_tokens)
            ready_for_use_shared_visual_token_piece = self.compound_rep_tokens_r2vproj[index](extra_vision_tokens)
            ready_for_use_shared_visual_tokens.append(ready_for_use_shared_visual_token_piece)
            ready_for_use_shared_textual_tokens.append(ready_for_use_shared_textual_token_piece)

        return torch.stack(compound_rep_tokens_text), \
               torch.stack(compound_rep_tokens_visual), \
               torch.stack(ready_for_use_shared_textual_tokens), \
               torch.stack(ready_for_use_shared_visual_tokens)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.scale = cfg.TRAINER.MMRL.SCALE
        self.cfg = cfg
        self.classnames = classnames
        self.prompt_learner = MultiModalRepresentationLearner(cfg, classnames, clip_model).type(clip_model.dtype)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.register_buffer("prompt_embeddings", self.prompt_learner.prompt_embeddings)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.text_encoder = TextEncoder_MMRL(clip_model)
        self.dtype = clip_model.dtype
        self.text_features_for_inference = None
        self.compound_rep_tokens_text_for_inference = None
        self.compound_rep_tokens_visual_for_inference = None

    def FUSE(self, after_vision_prompt, after_text_prompt, extra_vison_prompt, extra_text_prompt, class_info, need_repeat): 
        nc = len(self.classnames) 
        use_nc = 100
        many_sp,  few_sp = use_nc*0.2, use_nc*0.6 
        if isinstance(class_info, str):
            if class_info == "base":
                # batch-repeat first.
                # input prompt: Nlayer, Nctx, dim
                after_vision_prompt = after_vision_prompt.unsqueeze(dim=0).repeat(need_repeat, 1, 1, 1) 
                after_text_prompt = after_text_prompt.unsqueeze(dim=0).repeat(nc, 1, 1, 1)
                many_text = torch.arange(nc).cuda() < many_sp
                medium_text = (torch.arange(nc).cuda() > many_sp) & (torch.arange(nc).cuda() < few_sp)
                few_text = torch.arange(nc).cuda() > few_sp
                after_text_prompt[many_text] = after_text_prompt[many_text]*0.5 + extra_text_prompt.unsqueeze(dim=0)*0.5  # 12 n b
 
                after_vision_prompt = after_vision_prompt*0.5 + extra_vison_prompt.unsqueeze(dim=0)*0.5
                return after_vision_prompt, after_text_prompt

            elif class_info == "new":
                # # batch-repeat first.
                # # input prompt: Nlayer, Nctx, dim
                # after_vision_prompt = after_vision_prompt.unsqueeze(dim=0).repeat(need_repeat, 1, 1, 1)
                # after_text_prompt = after_text_prompt.unsqueeze(dim=0).repeat(nc, 1, 1, 1)
                # return after_vision_prompt, after_text_prompt
                 
                after_vision_prompt = after_vision_prompt.unsqueeze(dim=0).repeat(need_repeat, 1, 1, 1) 
                after_text_prompt = after_text_prompt.unsqueeze(dim=0).repeat(nc, 1, 1, 1)
                many_text = torch.arange(nc).cuda() < many_sp
                medium_text = (torch.arange(nc).cuda() > many_sp) & (torch.arange(nc).cuda() < few_sp)
                few_text = torch.arange(nc).cuda() > few_sp
                after_text_prompt[many_text] = after_text_prompt[many_text]*0.5 + extra_text_prompt.unsqueeze(dim=0)*0.5  # 12 n b
 
                after_vision_prompt = after_vision_prompt*0.5 + extra_vison_prompt.unsqueeze(dim=0)*0.5
                return after_vision_prompt, after_text_prompt
                
            else:
                raise NotImplementedError()

        else:
            # batch-repeat first.
            # input prompt: Nlayer, Nctx, dim
            after_vision_prompt = after_vision_prompt.unsqueeze(dim=0).repeat(need_repeat, 1, 1, 1)
            
            #------------------------------------------------------------------------------------------------------- original
            # after_text_prompt = after_text_prompt.unsqueeze(dim=0).repeat(need_repeat, 1, 1, 1)
            #------------------------------------------------------------------------------------------------------- plus version
            after_text_prompt = after_text_prompt.unsqueeze(dim=0).repeat(nc, 1, 1, 1)
  
            many_mask = class_info < many_sp
            medium_mask = (class_info > many_sp) & (class_info < few_sp)
            few_mask  = class_info > few_sp
            class_mask = torch.zeros_like(class_info, dtype=after_vision_prompt.dtype).cuda()
            class_mask[many_mask] = 2
            class_mask[medium_mask] = 1
            class_mask[few_mask] = 0
            
            # ------------------------------------------------------------------------------------------------------- original
            # after_text_prompt[many_mask] = 0.5 * after_text_prompt[many_mask] + 0.5 * extra_text_prompt  # 4 12 n b
            # ------------------------------------------------------------------------------------------------------- plus version
            many_text = torch.arange(nc).cuda() < many_sp
            medium_text = (torch.arange(nc).cuda() > many_sp) & (torch.arange(nc).cuda() < few_sp)
            few_text = torch.arange(nc).cuda() > few_sp
            after_text_prompt[many_text] = after_text_prompt[many_text]*0.5 + extra_text_prompt.unsqueeze(dim=0)*0.5  # 12 n b
            # -------------------------------------------------------------------------------------------------------
             
            after_vision_prompt[few_mask] = after_vision_prompt[few_mask]*0.5 +  extra_vison_prompt.unsqueeze(dim=0)*0.5
            return after_vision_prompt, after_text_prompt

    def forward(self, image, class_info=None):
        need_repeat = image.shape[0]
        if self.prompt_learner.training:
            after_tokens_text, after_tokens_visual, extra_text, extra_vision = self.prompt_learner()
            after_tokens_visual, after_tokens_text = self.FUSE(after_tokens_visual, after_tokens_text, extra_vision, extra_text, class_info, need_repeat) # class_info is just labels
            text_features = self.text_encoder(self.prompt_embeddings, self.tokenized_prompts, after_tokens_text)
        else:
            after_tokens_text, after_tokens_visual, extra_text, extra_vision = self.prompt_learner()
            after_tokens_visual, after_tokens_text = self.FUSE(after_tokens_visual, after_tokens_text, extra_vision, extra_text, class_info, need_repeat) # class_info is str
            text_features = self.text_encoder(self.prompt_embeddings, self.tokenized_prompts, after_tokens_text)

            # if self.text_features_for_inference is None:
            #     self.compound_rep_tokens_text_for_inference, self.compound_rep_tokens_visual_for_inference, extra_text, extra_vision = self.prompt_learner()
            #     # after_tokens_visual, after_tokens_text = self.FUSE(after_tokens_visual, after_tokens_text, extra_vision, extra_text)
            #     self.text_features_for_inference = self.text_encoder(self.prompt_embeddings, self.tokenized_prompts,
            #                                                          self.compound_rep_tokens_text_for_inference)
            #
            # after_tokens_text, after_tokens_visual = self.compound_rep_tokens_text_for_inference, self.compound_rep_tokens_visual_for_inference
            # text_features = self.text_features_for_inference

        image_features = self.image_encoder([image.type(self.dtype), after_tokens_visual])
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #--------------------------------------------------------------------------------------------------------------------- original
        # logits = self.logit_scale.exp() * torch.bmm(image_features.unsqueeze(dim=1),  text_features.permute(0,2,1)).squeeze()
        #--------------------------------------------------------------------------------------------------------------------- new version
        logits = self.logit_scale.exp() * image_features @ text_features.t()

        return logits, image_features, text_features


class MMRL_Loss(_Loss):
    def __init__(self, reg_weight=1.0, scale=0.7):
        super(MMRL_Loss, self).__init__()
        self.reg_weight = reg_weight
        self.scale = scale

    def forward(self, logits, label):
        xe_loss1 = F.cross_entropy(logits, label)
        return xe_loss1


import pickle


@TRAINER_REGISTRY.register()
class MMRL(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MMRL.PREC in ["fp16", "fp32", "amp"]

    def save_variable(self, v, fp):
        f = open(fp, 'wb')
        pickle.dump(v, f, 0)
        f.close()
        return

    def load_variable(self, fp):
        try:
            f = open(fp, 'rb')
            r = pickle.load(f)
            f.close()
            return r

        except EOFError:
            return "Error: empty file!"

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.num_classes = len(classnames)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg, "MMRL")

        if cfg.TRAINER.MMRL.PREC == "fp32" or cfg.TRAINER.MMRL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        phase = cfg.TRAINER.PHASE
        task = cfg.TRAINER.TASK
        self.task = task + phase
        if (phase == "train"):
            cachepath = osp.join(cfg.OUTPUT_DIR, "cache.txt")
            if (task == "B2N"):
                classnames_all = self.dm.dataset.all_classnames
                num_classes_all = len(classnames_all)
                num_classes_train = self.dm.dataset.num_classes
                self.num_classes_all = num_classes_all
                self.num_classes_train = num_classes_train
                self.ncl = classnames_all

            elif (task == "XD"):
                classnames_train = self.dm.dataset.all_classnames
                num_classes_train = len(classnames_train)
                self.num_classes_train = num_classes_train
                class_names_new = self.dm.dataset_new.all_classnames
                classnames_all = classnames_train
                classnames_all.extend(class_names_new)
                self.num_classes_all = len(classnames_all)
                self.ncl = classnames_all

            v = [self.num_classes_all, self.num_classes_train]
            self.save_variable(v, cachepath)


        elif (phase == "test"):
            cachepath = osp.join(os.getcwd(), cfg.MODEL_DIR, "cache.txt")
            subsample = cfg.DATASET.SUBSAMPLE_CLASSES
            self.task += subsample
            # v = self.load_variable(cachepath)
            # self.num_classes_all, self.num_classes_train = v[0], v[1]

            # text_features = torch.empty((self.num_classes_all, 512), dtype=clip_model.dtype).to(self.device)
            # visual_prototypes = torch.empty((self.num_classes_train, 512), dtype=clip_model.dtype).to(self.device)

        self.dtype = clip_model.dtype

        # with torch.no_grad():
        #     self.text_encoder_clip = TextEncoder_CLIP(clip_model_zero_shot)
        #     text_features_clip = _get_text_base_features_zero_shot(cfg, classnames, clip_model_zero_shot,
        #                                                            self.text_encoder_clip)
        #     self.text_features_clip = text_features_clip / text_features_clip.norm(dim=-1, keepdim=True)
        # self.image_encoder_clip = clip_model_zero_shot.visual

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, self.dm.dataset.all_classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        names_to_update = ["prompt_learner"] # prompt_learner

        for name, param in self.model.named_parameters():
            update = False

            for name_to_update in names_to_update:
                if name_to_update in name:
                    update = True
                    break
            param.requires_grad_(update)

        # Double check
        num_trainable_params = 0
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                num_trainable_params += param.data.nelement()
        print(f"Parameters to be updated: {enabled}")
        print(f"Number of trainable parameters: {num_trainable_params}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # self.image_encoder_clip.to(self.device)

        reg_weight = cfg.TRAINER.MMRL.REG_WEIGHT
        scale = cfg.TRAINER.MMRL.SCALE

        # NOTE: only give representation_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MMRL.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)
        #     self.image_encoder_clip = nn.DataParallel(self.image_encoder_clip)

        # cls_num_list = self.dm.dataset.get_cls_num_list()
        # cls_num_list_new = [0 for i in range(self.num_classes_all - self.num_classes_train)]
        # cls_num_list.extend(cls_num_list_new)
        self.criterion = MMRL_Loss(reg_weight=reg_weight, scale=scale)  # LogitAdjustedLoss(cls_num_list=cls_num_list)

    def model_inference(self, input):
        output = self.model(input, self.cfg.DATASET.SUBSAMPLE_CLASSES)[0]
        m = 50 # self.num_classes_train
        # (batchsize, :num_classes_train)
        output_base = output[..., :m]
        # (batchsize, num_classes_train:)
        output_new = output  
 
        if (self.task == "B2Ntrainbase" or self.task == "B2Ntestbase" or self.task == "XDtrainall"):
            return output_base
        elif (self.task == "B2Ntestnew" or self.task == "XDtestall"):
            return output 
        else:
            return output

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch) 
        logits = self.model(image, label)[0]
  
        loss = self.criterion(logits, label)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        output = logits
        loss_summary = {
                        "loss": loss.item(),
                        "acc": compute_accuracy(output, label)[0].item()
                        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        sub_cls = self.cfg.DATASET.SUBSAMPLE_CLASSES
        dataset = self.cfg.DATASET.NAME
        task = self.cfg.TASK

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}-{self.cfg.DATASET.SUBSAMPLE_CLASSES}* set during training")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            logits = self.model(input, self.cfg.DATASET.SUBSAMPLE_CLASSES)[0]

            if task == "B2N":
                output = logits  # if sub_cls == "base" else logits
            elif task == "FS":
                output = logits
            elif task == "CD":
                output = logits  # if dataset == "ImageNet" else logits
            else:
                raise ValueError("The TASK must be either B2N, CD, or FS.")

            self.evaluator.process(output, label)
        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        # model_file = 'model-best.pth.tar'

        # if epoch is not None:
        #     model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            # model_path = osp.join(directory, name, model_file)
            model_path_prefix = osp.join(directory, name)
            if not osp.exists(model_path_prefix):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path_prefix)
                )
            for file in os.listdir(model_path_prefix):
                if "model-best.pth" in file:
                    model_path = osp.join(model_path_prefix, file)
                    break
                if "model.pth" in file:
                    model_path = osp.join(model_path_prefix, file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            state_dict = {k: v for k, v in state_dict.items() if "prompt_embeddings" not in k}

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)