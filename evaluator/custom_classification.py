import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix

from dassl.evaluation import EVALUATOR_REGISTRY, EvaluatorBase


@EVALUATOR_REGISTRY.register()
class CustomClassification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )
        pp = f"=> result * REPDIM:{self.cfg.TRAINER.PromptLT.REP_DIM}-NUMSHOTS:{self.cfg.DATASET.NUM_SHOTS}-IMBRATIO:{self.cfg.DATASET.IMBALANCE_RATIO}-{self.cfg.TRAINER.PHASE}-{self.cfg.DATASET.SUBSAMPLE_CLASSES}-{self.cfg.DATASET.NAME}-{self.cfg.TRAINER.NAME}-{self.cfg.SEED} * total: {self._total:,} * correct: {self._correct:,} * accuracy: {acc:.1f}% * error: {err:.1f}% * macro_f1: {macro_f1:.1f}%\n"
        with open("/data/wxc/youmi.txt","a") as f:
            f.write(pp)
        f.close()

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                # print(
                #     f"* class: {label} ({classname})\t"
                #     f"total: {total:,}\t"
                #     f"correct: {correct:,}\t"
                #     f"acc: {acc:.1f}%"
                # )
            
            accs_string = np.array2string(np.array(accs), precision=2)
            print(f"* class acc: {accs_string}")

            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results
