# 🔥 PromptLT: Long-tailed Multimodal Prompt Learning 🔥

This is the source code for the paper: PromptLT: Long-tailed Multimodal Prompt Learning.

<p align="center"><img src="fig/larger.jpg" style="max-width: 70%; height: auto;" id="title-icon">    </p>

##### <font color=Red>We feel sorry that lines below in the paper have a little problems.</font> 
Some modifications here:
1. line 147: head-new -> head-base
2. line 154: tail-based -> tail-base; unlabeled base classes ~~and new tail-base samples~~ can then naturally ...
3. Equation 8: exchange t+ and v+ 
4. EOF-token -> EOT-token

##### <font color=Red>Apologize again for potential misunderstanding. Thanks for your attention and caring!</font>

## What's New:
2025.5 We release the code !

## Install

This code is built on top of the codebase for [CoOp](https://github.com/KaiyangZhou/CoOp) and [Candle](https://github.com/shijxcs/Candle/). Please follow their instructions to prepare the required environments and datasets.

## Run
Please refer to the running scripts under `scripts/promptlt` for different experimental settings.  

Make sure to change the path of `DATA` in corresponding scripts and run the commands under the main directory.  

In `run_example.sh`, I provide some examples on how to run the model.
