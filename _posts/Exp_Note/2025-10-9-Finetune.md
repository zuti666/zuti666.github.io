---
layout: post
title:  Experiments 1
categories: [Continual Learning,  Experiments, FInetune]
description: 
keywords: [Continual Learning,  Experiments, FInetune]
---





# Exps 1 : Finetune Vit on Image-r 



Purpose : Check the difference of SAM and SGD on Finetune Whole Model, Evaluate the flat(weight-loss), and the feature space

Set: SGD vs SAM,  Finetune whole model vs Finetune only the calssifier 





Base Model : vit_base_patch16_224

Downstream Dataset:  imagenet-r 



## Exp1:  Finetune whole model

For SGD: 









## Exp2: Fix the backbone Finetune only the linear Classifier



 python Result_analyse/analyse_total.py

：

- python scripts/prepare_tiny_imagenet_c_split.py --in-root data/tiny-imagenet-c/extracted/Tiny-ImageNet-C --out-root data/tiny-imagenet-c-r --train-ratio 0.8 --seed 42




python scripts/prepare_tiny_imagenet_p_split.py --in-tar-root data/tiny-imagenet-p/extracted/Tiny-ImageNet-P --out-root data/tiny-imagenet-p-r --train-ratio 0.8 --seed 42 --frames-per-video 1
