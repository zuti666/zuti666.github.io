---
layout: post
title:  Paper Reading 16 LLM finetuning and forgetting - 3
categories: [Paper Reading,  LLM, Continual Learning,] 
description:  [Learn or Recall? Revisiting Incremental Learning with Pre-trained Language Models]
keywords: [Paper Reading,  LLM, Continual Learning, ] 
---



# Paper Reading 16  16 LLM finetuning and forgetting - 3 





####  EFFECT OF MODEL AND PRETRAINING SCALE ON CATASTROPHIC FORGETTING IN NEURAL NETWORKS

- **EFFECT OF MODEL AND PRETRAINING SCALE ON  CATASTROPHIC FORGETTING IN NEURAL NETWORKS**

  [`semanticscholar`](https://www.semanticscholar.org/paper/9490d42c4869e6d6f3308c9813b1cfe31ff80137)  [`Paper`](https://www.semanticscholar.org/paper/9490d42c4869e6d6f3308c9813b1cfe31ff80137)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F9490d42c4869e6d6f3308c9813b1cfe31ff80137%3Ffields%3DcitationCount)

  2022     

![img](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250112122552360.png)



![img](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250112122551446.png)



large, pretrained ResNets and Transformers are significantly more resistant to forgetting than randomly-initialized, trained-from-scratch models; this robustness systematically improves with scale of both model and pretraining dataset size.



看起来，模型越大，神经网络参数越多，所对应的数据数据集合也就越大，从而导致 遗忘越小。 其在 zero-shoot 和 finetuned 之后的效果也都表现更好。











- **An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning**

  [`semanticscholar`](https://www.semanticscholar.org/paper/838cd69a0b6c9c244a6eebb0f4742c0625132de6)  [`Paper`](https://www.semanticscholar.org/paper/838cd69a0b6c9c244a6eebb0f4742c0625132de6)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F838cd69a0b6c9c244a6eebb0f4742c0625132de6%3Ffields%3DcitationCount)

  2023    arXiv.org 



![img](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250112124321912.png)





![image-20250112124338617](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250112124338683.png)
