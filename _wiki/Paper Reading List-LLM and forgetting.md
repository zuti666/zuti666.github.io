---
layout: wiki
title: Paper Reading List - LLM and forgetting
categories: LLM and forgetting
description: LLM and forgetting 论文阅读汇总记录
keywords: LLM and forgetting
---





# Paper Reading--LLM and forgetting



## Survey

**2025**





**2024**

- 



## Forgetting in   LLM 



- **EFFECT OF MODEL AND PRETRAINING SCALE ON  CATASTROPHIC FORGETTING IN NEURAL NETWORKS**

​	[`semanticscholar`](https://www.semanticscholar.org/paper/9490d42c4869e6d6f3308c9813b1cfe31ff80137)  [`Paper`](https://www.semanticscholar.org/paper/9490d42c4869e6d6f3308c9813b1cfe31ff80137)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F9490d42c4869e6d6f3308c9813b1cfe31ff80137%3Ffields%3DcitationCount)

​	2022   

​	 scale matters, big model have high anti-forgetting abilty



- **CAN BERT REFRAIN FROM FORGETTING ON SEQUENTIAL TASKS? A PROBING STUDY**

  [`semanticscholar`](https://www.semanticscholar.org/paper/201047e827ed9587158fc71256c576c8544e3dfc)  [`Paper`](https://www.semanticscholar.org/paper/201047e827ed9587158fc71256c576c8544e3dfc)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F201047e827ed9587158fc71256c576c8544e3dfc%3Ffields%3DcitationCount)

  2023    International Conference on Learning Representations 
  
  预训练模型有着很强的持续学习能力，同一任务下的子任务在训练前后保持有序，但是会和新的任务数据产生重叠，这是遗忘的主要原因，而 之前数据的加入会减轻这种遗忘。
  
  ![image-20250112143024907](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250112143024955.png)



- **Learn or Recall? Revisiting Incremental Learning with Pre-trained Language Models**

  [`semanticscholar`](https://www.semanticscholar.org/paper/9e2a811a6f5d1c5352ce19ac24303810eb1867f7)  [`Paper`](https://www.semanticscholar.org/paper/9e2a811a6f5d1c5352ce19ac24303810eb1867f7)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F9e2a811a6f5d1c5352ce19ac24303810eb1867f7%3Ffields%3DcitationCount)

  2023    Annual Meeting of the Association for Computational Linguistics
  
  使用了上面一片论文提出的probing performance的思路，固定大模型，每一个任务都只训练一个分类器，将结果视为大模型的表征能力。发现这么做的话大模型都能表现很好。
  
  所以如果大模型固定的时候，就不能让之前的分类器权重也发生变化，这就是表现下降的原因。提出方法固定大模型，固定之前的分类器。 实验表明，即使在持续学习过程中不固定大模型，只固定之前的分类器，效果也会得到很大提升。
  
  但是不足在于这个文章的实验只在文本分类任务，同一个数据集下面进行测试的。
  
  ![image-20250112143535410](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250112143535465.png)
  
  
  
  ![image-20250112143730870](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250112143730934.png)
  
  

### common insight

1. scale matters, big model have high anti-forgetting abilty

   



# Finetuning- Instruction tuning 



- **FINETUNED LANGUAGE MODELS ARE ZERO-SHOT  LEARNERS**

​	[`semanticscholar`](https://www.semanticscholar.org/paper/ff0b2681d7b05e16c46dfb71d980cc2f605907cd)  [`Paper`](https://www.semanticscholar.org/paper/ff0b2681d7b05e16c46dfb71d980cc2f605907cd)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fff0b2681d7b05e16c46dfb71d980cc2f605907cd%3Ffields%3DcitationCount)

​	2021    International Conference on Learning Representations 

​	作者介绍了 instruction tuning的概念， finetuning language models on a collection of datasets described via instructions， 然后再不同的任务和模型了进行了实验，表明 instruction tuning能够提升模型zero-shoot 的表现

​	![image-20250112142034022](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250112142034060.png)





- **An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning**

  [`semanticscholar`](https://www.semanticscholar.org/paper/838cd69a0b6c9c244a6eebb0f4742c0625132de6)  [`Paper`](https://www.semanticscholar.org/paper/838cd69a0b6c9c244a6eebb0f4742c0625132de6)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F838cd69a0b6c9c244a6eebb0f4742c0625132de6%3Ffields%3DcitationCount)

  2023    arXiv.org 

​	作者在不同的任务上，测试了 在 continual learning 设置下进行 instruction tuning 的效果。实验表明，在参数从1b到7 b的LLM中通常会观察到灾难性遗忘。此外，随着模型规模的增加，遗忘的严重性加剧。

![img](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUf6JfHCmuM_kFnGoUiJujCyHjb9Ph3_50oNtUI4xDdS3k_sTH3o4OXR7bFCDpl_GYTeW0Y_LHDnHV0bwAFdrD5No7bk52gWbCNYDM4THrvnHDhz8zRxIaZwLeibfkTRoJb7koVnWg=nw?key=rA2QzXi6sCFyF5QWS0mimBun)



![image-20250112142349142](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250112142349196.png)
