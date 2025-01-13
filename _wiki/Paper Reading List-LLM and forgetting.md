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

   



## Finetuning- Instruction tuning 



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





## New Reading not category

- **Amuro & Char: Analyzing the Relationship between Pre-Training and Fine-Tuning of Large Language Models**

  [`semanticscholar`](https://www.semanticscholar.org/paper/c6bd5689eaf755e274f1286b12cf27021eb7da32)  [`Paper`](https://www.semanticscholar.org/paper/c6bd5689eaf755e274f1286b12cf27021eb7da32)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fc6bd5689eaf755e274f1286b12cf27021eb7da32%3Ffields%3DcitationCount)

  2024    arXiv.org 

  测试了在预训练的不同训练阶段的大模型的zero-shot以及 fine-tuning(full-parameter finetuning and instruction Tuning)之后的表现。

  但是只使用一个模型，且参数量很小 OLMo-1B。  

  对于大模型没有训练过的数据，finetuning能带来更好的提升效果，但也会造成大模型已有知识的遗忘。

  

  method:

  ![image-20250113120041928](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250113120041979.png)

  experiment result: 

  ![image-20250113120028229](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250113120028288.png)









- **Balancing Continuous Pre-Training and Instruction Fine-Tuning: Optimizing Instruction-Following in LLMs**

  [`semanticscholar`](https://www.semanticscholar.org/paper/9393b0c00d509d58f9e8bb782fb9ec2c3bbf1b3c)  [`Paper`](https://www.semanticscholar.org/paper/9393b0c00d509d58f9e8bb782fb9ec2c3bbf1b3c)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F9393b0c00d509d58f9e8bb782fb9ec2c3bbf1b3c%3Ffields%3DcitationCount)

  ​     Investigate two setting , the first is Pre-Training--> Finetuning -->  Pre-Training om new data , the other is Pre-Training -->  Pre-Training om new data --> Finetuning .  Result show the latter is better.  

  ​	Experiments using 8B LLmaMa 3.1 . 









- **CEM: A Data-Efficient Method for Large Language Models to Continue Evolving From Mistakes**

  [`semanticscholar`](https://www.semanticscholar.org/paper/bcce50b620210054b59ec84bee7b844099bb3f89)  [`Paper`](https://www.semanticscholar.org/paper/bcce50b620210054b59ec84bee7b844099bb3f89)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fbcce50b620210054b59ec84bee7b844099bb3f89%3Ffields%3DcitationCount)

  ​     This paper propose a method that provide the mistake result to the model  to train it. The method is a little bit of complex. But the paper mention a view that the failure has two reason: unfamiliarity with the task schema and insufficient task knowledge.

  

  ![image-20250113130630440](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250113130630506.png)

![image-20250113130923093](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250113130923194.png)



- **ConTinTin: Continual Learning from Task Instructions**

  [`semanticscholar`](https://www.semanticscholar.org/paper/0c8908707b4609bc53ea7a7c1d855088b7294dcf)  [`Paper`](https://www.semanticscholar.org/paper/0c8908707b4609bc53ea7a7c1d855088b7294dcf)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F0c8908707b4609bc53ea7a7c1d855088b7294dcf%3Ffields%3DcitationCount)

  2022    Annual Meeting of the Association for Computational Linguistics 

  ​     

  The paper want the PLMs lean new task by text instruction and some examples. The author expect the method have better performance is Knowledge maintenance and transfer.





![image-20250113133702435](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250113133702480.png)



![image-20250113133341278](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250113133341355.png)



![image-20250113133605765](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250113133605821.png)







- **Continual Pre-Training of Large Language Models: How to (re)warm your  model?**

  [`semanticscholar`](https://www.semanticscholar.org/paper/193955704f66923ac20a664bd184ed4663b2bdf9)  [`Paper`](https://www.semanticscholar.org/paper/193955704f66923ac20a664bd184ed4663b2bdf9)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F193955704f66923ac20a664bd184ed4663b2bdf9%3Ffields%3DcitationCount)

  2023    arXiv.org 

​	





- **Does RoBERTa Perform Better than BERT in Continual Learning: An Attention Sink Perspective**

  [`semanticscholar`](https://www.semanticscholar.org/paper/9a1f22898d95514d1d4223b04cb3e0e2feef7e15)  [`Paper`](https://www.semanticscholar.org/paper/9a1f22898d95514d1d4223b04cb3e0e2feef7e15)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F9a1f22898d95514d1d4223b04cb3e0e2feef7e15%3Ffields%3DcitationCount)

  2024    arXiv.org 



