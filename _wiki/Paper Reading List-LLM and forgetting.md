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







- **Towards Lifelong Learning of Large Language Models: A Survey**

  [`semanticscholar`](https://www.semanticscholar.org/paper/dcd0a2e67235add9e520e43c1d1fb4a89d76f98d)  [`Paper`](https://www.semanticscholar.org/paper/dcd0a2e67235add9e520e43c1d1fb4a89d76f98d)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fdcd0a2e67235add9e520e43c1d1fb4a89d76f98d%3Ffields%3DcitationCount)

  2024    arXiv.org 

![image-20250117153117899](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250117153117966.png)





![image-20250117152308109](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250117152308188.png)



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

​	The author investigate the warm-up state in the pre-training.  Our results show that while rewarming models first increases the loss on upstream and downstream data, in the longer run it improves the downstream performance.  







- **Does RoBERTa Perform Better than BERT in Continual Learning: An Attention Sink Perspective**

  [`semanticscholar`](https://www.semanticscholar.org/paper/9a1f22898d95514d1d4223b04cb3e0e2feef7e15)  [`Paper`](https://www.semanticscholar.org/paper/9a1f22898d95514d1d4223b04cb3e0e2feef7e15)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F9a1f22898d95514d1d4223b04cb3e0e2feef7e15%3Ffields%3DcitationCount)

  2024    arXiv.org 

​	The result is  RoBERTa is not perform better than BERT in the attention change, because it has attention sinks problem. 

![image-20250114123834873](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250114123834946.png)

​	



- **Efficient Continual Pre-training for Building Domain Specific Large Language Models**

  [`semanticscholar`](https://www.semanticscholar.org/paper/739cf040ed2c2af49077db48d489a46be5fb6157)  [`Paper`](https://www.semanticscholar.org/paper/739cf040ed2c2af49077db48d489a46be5fb6157)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F739cf040ed2c2af49077db48d489a46be5fb6157%3Ffields%3DcitationCount)

  2023    Annual Meeting of the Association for Computational Linguistics 

  The author presents FinPythia-6.9B , which uses  the continual pre-training strategy to train Pythia  in a downsteam finance domain . 

  the model is Pythia





- **Gradient Localization Improves Lifelong Pretraining of Language Models**

  [`semanticscholar`](https://www.semanticscholar.org/paper/1e5a61b0ce26662a855d3c9b0ceb76e9a2e1bd8c)  [`Paper`](https://www.semanticscholar.org/paper/1e5a61b0ce26662a855d3c9b0ceb76e9a2e1bd8c)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F1e5a61b0ce26662a855d3c9b0ceb76e9a2e1bd8c%3Ffields%3DcitationCount)

  ​     Gradient norms during pretraining reveal that certain layers of LLMs are more critical for learning new or temporally updated information.  Demonstrates that targeting gradient-dominant layers improves both knowledge retention and acquisition.

  ​	the model is GPT 2

![image-20250114140735824](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250114140735903.png)





- **HellaSwag: Can a Machine Really Finish Your Sentence?**

  [`semanticscholar`](https://www.semanticscholar.org/paper/8b0f27bb594b1eaaf493eaf1e2ee723a2b0a19ad)  [`Paper`](https://www.semanticscholar.org/paper/8b0f27bb594b1eaaf493eaf1e2ee723a2b0a19ad)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F8b0f27bb594b1eaaf493eaf1e2ee723a2b0a19ad%3Ffields%3DcitationCount)

  ​     

  The author use Adversarial Filtering (AF) methods which iteratively select an adversarial set of machine-generated wrong answers to form a new dataset HellaSwag. Though its questions are trivial for humans (>95% accuracy), state-of-the-art models struggle (<48%).

  ![image-20250114141606589](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250114141606637.png)



- **How Do Large Language Models Acquire Factual Knowledge During Pretraining?**

  [`semanticscholar`](https://www.semanticscholar.org/paper/0244faca33c8b4105daf8617e2b0db20cad511bb)  [`Paper`](https://www.semanticscholar.org/paper/0244faca33c8b4105daf8617e2b0db20cad511bb)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F0244faca33c8b4105daf8617e2b0db20cad511bb%3Ffields%3DcitationCount)

  method:  Injected synthetic factual knowledge into pretraining data to analyze memorization and generalization capabilities.

  Conclusion: Factual knowledge is acquired incrementally with each minibatch update during pretraining. The effectiveness of factual knowledge acquisition does not consistently improve with longer pretraining or additional data.

  

  ​     ![image-20250114143703916](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250114143703962.png)

  ![image-20250114144711203](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250114144711263.png)



- **Improving Multimodal Large Language Models Using Continual Learning**

  [`semanticscholar`](https://www.semanticscholar.org/paper/83282cfd95e17c2a6fd70a9383687c9ba3fb3c62)  [`Paper`](https://www.semanticscholar.org/paper/83282cfd95e17c2a6fd70a9383687c9ba3fb3c62)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F83282cfd95e17c2a6fd70a9383687c9ba3fb3c62%3Ffields%3DcitationCount)

  2024    arXiv.org 

​	Train a LLM into MLLM. First align the embeddings from the visual encoder with the text embeddings of the LLM using a two-layer MLP. Then froze the visual encoder and train the LLM with MLP in continual setting using vison-language task.  

The findings emphasize that continual learning methods, when applied to MLLMs, can successfully mitigate linguistic forgetting while preserving vision-language task performance. 



![image-20250114202530020](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250114202530071.png)

This picture compares the performance of the base unimodal LLM (language-only), the original LLaVA 1.5 (naive fine-tuning), and the LLaVA 1.5 trained with the best CL method.



- **Investigating Continual Pretraining in Large  Language Models: Insights and Implications**

  [`semanticscholar`](https://www.semanticscholar.org/paper/12358df20ccf4085e6c8a45d3ab5fa15714abcd6)  [`Paper`](https://www.semanticscholar.org/paper/12358df20ccf4085e6c8a45d3ab5fa15714abcd6)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F12358df20ccf4085e6c8a45d3ab5fa15714abcd6%3Ffields%3DcitationCount)

  2024    arXiv.org 

  1. CL improved downstream performance when domains shared semantic similarities. 
  2. Randomized order optimized backward and forward transfer.

  The two results reflect a **trade-off between specialization and generalization**:

  - **Semantic Similarity**: Encourages **domain-specific specialization** by leveraging relatedness, which boosts performance in closely related downstream tasks.
  - **Randomized Order**: Promotes **generalization and retention** across a wide variety of domains, even those that are unrelated, by balancing exposure to diverse knowledge.

![image-20250114212142409](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250114212142458.png)





- **Investigating the Catastrophic Forgetting in Multimodal Large Language Models**

  [`semanticscholar`](https://www.semanticscholar.org/paper/a281094d05e96b7cca044fdd87ff7c3c65649e20)  [`Paper`](https://www.semanticscholar.org/paper/a281094d05e96b7cca044fdd87ff7c3c65649e20)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fa281094d05e96b7cca044fdd87ff7c3c65649e20%3Ffields%3DcitationCount)

  2023    arXiv.org 

  Purpose:

​	Methods: Evaluating MulTimodality for evaluating the catastrophic forgetting in MLLMs, by treating each MLLM as an image classifier. The paper use another LLM to justify.  

​	Conclusion: All tested MLLMs exhibited catastrophic forgetting compared to their vision encoder performance.

​	Limitation: But the author only study from the image classification task which is too easy to test the MLLM's abilaity.



![image-20250116100128910](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250116100128962.png)

![image-20250116110459161](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250116110459230.png)





- **Large language models encode clinical knowledge**

  [`semanticscholar`](https://www.semanticscholar.org/paper/6052486bc9144dc1730c12bf35323af3792a1fd0)  [`Paper`](https://www.semanticscholar.org/paper/6052486bc9144dc1730c12bf35323af3792a1fd0)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F6052486bc9144dc1730c12bf35323af3792a1fd0%3Ffields%3DcitationCount)

  ​     1 This paper propose the MultiMedQA, a benchmark combining six existing medical question answering dataset.

  2  Introduction of Med-PaLM, achieving near-clinician-level performance on specific axes

  

  ![image-20250116113616849](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250116113616914.png)



- **LINEARLY MAPPING FROM IMAGE TO TEXT SPACE**

  [`semanticscholar`](https://www.semanticscholar.org/paper/c4cb3f7056f1216c1ddfbe4b9e55cbc07a1e43b9)  [`Paper`](https://www.semanticscholar.org/paper/c4cb3f7056f1216c1ddfbe4b9e55cbc07a1e43b9)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fc4cb3f7056f1216c1ddfbe4b9e55cbc07a1e43b9%3Ffields%3DcitationCount)

  ​     Demonstrates that visual representations can be transferred to text space with a simple linear projection.

  

  ![image-20250116131242052](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250116131242115.png)



- **Mitigating Catastrophic Forgetting in Language Transfer via Model Merging**

  [`semanticscholar`](https://www.semanticscholar.org/paper/2018a64911680af735172e3bab2719a80927279f)  [`Paper`](https://www.semanticscholar.org/paper/2018a64911680af735172e3bab2719a80927279f)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F2018a64911680af735172e3bab2719a80927279f%3Ffields%3DcitationCount)

  ​     

  2024    Conference on Empirical Methods in Natural Language Processing 

  

  iteratively merging multiple models, fine-tuned on a subset of the available training data.

  

  ![image-20250116132406759](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250116132406816.png)







- **Overcoming the Stability Gap in Continual Learning**

  [`semanticscholar`](https://www.semanticscholar.org/paper/7d347d3c06b4a7d0292560a9409500e493aed1e9)  [`Paper`](https://www.semanticscholar.org/paper/7d347d3c06b4a7d0292560a9409500e493aed1e9)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F7d347d3c06b4a7d0292560a9409500e493aed1e9%3Ffields%3DcitationCount)

  2023    arXiv.org 

​	invest the learning and forgetting in the process of training in Continual learning setting. 

![image-20250117103024452](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250117103024535.png)







- **Select and Distill:  Selective Dual-Teacher Knowledge Transfer for  Continual Learning on Vision-Language Models**

  [`semanticscholar`](https://www.semanticscholar.org/paper/ddcbdb923f0f76f7b64e3bc8adee78240cfe03ad)  [`Paper`](https://www.semanticscholar.org/paper/ddcbdb923f0f76f7b64e3bc8adee78240cfe03ad)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fddcbdb923f0f76f7b64e3bc8adee78240cfe03ad%3Ffields%3DcitationCount)

  2024    European Conference on Computer Vision 

  study from both two teacher is better than one 

  

  ![image-20250117151002790](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250117151002856.png)



![image-20250117151044352](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250117151044422.png)



![image-20250117151411254](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250117151411300.png)



- **Towards Effective and Efficient Continual Pre-training of Large Language Models**

  [`semanticscholar`](https://www.semanticscholar.org/paper/df448a66ddc961aaec74244b3d2ef6cfb792d1c1)  [`Paper`](https://www.semanticscholar.org/paper/df448a66ddc961aaec74244b3d2ef6cfb792d1c1)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fdf448a66ddc961aaec74244b3d2ef6cfb792d1c1%3Ffields%3DcitationCount)

  2024    arXiv.org 

  high quantity data can improve data performance.

  





## Subspace



PROGRESSIVE PROMPTS: CONTINUAL LEARNING FOR  LANGUAGE MODELS

**PROGRESSIVE PROMPTS: CONTINUAL LEARNING FOR  LANGUAGE MODELS**

- [`semanticscholar`](https://www.semanticscholar.org/paper/86478f285356b5c8d27423e6b939634d9e010fba)  [`Paper`](https://www.semanticscholar.org/paper/86478f285356b5c8d27423e6b939634d9e010fba)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F86478f285356b5c8d27423e6b939634d9e010fba%3Ffields%3DcitationCount)

2023    International Conference on Learning Representations 



use a  fixed prompt for  former task. Prompts trains a separate prompt for each encountered task without modifying its parameters when new tasks are learned, old tasks do not suffer from forgetting.

![image-20250117200209049](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250117200209125.png)

Using 15 task and 10 different sequence as benchmark

![image-20250117200256558](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250117200256635.png)



![image-20250117200311178](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250117200311247.png)





Orthogonal Gradient Descent for Continual Learning

**Orthogonal Gradient Descent for Continual Learning**

- [`semanticscholar`](https://www.semanticscholar.org/paper/841c970f7ef35e28dbbe054d0a7c5df252533a4e)  [`Paper`](https://www.semanticscholar.org/paper/841c970f7ef35e28dbbe054d0a7c5df252533a4e)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F841c970f7ef35e28dbbe054d0a7c5df252533a4e%3Ffields%3DcitationCount)

​     propose the OGD method, which means projecting the gradients from new tasks onto a subspace in which the neural network output on previous task does not change and the projected gradient is still in a useful direction for learning the new task.

![image-20250117200557596](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250117200557651.png)







LFPT5: A UNIFIED FRAMEWORK FOR LIFELONG  FEW-SHOT LANGUAGE LEARNING BASED ON PROMPT  TUNING OF T5

- **LFPT5: A UNIFIED FRAMEWORK FOR LIFELONG  FEW-SHOT LANGUAGE LEARNING BASED ON PROMPT  TUNING OF T5**

  [`semanticscholar`](https://www.semanticscholar.org/paper/fa133b4200729a57db96ae50aff8c4a5ff819f43)  [`Paper`](https://www.semanticscholar.org/paper/fa133b4200729a57db96ae50aff8c4a5ff819f43)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Ffa133b4200729a57db96ae50aff8c4a5ff819f43%3Ffields%3DcitationCount)

  2021    International Conference on Learning Representations 





- **Orthogonal Subspace Learning for Language Model Continual Learning**

  [`semanticscholar`](https://www.semanticscholar.org/paper/28fde851680a40fbbc5c6a44bd3ac6f5ca4ad284)  [`Paper`](https://www.semanticscholar.org/paper/28fde851680a40fbbc5c6a44bd3ac6f5ca4ad284)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F28fde851680a40fbbc5c6a44bd3ac6f5ca4ad284%3Ffields%3DcitationCount)

  2023    Conference on Empirical Methods in Natural Language Processing 







- **Is Parameter Collision Hindering Continual Learning in LLMs?**

  [`semanticscholar`](https://www.semanticscholar.org/paper/f852e60dd32dc1f3f1f53ba8f76862f77c5cd8d2)  [`Paper`](https://www.semanticscholar.org/paper/f852e60dd32dc1f3f1f53ba8f76862f77c5cd8d2)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Ff852e60dd32dc1f3f1f53ba8f76862f77c5cd8d2%3Ffields%3DcitationCount)

  2024    arXiv.org 









