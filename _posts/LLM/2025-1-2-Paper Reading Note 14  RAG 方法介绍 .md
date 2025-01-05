---
layout: post
title:  Paper Reading 14 RAG- knnLM
categories: [Paper Reading, RAG, LLM, ] 
description:  [Retrieval-Augmented Generation for  AI-Generated Content: A Survey]
keywords: [Paper Reading, RAG, LLM,  ] 
---



# Paper Reading 14 RAG- 方法介绍



综述中提到的不同的RAG 范式 的几篇代表作的介绍，主要介绍其大概思想，不仔细阅读



# RAG fountation



Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks



- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**  

  2020   [`semanticscholar`](https://www.semanticscholar.org/paper/58ed1fbaabe027345f7bb3a6312d41c5aac63e22)  [`Paper`](https://arxiv.org/pdf/2005.11401.pdf)  `arXiv`   ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=$.citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F58ed1fbaabe027345f7bb3a6312d41c5aac63e22%3Ffields%3DcitationCount)



![image-20250104214422274](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250104214422328.png)



## Query-based RAG	

REALM  REALM: Retrieval-Augmented Language Model Pre-Training



- **REALM: Retrieval-Augmented Language Model Pre-Training**  

  2020   [`semanticscholar`](https://www.semanticscholar.org/paper/832fff14d2ed50eb7969c4c4b976c35776548f56)  [`Paper`](https://arxiv.org/pdf/2002.08909.pdf)  `arXiv`   ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F832fff14d2ed50eb7969c4c4b976c35776548f56%3Ffields%3DcitationCount)

![image-20250104203945122](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250104203945175.png)





## Latent Representation-based RAG

RETRO  Improving Language Models by Retrieving from Trillions of Tokens



-  RETRO **Improving Language Models by Retrieving from Trillions of Tokens**  

  None   [`semanticscholar`](https://www.semanticscholar.org/paper/002c256d30d6be4b23d365a8de8ae0e67e4c9641)  [`Paper`](https://www.semanticscholar.org/paper/002c256d30d6be4b23d365a8de8ae0e67e4c9641)     ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F002c256d30d6be4b23d365a8de8ae0e67e4c9641%3Ffields%3DcitationCount)

![image-20250104204254204](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250104204254266.png)



## Logit-based RAG

kNN-LMs  GENERALIZATION THROUGH MEMORIZATION:  NEAREST NEIGHBOR LANGUAGE MODELS



- **GENERALIZATION THROUGH MEMORIZATION:  NEAREST NEIGHBOR LANGUAGE MODELS**  

  kNN-LMs   2019   [`semanticscholar`](https://www.semanticscholar.org/paper/7be8c119dbe065c52125ee7716601751f3116844)  [`Paper`](https://arxiv.org/pdf/1911.00172.pdf)  `arXiv`   ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F7be8c119dbe065c52125ee7716601751f3116844%3Ffields%3DcitationCount)



这里knn 用来检索相似问题的答案，并根据距离转为概率， 然后在最终输出结果的时候，将retrieval的概率输出与generator的概率输出进行融合得到最终结果。由于是在概率层面进行融合，所以不同于上述两种分别基于文本和特征的融合。

![image-20250104204437557](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250104204437609.png)









## Speculative RAG

coG Copy Is All You Need  



- **Copy Is All You Need**  

  coG 2023   [`semanticscholar`](https://www.semanticscholar.org/paper/8b25d0065d30ed3c9e6a6cae94de53ef132d656d)  [`Paper`](https://arxiv.org/pdf/2307.06962.pdf)  `arXiv`  

  International Conference on Learning Representations  ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F8b25d0065d30ed3c9e6a6cae94de53ef132d656d%3Ffields%3DcitationCount)



formulate text generation as progressively copying text segments (e.g., words or phrases) from an existing text collection.

![image-20250104212223008](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250104212223066.png)

这个方法很邪典呀，我们不生产文章，我们只是文字的搬运工，哈哈哈。让我想起了自己写东西的经历，东拼西凑。



RETRIEVAL IS ACCURATE GENERATION



- **RETRIEVAL IS ACCURATE GENERATION**  

  2024   [`semanticscholar`](https://www.semanticscholar.org/paper/9bbcc6eb7ab49ebed302118a98c9e28ea88987b2)  [`Paper`](https://arxiv.org/pdf/2402.17532.pdf)  `arXiv`  International Conference on Learning Representations  ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F9bbcc6eb7ab49ebed302118a98c9e28ea88987b2%3Ffields%3DcitationCount)

 selects contextaware phrases from a collection of supporting documents. 这个思路和上面是一致的，区别是 训练过程，初始化之后使用强化学习进行训练，我只能说好家伙

![image-20250104212757524](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250104212757582.png)



![image-20250104212812285](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250104212812332.png)

这个例子很奇怪呀，这个词段的链接还好，但是如果想要出现一些全新的是不是有些困难，还是说就不存在所谓全新的表达方式



# RAG Enhancements

## Input Enhancement

### Query Transformation:

Query2doc: Query Expansion with Large Language Models



- **Query2doc: Query Expansion with Large Language Models**

  14 March 2023    Conference on Empirical Methods in Natural Language Processing

  [`semanticscholar`](https://www.semanticscholar.org/paper/ccc772d88c231275f24c4fac9b28bbe0942e1107)  [`Paper`](https://arxiv.org/pdf/2303.07678.pdf)  `arXiv`   ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fccc772d88c231275f24c4fac9b28bbe0942e1107%3Ffields%3DcitationCount)



首先根据问题让LLM生成一些文档，然后这些生成文档就可以作为参考，原因是LLM在训练的时候就使用了大量互联网上的信息

![image-20250104214528234](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250104214528284.png)



**HyDE**  Precise Zero-Shot Dense Retrieval without Relevance Labels

- **Precise Zero-Shot Dense Retrieval without Relevance Labels**

  [`semanticscholar`](https://www.semanticscholar.org/paper/5c32c653735b43a0a8923ca65ac191bd4bf15311)  [`Paper`](https://www.aclanthology.org/2023.acl-long.99.pdf)       ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F5c32c653735b43a0a8923ca65ac191bd4bf15311%3Ffields%3DcitationCount)

  **HyDE** 20 December 2022  Annual Meeting of the Association for Computational Linguistics

首先 生成 一系列的 伪文档，然后在真文档的特征空间周围搜索相似度高的伪文档 

![image-20250104233755672](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250104233755723.png)



**CoVe**  CHAIN-OF-VERIFICATION REDUCES HALLUCINATION  IN LARGE LANGUAGE MODELS


- **CHAIN-OF-VERIFICATION REDUCES HALLUCINATION  IN LARGE LANGUAGE MODELS**

  [`semanticscholar`](https://www.semanticscholar.org/paper/4b0b56be0ae9479d2bd5c2f0943db1906343c10f)  [`Paper`](https://arxiv.org/pdf/2309.11495.pdf)  `arXiv`     ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F4b0b56be0ae9479d2bd5c2f0943db1906343c10f%3Ffields%3DcitationCount)

  **CoVe** 20 September 2023  Annual Meeting of the Association for Computational Linguistics

The expanded queries undergo validation by LLM to achieve the effect of reducing hallucinations.

![image-20250105001228066](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250105001228118.png)









### Data Augmentation

Lift Yourself Up: Retrieval-augmented Text Generation with Self-Memory

- **Lift Yourself Up: Retrieval-augmented Text Generation with Self-Memory**

  [`semanticscholar`](https://www.semanticscholar.org/paper/41b796b026a1d322de6ef0b280d3e2e68eee65bd)  [`Paper`](https://arxiv.org/pdf/2305.02437.pdf)  `arXiv`     ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F41b796b026a1d322de6ef0b280d3e2e68eee65bd%3Ffields%3DcitationCount)

  3 May 2023  Neural Information Processing Systems

既然上面的方法行得通，LLM生成的伪文档包含一些信息，那么把这些信息放在一起岂不就是一个 数据库 。基于上述思路，这篇文章有了一个更大胆的想法，有了那么多伪文档，那可以从中选择一些较好地作为retrieval 的结果提供给后续生成。而上述 生成-挑选的过程可以不断进行，最终会得到一个非常大的数据，来代替固定内容的搜索备选数据库。 生成部分不断生成， 而 选择器是能够训练得到一个好的选择器的。



![image-20250104215705301](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250104215705356.png)



PROMPTAGATOR : FEW-SHOT DENSE RETRIEVAL  FROM 8 EXAMPLES



- **PROMPTAGATOR : FEW-SHOT DENSE RETRIEVAL  FROM 8 EXAMPLES**

   [`semanticscholar`](https://www.semanticscholar.org/paper/e86009d9f9b1cdf083a48d087552bc4153784451)  [`Paper`](https://www.semanticscholar.org/paper/e86009d9f9b1cdf083a48d087552bc4153784451)     

  2022  International Conference on Learning Representations

通过给LLM 提示词来创造数据，从而训练一个 retrieval 。 这里创造数据的过程用到了少量示例 所以说是few-shot 的。

![**image-20250104222354716**](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250104222354768.png)





## Retrieval Enhancement

### Retriever Fine-tuning:



**instruction fine-tuning**   Training language models to follow instructions with human feedback



REPLUG: Retrieval-Augmented Black-Box Language Models



- **REPLUG: Retrieval-Augmented Black-Box Language Models**

  [`semanticscholar`](https://www.semanticscholar.org/paper/07b14c24833400b79978b0a5f084803337e30a15)  [`Paper`](https://www.semanticscholar.org/paper/07b14c24833400b79978b0a5f084803337e30a15)   

  2023    North American Chapter of the Association for Computational Linguistics 



treats LM as a black box and update the retriever model based on the final results.



![image-20250104235359920](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250104235359972.png)

![image-20250104235523772](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250104235523842.png)

![image-20250104235633416](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250104235633466.png)







**PKG**  Augmented Large Language Models with Parametric Knowledge Guiding



**Augmented Large Language Models with Parametric Knowledge Guiding**

[`semanticscholar`](https://www.semanticscholar.org/paper/e0dc8e113dbdd2896fb6420ac93e0b976c47f2a2)  [`Paper`](https://www.semanticscholar.org/paper/e0dc8e113dbdd2896fb6420ac93e0b976c47f2a2)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fe0dc8e113dbdd2896fb6420ac93e0b976c47f2a2%3Ffields%3DcitationCount)

PKG 2023    arXiv.org 



首先在对应领域进行微调，然后 生成对应知识作为背景提供给 generator



![image-20250104234321714](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250104234321773.png)



Structure-Aware Language Model Pretraining Improves Dense Retrieval on Structured Data



- **Structure-Aware Language Model Pretraining Improves Dense Retrieval on Structured Data**

  [`semanticscholar`](https://www.semanticscholar.org/paper/a57b90cfc2eab46b773e65240d4ff910f05f989e)  [`Paper`](https://www.semanticscholar.org/paper/a57b90cfc2eab46b773e65240d4ff910f05f989e)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fa57b90cfc2eab46b773e65240d4ff910f05f989e%3Ffields%3DcitationCount)

  

可以用来检索结构型数据，使用对比学习 和 掩码实体预测来训练 retrieval

![image-20250105000418033](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250105000418080.png)



### Re-ranking:



KARD Knowledge-Augmented Reasoning Distillation for Small Language Models in Knowledge-Intensive Tasks

- **Knowledge-Augmented Reasoning Distillation for Small Language Models in Knowledge-Intensive Tasks**

  [`semanticscholar`](https://www.semanticscholar.org/paper/ebf3a59aacdd9982283d7f41229ee2a93800d6ef)  [`Paper`](https://www.semanticscholar.org/paper/ebf3a59aacdd9982283d7f41229ee2a93800d6ef)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Febf3a59aacdd9982283d7f41229ee2a93800d6ef%3Ffields%3DcitationCount)

  

![image-20250105002451444](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250105002451503.png)







### Recursive Retrieval

Query Rewriting for Retrieval-Augmented Large Language Models



- **Query Rewriting for Retrieval-Augmented Large Language Models**

  [`semanticscholar`](https://www.semanticscholar.org/paper/f743287be3ced6757de7ecb26d03815b22cd737b)  [`Paper`](https://www.semanticscholar.org/paper/f743287be3ced6757de7ecb26d03815b22cd737b)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Ff743287be3ced6757de7ecb26d03815b22cd737b%3Ffields%3DcitationCount)

Rewrite-RetrieveRead

![image-20250105001458083](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250105001458137.png)



Bridging the Preference Gap between Retrievers and LLMs



**Bridging the Preference Gap between Retrievers and LLMs**

[`semanticscholar`](https://www.semanticscholar.org/paper/f65ecb65d00f2e69a49465debfdd78efa0838cec)  [`Paper`](https://www.semanticscholar.org/paper/f65ecb65d00f2e69a49465debfdd78efa0838cec)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Ff65ecb65d00f2e69a49465debfdd78efa0838cec%3Ffields%3DcitationCount)

两头都不动，训练中间的，这要怎么训？强化学习，服了

![image-20250105003343432](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250105003343496.png)



![image-20250105003442198](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250105003442274.png)







### Hybrid Retrieval

ISEEQ: Information Seeking Question Generation Using Dynamic Meta-Information Retrieval and Knowledge Graphs

**ISEEQ: Information Seeking Question Generation Using Dynamic Meta-Information Retrieval and Knowledge Graphs**

[`semanticscholar`](https://www.semanticscholar.org/paper/77d2456630d7b22efe84bffcc7d4ad495ce50a6d)  [`Paper`](https://www.semanticscholar.org/paper/77d2456630d7b22efe84bffcc7d4ad495ce50a6d)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F77d2456630d7b22efe84bffcc7d4ad495ce50a6d%3Ffields%3DcitationCount)

2021    AAAI Conference on Artificial Intelligence 



使用了知识图来进行询问，其次这里提到了一个应用场景就是对话形式来获得有效信息，比如医生对患者的对话。主要结合了两种方法，一种是问题的扩展，另一种是使用知识图来进行搜索

![image-20250104221258437](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250104221258478.png)



![image-20250104221320204](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250104221320259.png)











# RAG application in Image 

## Image Generation

**KNN-Diffusion:** KNN-Diffusion: Image Generation via Large-Scale Retrieval



- **KNN-Diffusion: Image Generation via Large-Scale Retrieval**

  [`semanticscholar`](https://www.semanticscholar.org/paper/a225d5d846ba5110232ed5bb32d54ea742b1c2d4)  [`Paper`](https://www.semanticscholar.org/paper/a225d5d846ba5110232ed5bb32d54ea742b1c2d4)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fa225d5d846ba5110232ed5bb32d54ea742b1c2d4%3Ffields%3DcitationCount)

  

 Retrieves similar images or embeddings to condition diffusion models, enabling zero-shot image stylization and diverse outputs.

首先需要注意到的是这个方法的引用场景是图像编辑或者说是图像生成，所以会涉及到一个diffusion模型，这个模型接受原始image的输入 和编辑prompt 的描述。

 编辑prompt的输入空间来自 CLIP ，由于CLIP 模型下的图像和文本是共享空间的，所以及时训练和推理时候CLIP的输入不同，但输出结果是可以用来和图像特征进行融合输入到Diffsusion model 里面的。

其次这个方法使用到了Knn 方法，但这里的knn 和之前论文的knn 搜索的内容有所不同，这里的knn 是搜索的与输入图像相似的图像。在训练的时候，编辑图像区域  从原始输入 与 最近邻居图像的替代。 

这个文章的一个卖点就是 在训练的时候 不需要对应的图像文本对，而是 只需要图像即可，这一点其实是很神奇的。这一点是怎么做得到呢，其实是利用了CLIP 文本和图像共享空间的特性，把输入的文本提示转换为了图像输入来进行代替。 

然后由于kNN只能搜索相似的图像，因此可以推测图像直接的相似性肯定很高，所以是适合做图像编辑任务的，保留了图像的大部分原始特征。（废话，不是说保留了原始特征，而是说他就根本不能变化，因为他的训练过程只遇到了和他相似的图像）

还有一个想法就是CLIP真的很强大，进行转换效果都这么好。

>我觉得这是很好的思路，可以进行扩展

![image-20250104223500353](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250104223500422.png)



RE-IMAGEN: RETRIEVAL-AUGMENTED  TEXT-TO-IMAGE GENERATOR



- **RE-IMAGEN: RETRIEVAL-AUGMENTED  TEXT-TO-IMAGE GENERATOR**

  [`semanticscholar`](https://www.semanticscholar.org/paper/ec1ac8df419a241c3cc6bfd209a38b494af792ee)  [`Paper`](https://www.semanticscholar.org/paper/ec1ac8df419a241c3cc6bfd209a38b494af792ee)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fec1ac8df419a241c3cc6bfd209a38b494af792ee%3Ffields%3DcitationCount)

  

Given a text prompt, Re-Imagen accesses an external multi-modal knowledge base to retrieve relevant (image, text) pairs and uses them as references to generate the image.

![image-20250104232942323](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250104232942374.png)

![image-20250104232953594](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250104232953647.png)





# RAG application in Knowledge

G-Retriever  

G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering



- **G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering**

  [`semanticscholar`](https://www.semanticscholar.org/paper/a41d4a3b005c8ec4f821e6ee96672d930ca9596c)  [`Paper`](https://www.semanticscholar.org/paper/a41d4a3b005c8ec4f821e6ee96672d930ca9596c)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fa41d4a3b005c8ec4f821e6ee96672d930ca9596c%3Ffields%3DcitationCount)



![image-20250105003129747](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250105003129822.png)





# Evaluation

Benchmarking Large Language Models in Retrieval-Augmented Generation

**Benchmarking Large Language Models in Retrieval-Augmented Generation**

[`semanticscholar`](https://www.semanticscholar.org/paper/28e2ecb4183ebc0eec504b12dddc677f8aef8745)  [`Paper`](https://www.semanticscholar.org/paper/28e2ecb4183ebc0eec504b12dddc677f8aef8745)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F28e2ecb4183ebc0eec504b12dddc677f8aef8745%3Ffields%3DcitationCount)



![image-20250105002639987](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250105002640042.png)
