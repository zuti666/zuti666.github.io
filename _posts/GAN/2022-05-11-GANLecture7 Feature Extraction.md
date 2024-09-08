---
layout: post
title:  # GAN Lecture 7 (2018):   Feature Extraction
categories: GAN
description:  GAN Lecture 7，李宏毅,Info GAN, VAE-GAN, BiGAN
keywords: GAN

---

# GAN Lecture 7 (2018):   Feature Extraction

https://colab.research.google.com/github/zuti666/generative-models/blob/master/Feature_Extraction.ipynb

## Info GAN 要处理的问题 

input 和 output 不同维度上没有什么关系，不是简单的确定关系

![image-20231004181426978](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004181427.png)

## Info GAN 是什么

![image-20231004181819014](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004181819.png)

c 的维度代表某些特征

>[[1606.03657\] InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets (arxiv.org)](https://arxiv.org/abs/1606.03657)



![image-20231006202100728](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006202100.png)

![image-20231006205416967](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006205423.png)

![image-20231006205720267](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006205720.png)

![image-20231006200407995](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006200408.png)





## VAE -GAN

![image-20231004182527526](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004182527.png)



![image-20231006210220678](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006210220.png)

![image-20231006210233830](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006210233.png)







### Algorithm

![image-20231004182949821](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004182949.png)

![image-20231006210246984](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006210247.png)

公式



![image-20231006210354131](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006210354.png)

![image-20231006210511349](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006210511.png)

![image-20231006210532347](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006210532.png)

>[Autoencoding beyond pixels using a learned similarity metric (mlr.press)](http://proceedings.mlr.press/v48/larsen16)

## BiGAN /ALi

![image-20231004183116691](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004183116.png)

### Algorithm

![image-20231004183351746](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004183351.png)

### 原理

![image-20231004183523389](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004183523.png)

### 只learn d/e 与学习 autoencodeer 有什么区别

![image-20231004183752725](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004183752.png)





>[[BiGAN\] Adversarial Feature Learning (arxiv.org)](https://arxiv.org/abs/1605.09782)
>
>
>
>encoder 和 decoder 联手骗过 Discrimniator
>
>![image-20231004184624047](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004184624.png)
>
>
>

![image-20231007114039761](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231007114039.png)

![image-20231004184644231](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004184644.png)



## Triple GAN

![image-20231004190037023](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004190037.png)





>[Triple Generative Adversarial Nets (neurips.cc)](https://proceedings.neurips.cc/paper_files/paper/2017/hash/86e78499eeb33fb9cac16b7555b50767-Abstract.html)
>
>少量label data ,大量 unlabel data , semi-supervised learning
>
>目标是学习一个好的 classifier
>
>![image-20231004190207829](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004190207.png)

公式

![image-20231007120048135](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231007120048.png)

![image-20231007120108874](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231007120108.png)

![image-20231007120536075](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231007120536.png)



## Domain-adversarial training

抽取特征

![image-20231004185043528](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004185043.png)

![image-20231004185241798](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004185241.png)





>[Domain-adversarial](https://www.jmlr.org/papers/volume17/15-239/15-239.pdf)
>
>



## Feature Disentangle 

Encoder 和 Decodeer 中间抽取的特征并不一定是自己想要的信息，并且不清楚特征对应什么

我们想要对应不同东西对应的特征

![image-20231004185636668](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004185636.png)

### 怎么做—1控制不同特征

![image-20231004185724281](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004185724.png)

### 怎么做 ，2训练一个分类器 GAN的思想

![image-20231004185801904](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004185801.png)



>[Speaker-Invariant Training Via Adversarial Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461932)
>
>![image-20231004192009727](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004192009.png)