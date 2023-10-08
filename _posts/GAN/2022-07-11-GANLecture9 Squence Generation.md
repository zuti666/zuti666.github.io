---
layout: post
title:  # GAN Lecture 9 (2018):   Sequence Generation
categories: GAN
description:  GAN Lecture 9，李宏毅,Sequence Generation
keywords: GAN

---

# GAN Lecture 9 (2018):   Sequence Generation



## 问题

![image-20231004195730630](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004195730.png)



## 回顾 

![image-20231004200206494](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004200206.png)

# Improving Supervised Seq-to-seq Model

## 方法1 RL

![image-20231004200309937](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004200309.png)

![image-20231004200424315](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004200424.png)

![image-20231004200612231](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004200612.png)

### 回顾 policy gradient 

![image-20231004201245600](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004201245.png)

![image-20231004201537146](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004201537.png)

### 直觉解释

![image-20231004201707978](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004201708.png)

### 实际应用

![image-20231004202141891](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004202141.png)

### 对比

![image-20231004202529959](../../../AppData/Roaming/Typora/typora-user-images/image-20231004202529959.png)

### Alpha Go style training 

![image-20231004202901848](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004202901.png)



## 方法2 GAN

![image-20231004203006715](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004203006.png)

>
>

### Algorithm 

![image-20231004203657514](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004203657.png)

### sequence model

![image-20231004203912250](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004203912.png)

文字生成 是一串 token ,没法微分，

### 怎么求解这个问题

![image-20231004204114224](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004204114.png)

### 方法1 Gumbel-softmax

一个trick 使得不能微分能够微分

### 方法2  

![image-20231004204223924](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004204223.png)

### 存在问题

![image-20231004204422717](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004204422.png)

### 方法3 RL

![image-20231004204813453](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004204813.png)



![image-20231004204949382](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004204949.png)

### 存在问题

![image-20231004205328130](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004205328.png)

采样不够多

### Solution

![image-20231004205510409](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004205510.png)

## 方法3

![image-20231004205610331](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004205610.png)



![image-20231004205647257](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004205647.png)

## More Application 

![image-20231004205901818](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004205901.png)

# Unsupervised Conditional Sequence Generation

![image-20231004210021003](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004210021.png)

## 1 Text Style Transfer

![image-20231004210050622](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004210050.png)

### 做法1  Direct Transformation

![image-20231004210246075](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004210246.png)



把word换成对应的embeding 

### 做法2  Projection to Common Space

![image-20231004210506829](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004210506.png)



## 2 Unsupervised Abstractive Summarization

![image-20231004210924291](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004210924.png)hh

### 回顾

把文章和摘要当作不同的domain

![image-20231004211030101](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004211030.png)

### 做法—类似cycle GAN

![image-20231004211435150](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004211435.png)

### 另一个角度理解

![image-20231004211602883](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004211602.png)

![image-20231004211709011](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004211709.png)

## 	Unsupervised Machine Translation 

![image-20231004212143943](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004212143.png)



## Unsupervised Speech Recognition 

![image-20231004212343190](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004212343.png)
