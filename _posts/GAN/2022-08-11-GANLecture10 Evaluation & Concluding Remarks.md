---
layout: post
title:  # GAN Lecture 10 (2018):   Evaluation & Concluding Remarks
categories: GAN
description:  GAN Lecture 10，李宏毅,Evaluation & Concluding Remarks
keywords: GAN

---

# GAN Lecture 10 (2018):   Evaluation & Concluding Remarks

![image-20231004212804136](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004212804.png)

## 传统方法 —Likelihood 

![image-20231004212947502](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004212947.png)

##  解决方法1 Kernel Density Estimation

![image-20231004213921086](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004213921.png)

问题1 ： sample样本数量应该是多少？

问题2 ： 怎么估测一个生成结果的 分布

问题3 ： Likelihood 也不一定是一个好的度量结果

## 问题： Likelihood 不能反应生成结果的质量

![image-20231004214226429](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004214226.png)

## Objective Evaluation  用一个训练好的classifer来判别结果 

方向一 ，生成结果 能否很好，一张图的质量

方向二 ，diverse的问题，结果要有多个类别，一类图的类别

![image-20231004214508429](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004214508.png)

## Inception Score

![image-20231004214626758](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004214626.png)

## 只算相似度是不够的，尤其是pixel的相似度

![image-20231004215321607](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004215321.png)

## Mode Dropping  生成结果的多样性减少

![image-20231004215407633](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004215407.png)

## Solution  1

![image-20231004215937225](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004215937.png)

## Solution 2  看一群图

![image-20231004220000890](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004220000.png)



![image-20231004220041092](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004220041.png)

## 课堂GAN 总结

![image-20231004220358785](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004220358.png)

![image-20231004220623990](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004220624.png)
