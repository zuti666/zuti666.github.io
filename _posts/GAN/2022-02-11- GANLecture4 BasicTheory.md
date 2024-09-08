---
layout: post
title:  GAN Lecture 4 (2018) Basic Theory 
categories: GAN
description:  GAN Lecture 4，李宏毅
keywords: GAN

---

# GAN Lecture 4 (2018):  Basic Theory



## 生成任务目标是找到一个distribution

![image-20231004110203954](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004110204.png)

有一个distribution ,对应的是人脸

gan的任务就是找出这一个distribution



之前生成任务做法

## Maximum Likelihood Estimation

![image-20231004110515532](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004110515.png)



![image-20231004110851346](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004110851.png)



找一个 $\theta$,使得两个domain之间的KL距离最小



mainflod

由于 G	是一个深层次的网络，所以从简单的输入z映射到一个复杂的结果



## Generator 的目标使得 生成结果$P_G$和$P_{data}$越近越好

![image-20231004111331954](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004111332.png)

问题： 我们并不知道$P_G$和$P_{data}$这两个东西具体是什么

## Discriminator  用来度量$P_G$和$P_{data}$

![image-20231004111556072](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004111556.png)

![image-20231004111850271](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004111850.png)

![image-20231004112123642](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004112123.png)

## 数学证明  目标函数 等价于 最大分类的divergence

![image-20231004112514062](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004112514.png)

 ![image-20231004112843188](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004112843.png)

## 把目标结果带入原问题得到的结果就是 JS Divergence

![image-20231004113131805](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004113131.png)

![image-20231004113239643](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004113239.png)

## 最好的Generator是哪一个

 ![image-20231004113916396](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004113916.png)

## min max 问题 

![image-20231004114040737](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004114040.png)

## 怎么求解

![image-20231004114448754](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004114448.png)

![image-20231004114810845](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004114810.png)

![image-20231004115038000](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004115038.png)

问题，其实G变化了，如果太大了，那就不再是原来的divergence 

## 实际上是怎么做的 sample,cross entropy

![image-20231004115317018](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004115317.png)

## 总的Gan算法回顾

![image-20231004115730293](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004115730.png)



## 实际训练中的 Generator 的目标函数

![image-20231004120014617](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004120014.png)

## 直观理解

![image-20231004120232409](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004120232.png)

## 问题

![image-20231004120551087](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004120551.png)

1 最后 D 变成了什么，可能成了一个分类器？
