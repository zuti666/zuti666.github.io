---
layout: post
title:  GAN Lecture 5 (2018)  General Framework
categories: GAN
description:  GAN Lecture 5，李宏毅
keywords: GAN

---

# GAN Lecture 5 (2018):  General Framework



## f-divergence

![image-20231004132202310](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004132202.png)

不同的f,会产生不同的 divergence

## Fenchel Conjugate

![image-20231004132346299](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004132346.png)

每一个 f 都有一个 f* 和他对应 ，哪一个t会使得f*(t)最大，

x为定义域，t为自变量，使得f(t)最大

另一种方法，直接绘制 xt-f(x) ,给定一个t , 找出最大的 f*(t)

![image-20231004132814010](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004132814.png)

## 举例说明

![image-20231004132921064](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004132921.png)

## 举例证明 

![image-20231004132938658](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004132938.png)



## 和GAN有关

![image-20231004133456951](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004133457.png)

![image-20231004133704217](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004133704.png)

## 不同的divergence

![image-20231004133730637](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004133730.png)

>[[1606.00709\] f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization (arxiv.org)](https://arxiv.org/abs/1606.00709)
>
>



## Mode Collapse

![image-20231004133834817](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004133834.png)

## Mode Dropping

![image-20231004133935259](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004133935.png)

## 一个远古的猜想  divergence的原因？

![image-20231004134011742](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004134011.png)



