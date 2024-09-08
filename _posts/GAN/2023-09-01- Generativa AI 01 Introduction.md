---
layout: post
title:   Generative AI 01 Introduction
categories: GAN
description:  李宏毅 生成式AI 
keywords: GAN 生成式AI 
---



# 图像生成 



# 本质的上的共同目标

![image-20231008165344469](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008165344.png)



![image-20231008165411198](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008165411.png)



# 常见的图像生成模型

![image-20231008160345966](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008160346.png)

影响生成模型就是根据一个Normal Distribution 对应得到影响

![image-20231008160430880](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008160430.png)



## VAE	

decoder 就是学习这个对应关系

encoder 是将图片生成code

然后把decoder和encoder 对应起来，使得输入和输出对应

注意，这样并不能保证，所以要强制中间产生的code的分布要像一个 Normal Distribution

![image-20231008160645281](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008160645.png)



## Flow-based 

先得到一个encoder ,使得图片对应的code一个 Normal Distribution

然后限制这个encoder是可逆的，

这里限制encoder是可逆的

并且图片和code维度应该是一致的，因为要保证可逆关系

![image-20231008160835587](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008160835.png)

## Diffusion Model

把一张图片一直加noise，直到得到一个和 normal distribution差不多的结果

生图片就是逆过程，学习一个denoise的model，从带有noise的结果还原图片

![image-20231008161215987](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008161216.png)

## GAN

只学习一个decoder，从Normal的噪声输入得到图片

然后学习一个 discriminator，用来分辨生成图片和真实图片

![image-20231008161345170](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008161345.png)



## 差异

![image-20231008161427446](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008161427.png)

![image-20231008161635850](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008161635.png)

# 生成式学习 两种策略



## 策略一 各个击破 autoregressive AR Model

文字 一个一个token

图像 一个一个像素

![image-20231008162021016](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008162021.png)



## 策略二 一次到位 Non- autoregressive NAR model



文字生成，怎么直到输出答案的长度呢

一 固定长度，把无效的去掉

二现决定长度

## 区别

![image-20231008162416078](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008162416.png)

## 结合

![image-20231008162729963](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008162730.png)

![image-20231008162755536](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008162755.png)



# 文字生成图片 模型框架

![image-20231008162923962](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008162924.png)

## 例子

### 例子1 Stable Diffusion 

![image-20231008163016010](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008163016.png)

### 例子2 DALL-E series

![image-20231008163048475](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008163048.png)

### 例子3 Imagen

![image-20231008163120955](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008163121.png)

## 1 文字的 encoder

 对结果影响很大

![image-20231008163207785](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008163207.png)

### FID 衡量图像生成效果

有一个图片分配的model，

然后 生成图片和真实图片的距离

假设两组都是 高斯分布 ，然后计算 Frechet distance

距离越小越好

问题，需要大量的结果才能衡量FID

![image-20231008163452650](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008163452.png)



###  CLIP 4

![image-20231008163610118](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008163610.png)

## 2 decoder

1 如果中间产物是小图，decoder将小图变成大图

![image-20231008163729603](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008163729.png)

2 如果中间是latent Representation,那就训练一个auto encoder

![image-20231008163855766](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008163855.png)

## 3 Generation model

![image-20231008164152826](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008164152.png)

![image-20231008164246691](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008164246.png)

![image-20231008164326649](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008164326.png)



# 数学推导

## 目标

找到一个$\theta$ ，让sample的样本在生成得到的概率分布$P_\theta$中越大越好,（这其实等价于让分布$P_\theta$和$P_{data}$越接近越好）

![image-20231008165643090](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008165643.png)



![image-20231008165825003](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008165825.png)



## VAE

认定生成的分布是一个 高斯分布

![image-20231008170112709](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008170112.png)





## VAE 下界

![image-20231008170216625](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008170216.png)



## DDPM

把这个结果也想象成 高斯分布

![image-20231008170501361](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008170501.png)



![image-20231008170713352](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008170713.png)



## 如何直接计算出来

![image-20231008170817975](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008170818.png)

![image-20231008170849959](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008170850.png)

![image-20231008171021016](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008171021.png)

![image-20231008171252306](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008171252.png)

## 如何最小化

![image-20231008171423313](../../AppData/Roaming/Typora/typora-user-images/image-20231008171423313.png)

![image-20231008171454155](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008171454.png)

![image-20231008171622846](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008171622.png)

## Denoise的目标

![image-20231008171842208](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008171842.png)

![image-20231008171913288](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008171913.png)

