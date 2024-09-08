---
layout: post
title:    Generative AI 03 Diffusion Model
categories: GAN
description:  李宏毅 生成式AI 
keywords: GAN 生成式AI 
---



# Diffusion Model



Difusion Model

[Understanding Diffusion Probabilistic Models (DPMs) | by Joseph Rocca | Towards Data Science](https://towardsdatascience.com/understanding-diffusion-probabilistic-models-dpms-1940329d6048)



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



 CLIP 4

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







## VAE vs Diffusion Model

![image-20231008164605191](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008164605.png)



# 算法

## 1 Training

![image-20231008164750114](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008164750.png)



![image-20231008164831822](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008164831.png)

加入噪声的时候，并不是我们想象中的一步步产生结果

![image-20231008164953244](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008164953.png)

## 2 Sampling

![image-20231008165208757](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008165208.png)



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

![image-20231008171423313](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008171935.png)

![image-20231008171454155](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008171454.png)

![image-20231008171622846](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008171622.png)

## Denoise的目标

![image-20231008171842208](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008171842.png)

![image-20231008171913288](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008171913.png)



## 为什么还要加noise

![image-20231008172126802](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008172126.png)

直接选择几率最大的，可能效果不好

![image-20231008172318299](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008172318.png)

验证效果

![image-20231008172449051](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008172449.png)

# DM for  speech

![image-20231008172555060](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008172555.png)

# for Text

存在问题 文字是离散的，没法加noise

解决方法1，把noise加载 latent space

![image-20231008172758141](C:/Users/asus/AppData/Roaming/Typora/typora-user-images/image-20231008172758141.png)

![image-20231008172749451](C:/Users/asus/AppData/Roaming/Typora/typora-user-images/image-20231008172749451.png)

方法2 其他的 noise Distribution

![image-20231008172808554](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008172808.png)

## nar 方法

![image-20231008173140529](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008173140.png)



其他方法

1 Mask-Predict

在latent space上再进行 auto coder

![image-20231008173213855](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008173213.png)

![image-20231008173318369](C:/Users/asus/AppData/Roaming/Typora/typora-user-images/image-20231008173318369.png)

![image-20231008173342874](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231008173343.png)