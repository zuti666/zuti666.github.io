---
layout: post
title:  GAN Lecture 3 (2018) Unsupervised Conditional Generation
categories: GAN
description:  GAN Lecture 3，李宏毅
keywords: GAN
---

# GAN Lecture 3 (2018): Unsupervised Conditional Generation



## 问题

风格转换任务，没有label ，只有两堆data,machine学习其中的转换



![image-20231003223132108](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231003223132.png)

## 两类做法

![image-20231003223507212](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231003223507.png)

1 直接做 

差距小，例如颜色，纹理

2 先encoder抽特征，然后decoder根据特征生成

差距大



## 做法一

![image-20231003223559429](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231003223559.png)

问题，G生成梵高风格画作，可以骗过D ，但是所生成的结果与原来的输入X无关

D 判别是否符合梵高风格 ，G 任务不仅仅是生成风格类似的结果，还要保持输入输出一致

### 做法1 无视这个问题，直接做

有可能work的原因，G的输入和输出差不多

> [[1709.00074\] The Role of Minimal Complexity Functions in Unsupervised Learning of Semantic Mappings (arxiv.org)](https://arxiv.org/abs/1709.00074)
>
> 论文解释了，直接学习是有可能的，当输入和输出的domain比较接近的时候，一个浅的网络就能学到，但是当domain差距过大就难以实现了
>
> 当G很sallow，work



### 做法2 找一个pre-train的network

![image-20231003224248786](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231003224248.png)

> [[Yaniv Taigman\] Unsupervised Cross-Domain Image Generation (arxiv.org)](https://arxiv.org/abs/1611.02200)
>
> #### baseline of DTN : 
>
> 两个风险：
>
> 风险1 ：
>
> ![image-20231004103336202](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004103336.png)
>
> 风险2：
> ![image-20231004103350485](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004103350.png)
>
> 用一个$f$函数对输入图像$x$和$G$生成的图像$G(x)$进行判别，要它们属于一个domain，两者的距离越小越好
>
> 
>
> #### DTN后续改进 
>
> 
>
> ![image-20231004104151727](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004104151.png)
>
> 
>
> **改进1** 
>
> 第一个改进是G的网络架构，在f的基础上添加一些层g
>
> ![image-20231004104409486](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004104409.png)
>
> **改进2** 
>
> 第二个是，使用f来衡量x和生成结果g(f(x))的相似度，同时还要保证参考的图像不变
>
> ![image-20231004104254753](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004104254.png)
>
> ![image-20231004104358175](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004104358.png)



### 做法3 CYCLE GAN



![image-20231003224920135](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231003224920.png)

> [[Jun-Yan-Zhu\] Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (arxiv.org)](https://arxiv.org/abs/1703.10593v6)
>
> ![image-20231004131010937](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004131010.png)
>
> 
>
> ![image-20231004125423159](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004125423.png)
>
> ![image-20231004125448576](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004125448.png)
>
> ![image-20231004125514665](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004125514.png)
>
> ![image-20231004125540206](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004125540.png)
>
> ![image-20231005145821855](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005145824.png)





### cycleGAN存在的问题



![image-20231003225231202](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231003225231.png)

> [[Casey Chu \] CycleGAN, a Master of Steganography (arxiv.org)](https://arxiv.org/abs/1712.02950)
>
> 论文提到 cycle Gan 学习过程会学到藏东西



![image-20231003225643414](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231003225643.png)

### DIsco GAN

> [Learning to Discover Cross-Domain Relations with Generative Adversarial Networks (mlr.press)](http://proceedings.mlr.press/v70/kim17a.html)
>
> ![image-20231004130108040](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004130108.png)
>
> ![image-20231004130921000](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004130921.png)
>
> ![image-20231004130932727](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004130932.png)
>
> ![image-20231004130942223](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004130942.png)



### Dual GAN

> [[1704.02510v4\] DualGAN: Unsupervised Dual Learning for Image-to-Image Translation (arxiv.org)](https://arxiv.org/abs/1704.02510v4)
>
> 主要思想
>
> ![image-20231004130748082](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004130748.png)
>
> 公式
>
> ![image-20231005134141037](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005134141.png)
>
> ![image-20231005134214469](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005134214.png)
>
> ![image-20231005134229001](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005134229.png)





### 做法4 Star GAN

![image-20231005195020408](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005195020.png)

![image-20231005195044154](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005195044.png)

![image-20231003230029942](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231003230030.png)

![image-20231003230159087](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231003230159.png)



> 论文
>
> [CVPR 2018 Open Access Repository (thecvf.com)](https://openaccess.thecvf.com/content_cvpr_2018/html/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.html)
>
> 公式
>
> adversarial Loss
>
> ![image-20231005195324992](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005195325.png)
>
> Domain Classification Loss
>
> ![image-20231005195436358](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005195436.png)
>
> ![image-20231005195546687](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005195546.png)
>
> Reconstruction Loss
>
> ![image-20231005200447658](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005200447.png)
>
> Full Object
>
> ![image-20231005200522677](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005200522.png)

## 做法二

![image-20231003230507079](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231003230507.png)

问题：两者没有联系

![image-20231003230630972](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231003230631.png)

### 方法1 共享参数

![image-20231003231048031](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231003231048.png)

Couple GAN

[Coupled Generative Adversarial Networks (neurips.cc)](https://proceedings.neurips.cc/paper_files/paper/2016/hash/502e4a16930e414107ee22b6198c578f-Abstract.html)

![image-20231005202121161](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005202121.png)

UNIT 

[Unsupervised Image-to-Image Translation Networks (neurips.cc)](https://proceedings.neurips.cc/paper_files/paper/2017/hash/dc6a6489640ca02b0d42dabeb8e46bb7-Abstract.html)



### 方法2  加一个domain的discriminator

![image-20231003231449189](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231003231449.png)

[[1711.00043\] Unsupervised Machine Translation Using Monolingual Corpora Only (arxiv.org)](https://arxiv.org/abs/1711.00043)

![image-20231005205028172](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005205028.png)

![image-20231005205212942](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005205212.png)

![image-20231005205225996](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005205226.png)



公式

DENOISING AUTO-ENCODING

![image-20231005205413081](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005205413.png)

 CROSS DOMAIN TRAINING

![image-20231005205435987](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005205436.png)

ADVERSARIAL TRAINING

![image-20231005205536835](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005205536.png)

**Final Objective function**

![image-20231005205555067](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005205555.png)



### 方法3  Cycle Consistency 

![image-20231003231742559](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231003231742.png)



[CVPR 2018 Open Access Repository (thecvf.com)](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Anoosheh_ComboGAN_Unrestrained_Scalability_CVPR_2018_paper.html)

![image-20231005212316433](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005212316.png)

![image-20231005212153110](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005212153.png)

### 方法4 semantic consistency

![image-20231003231832215](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231003231832.png)

[[1611.02200\] Unsupervised Cross-Domain Image Generation (arxiv.org)](https://arxiv.org/abs/1611.02200)

[XGAN: Unsupervised Image-to-Image Translation for Many-to-Many Mappings | SpringerLink](https://link.springer.com/chapter/10.1007/978-3-030-30671-7_3)



![image-20231005215131488](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005215131.png)

公式

Total Loss

![image-20231005215405074](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005215405.png)

*Reconstruction loss*

![image-20231005215438578](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005215438.png)

*Domain-adversarial loss*

![image-20231005215502045](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005215502.png)

*Semantic consistency loss*

![image-20231005215524332](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005215524.png)

 generative adversarial loss

![image-20231005215600410](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005215600.png)

*Teacher loss*

![image-20231005215620444](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231005215620.png)
