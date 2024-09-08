---
layout: post
title:  GAN Lecture 6 (2018) GANLecture6 Tips for Improving GAN
categories: GAN
description:  GAN Lecture 6，李宏毅
keywords: GAN

---

# GAN Lecture 6 (2018):   Tips for Improving GAN



https://colab.research.google.com/github/zuti666/generative-models/blob/master/Tips_for_Improving_GAN.ipynb

## 1 在实际中，任何两个manifolds都不会perfectly align.

首先是，为什么生成样本和真实样本很难有不可忽略的重叠部分？

![image-20211114183831684](https://zuti.oss-cn-qingdao.aliyuncs.com/img/image-20211114183831684.png)



但是如果两个分布完全没有重叠的部分，或者它们重叠的部分可忽略，

**理解**：

GAN中的生成器一般是从某个低维（比如100维）的随机分布中采样出一个编码向量，再经过一个神经网络生成出一个高维样本（比如64x64的图片就有4096维)。当生成器的参数固定时，生成样本的概率分布虽然是定义在4096维的空间上，但它本身所有可能产生的变化已经被那个100维的随机分布限定了，其本质维度就是100，再考虑到神经网络带来的映射降维，最终可能比100还小，所以生成样本分布的支撑集就在4096维空间中构成一个最多100维的低维流形，“撑不满”整个高维空间。

“撑不满”就会导致真实分布与生成分布难以“碰到面”，这很容易在二维空间中理解：一方面，二维平面中随机取两条曲线，它们之间刚好存在重叠线段的概率为0；另一方面，虽然它们很大可能会存在交叉点，但是相比于两条曲线而言，交叉点比曲线低一个维度，长度（测度）为0，可忽略。三维空间中也是类似的，随机取两个曲面，它们之间最多就是比较有可能存在交叉线，但是交叉线比曲面低一个维度，面积（测度）是0，可忽略。

从低维空间拓展到高维空间，就有了如下逻辑：因为一开始生成器随机初始化，所以![[公式]](https://www.zhihu.com/equation?tex=P_g)几乎不可能与![[公式]](https://www.zhihu.com/equation?tex=P_r)有什么关联，所以它们的支撑集之间的重叠部分要么不存在，要么就比![[公式]](https://www.zhihu.com/equation?tex=P_r)和![[公式]](https://www.zhihu.com/equation?tex=P_g)的最小维度还要低至少一个维度，故而测度为0。

另一个理解：即使存在重叠，但是我们是一个取样的过程，所以会出现即使出现重叠，但并不能充分表现出来

## TOWARDS PRINCIPLED METHODS FOR TRAINING GENERATIVE ADVERSARIAL NETWORKS  

[[1701.04862\] Towards Principled Methods for Training Generative Adversarial Networks (arxiv.org)](https://arxiv.org/abs/1701.04862)

>
>
>针对第二点提出了一个解决方案，就是对生成样本和真实样本加噪声，直观上说，使得原本的两个低维流形“弥散”到整个高维空间，强行让它们产生不可忽略的重叠。而一旦存在重叠，JS散度就能真正发挥作用，此时如果两个分布越靠近，它们“弥散”出来的部分重叠得越多，JS散度也会越小而不会一直是一个常数，于是（在第一种原始GAN形式下）梯度消失的问题就解决了。在训练过程中，我们可以对所加的噪声进行退火（annealing），慢慢减小其方差，到后面两个低维流形“本体”都已经有重叠时，就算把噪声完全拿掉，JS散度也能照样发挥作用，继续产生有意义的梯度把两个低维流形拉近，直到它们接近完全重合。以上是对原文的直观解释。





## 改进1 LeastSquareGAN

![image-20211114191806698](https://zuti.oss-cn-qingdao.aliyuncs.com/img/image-20211114191806698.png)

鉴别器做的实际是一个二分类问题，判别函数就是Sigmoid函数，但是我们从Sigmoid函数可以看出，当鉴别器具有很好的分类效果时，对应图像的梯度几乎为0，也就是出现了梯度消失的问题。

LSGAN就用线性函数代替Sigmoid函数，将一个分类问题变成了一个拟合问题，根据拟合的结果对样本进行分类。

![image-20231006110711134](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006110711.png)

>[Least_Squares_Generative](https://openaccess.thecvf.com/content_iccv_2017/html/Mao_Least_Squares_Generative_ICCV_2017_paper.html)
>
>公式
>
>![image-20231006111330489](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006111330.png)
>
>
>
>![image-20231004163208778](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004163208.png)
>
>![image-20231006111012049](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006111012.png)
>
>证明 LSGANs and the Pearson χ2 divergence 的关系
>
>![image-20231006110914984](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006110915.png)
>
>![image-20231006110926223](../../../AppData/Roaming/Typora/typora-user-images/image-20231006110926223.png)
>
>![image-20231006110956515](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006110956.png)

## Softmax GAN

[[1704.06191\] Softmax GAN (arxiv.org)](https://arxiv.org/abs/1704.06191)

公式

![image-20231006115804152](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006115804.png)

![image-20231006115817315](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006115817.png)



## 改进2 WGAN



使用Wassersetin Distance 来衡量两个分布的差异

### 介绍 推土机距离 EarthMove Distance1

![image-20211114192610826](https://zuti.oss-cn-qingdao.aliyuncs.com/img/image-20211114192610826.png)



在一维分布时，很形象

但在二维上面，事实上有很多种方法，推土机距离就是定义为穷举所有的方案，距离最小的哪个

![image-20211114192738066](https://zuti.oss-cn-qingdao.aliyuncs.com/img/image-20211114192738066.png)



![image-20211114192832997](https://zuti.oss-cn-qingdao.aliyuncs.com/img/image-20211114192832997.png)

### 正式定义

![image-20211114193646039](https://zuti.oss-cn-qingdao.aliyuncs.com/img/image-20211114193646039.png)



### Earth Mover’s Distance 的优点

![image-20211114194008576](https://zuti.oss-cn-qingdao.aliyuncs.com/img/image-20211114194008576.png)

能够衡量不相交的时候的距离，并且距离越近效果越好



### WGAN 使用 Wassterstein 

![image-20211114194355129](https://zuti.oss-cn-qingdao.aliyuncs.com/img/image-20211114194355129.png)



$D \in 1-Lipschitz$就是要包证收敛

给出 Lipschitz Function 含义

![image-20211114194717452](https://zuti.oss-cn-qingdao.aliyuncs.com/img/image-20211114194717452.png)

在原始的WGAN中为了保持1-lipschitz，使用Weight Clipping只是给出了范围限制。但是这样并没有解决这个问题

在后序的Improved WGAN（WGAN-GP）中，使用显示样本的梯度效小于1来解决1-lipschitz的问题

>[Wasserstein Generative Adversarial Networks (mlr.press)](https://proceedings.mlr.press/v70/arjovsky17a.html)
>
>![image-20231004163325428](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004163325.png)

公式

![image-20231006121857830](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006121857.png)

![image-20231006122009737](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006122009.png)

![image-20231006122038567](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006122038.png)



## Improved WGAN

![image-20211114195347226](https://zuti.oss-cn-qingdao.aliyuncs.com/img/image-20211114195347226.png)

### 对哪里进行限制—中间部分即可

![image-20231004144150938](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004144151.png)

### 怎么进行限制 —接近1

![image-20211114212538050](https://zuti.oss-cn-qingdao.aliyuncs.com/img/image-20211114212538050.png)

>Improved Training of Wasserstein GANs
>
>[Improved Training of Wasserstein GANs (neurips.cc)](https://proceedings.neurips.cc/paper_files/paper/2017/hash/892c3b1c6dccd52936e27cbd0ff683d6-Abstract.html)
>
>wgan存在的问题：
>
>![image-20231006125601067](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006125601.png)
>
>算法流程
>
>![image-20231006125620091](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006125620.png)



还是有问题的，按着WGP

## DragonGAN

> [[dragan\] On Convergence and Stability of GANs (arxiv.org)](https://arxiv.org/abs/1705.07215)
>
> **猜测模型倒塌（**mode collapse**）是由于在非凸情况下出现了局部平衡，作者观测到局部平衡总是在判别函数中的真实数据周围表现出了尖锐的梯度，因此作者提出DRAGAN，在模型中引入梯度惩罚机制以避免局部平衡。**
>
> 模型倒塌的原因前面也介绍过了，这种尖锐的梯度，会使得多个z矢量映射到单个输出x，造成博弈的退化平衡（实际表现出来也就是输入的多组变量都会产生一致的结果），为了减少这种现象，可以对判别器添加惩罚项：
>
> ![image-20231004171402528](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004171402.png)
>
> 这个方法也确实能够提高模型训练的稳定性。这也解释了为什么WGAN能一定程度上解决模型倒塌，进一步的研究，这种机制非常难达到，一旦过度惩罚（over-penalized），判别器则会引入一些噪声，因此更好的惩罚项应该如下设置：
>
> ![image-20231004171409729](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004171409.png)
>
> 最后，出于一些经验性的优化考虑，作者最终所采用的惩罚项为：
>
> ![image-20231004171419026](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004171419.png)



## 改进3 Spectrum Norm

![image-20211114212740068](https://zuti.oss-cn-qingdao.aliyuncs.com/img/image-20211114212740068.png)

>[[Miyato\] Spectral Normalization for Generative Adversarial Networks (arxiv.org)](https://arxiv.org/abs/1802.05957)
>
>限制权重的更新程度
>
>![image-20231004172330289](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231004172330.png)



## 从GAN到WGAN算法的改进

**而改进后相比原始GAN的算法实现流程却只改了四点**：

- 判别器最后一层去掉sigmoid
- 生成器和判别器的loss不取log
- 每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c
- 不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行

![image-20211114213101647](https://zuti.oss-cn-qingdao.aliyuncs.com/img/image-20211114213101647.png)



## 改进4 Energy-based GAN (EBGAN)



![image-20211114213512580](https://zuti.oss-cn-qingdao.aliyuncs.com/img/image-20211114213512580.png)



![image-20231006151259974](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006151300.png)

鉴别器不需要等生成器生成较好的

![image-20211114213739921](https://zuti.oss-cn-qingdao.aliyuncs.com/img/image-20211114213739921.png)

>[[1609.03126\] Energy-based Generative Adversarial Network (arxiv.org)](https://arxiv.org/abs/1609.03126)
>
>
>
>

公式

![image-20231006151426118](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006151426.png)



## MAGAN

[[1704.03817\] MAGAN: Margin Adaptation for Generative Adversarial Networks (arxiv.org)](https://arxiv.org/abs/1704.03817)



![image-20231007182036082](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231007182036.png)

![image-20231007182240512](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231007182240.png)







## BEGAN

[[1703.10717\] BEGAN: Boundary Equilibrium Generative Adversarial Networks (arxiv.org)](https://arxiv.org/abs/1703.10717)

网络架构

![image-20231006152840911](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006152840.png)

公式

![image-20231006152947836](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006152947.png)

![image-20231006153017483](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006153017.png)

## 改进5 Loss-Sensitive GAN

![image-20211114213939497](https://zuti.oss-cn-qingdao.aliyuncs.com/img/image-20211114213939497.png)



>[Loss-Sensitive Generative Adversarial Networks on Lipschitz Densities | SpringerLink](https://link.springer.com/article/10.1007/s11263-019-01265-2)
>
>margain 先来评估距离，



![image-20231006192956184](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006192956.png)

![image-20231006193147428](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006193147.png)

![image-20231006193212886](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006193212.png)

## DCGAN

[[1511.06434\] Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (arxiv.org)](https://arxiv.org/abs/1511.06434)

主要思想

![image-20231006200808497](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006200808.png)

![image-20231006200901230](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20231006200901.png)
