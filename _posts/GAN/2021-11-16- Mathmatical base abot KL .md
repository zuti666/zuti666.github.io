---
layout: post
title:  从统计学理解KL散度   KL divergence
categories: GAN math
description: 最大似然估计，最大后验估计，KL散度
keywords: GAN math
---

# 最大似然估计


# 贝叶斯派与频率派

## 概率理解

1. 频率派：概率是一个确定的值，模型中的参数也是一个确定的值。样本数据是由确定的概率分布生成的，因此数据是随机的。多次重复试验，使用事件发生的频率近似代替概率 。

   >对于一个模型或者也可说一个分布中的参数，我们相信它是固定不变的，是固有的属性。而我们观察（采样）到的数据是这个分布中的一个**独立同分布样本**。
   >
   >也就是说，不管怎么采样，我们根据采样的数据然后对参数的估计都应该是不会变的。如果根据数据估计出来的参数和真实模型不符合，只可能是引入了噪声而已。
   >
   >另一个问题，很明显，需要大量数据我们才能得到一个更好的结果，但如果观测样本有限呢？那我们认为概率依然是确定的，是你的观测的问题。

2. 贝叶斯派： 把参数θ视作随机变量，而样本X是固定的，其着眼点在参数空间，重视参数θ的分布，固定的操作模式是通过参数的先验分布结合样本信息得到参数的后验分布。 从**观察者的角度**出发，观察者根据观测得到的事实，不断更新修正对模型的认知。

# 统计和概率

概率（probabilty）和统计（statistics）看似两个相近的概念，其实研究的问题刚好相反。

概率研究的问题是，已知一个模型和参数，怎么去预测这个模型产生的结果的特性（例如均值，方差，协方差等等）。 举个例子，我想研究怎么养猪（模型是猪），我选好了想养的品种、喂养方式、猪棚的设计等等（选择参数），我想知道我养出来的猪大概能有多肥，肉质怎么样（预测结果）。

统计研究的问题则相反。统计是，有一堆数据，要利用这堆数据去预测模型和参数。仍以猪为例。现在我买到了一堆肉，通过观察和判断，我确定这是猪肉（这就确定了模型。在实际研究中，也是通过观察数据推测模型是／像高斯分布的、指数分布的、拉普拉斯分布的等等），然后，可以进一步研究，判定这猪的品种、这是圈养猪还是跑山猪等等（推测模型参数）。

**概率是已知模型和参数，推数据。统计是已知数据，推模型和参数**

------------------------------------------------
>版权声明：本文为CSDN博主「nebulaf91」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
>原文链接：https://blog.csdn.net/u011508640/article/details/72815981
>
>



# 贝叶斯公式的理解

我们可以将$P(\theta |X)$​进行贝叶斯概率进行展开
$$
P(\theta | X) = \frac{ P(X| \theta)}{ P(X)} \cdot P(\theta)  \ \ \ (2)
$$
上式中$ P (\theta|X)$​​​称作​后验概率 ， $P(\theta)$​称作先验概率。$P(X|\theta)$叫做似然度，$P(X)$是边缘概率，与待估计参数$\theta$​无关，又叫做配分函数

这个式子就有迭代的含义，即，给一个基于经验的先验概率$P(\theta)$ ,然后基于观测值 $L(\theta|X)$与$P(X)$ 来不断修正对$P(\theta)$的认知，最终得到基于事实的后验概率$P(\theta|X)$。


$$
\begin{aligned}

P(\theta _i| X) 
&= \frac{ P(X| \theta_i)}{ P(X)} \cdot P(\theta_i) \\
&=\frac{P(X|\theta_i) \cdot P(\theta_i)}{\sum P(X|\theta)\cdot P(\theta)} \\
&= \frac{P(X|\theta_i) \cdot P(\theta_i)}{\int P(X|\theta)\cdot P(\theta) d\theta} \\
&=\frac{P(X|\theta_i) \cdot P(\theta_i)}{P(X|\theta_i) \cdot P(\theta_i)+\sum_{k \neq i} P(X|\theta_k)\cdot P(\theta_k)}

\end{aligned}
$$


在上式中，事实上$\theta_k$​​​​的取值是无限多的。计算是不容易的。

在上式中，$P(X)$保证结果是在$(0,1)$​之间，这也就是为什么被称为配分函数。

贝叶斯概率是怎样根据观测得到的数据对先验概率进行修正呢？就是来看观测的数据在所有可能性中占比是更大还是更小。



# 从贝叶斯角度理解梯度下降法

![image-20211108220514209](https://zuti.oss-cn-qingdao.aliyuncs.com/img/image-20211108220514209.png)





# 似然函数

对于这个函数：

$P ( x ∣θ )$

输入有两个：$x$表示某一个具体的数据； $ \theta$表示模型的参数。

如果 $\theta$是已知确定的，$x$​是变量，这个函数叫做概率函数(probability function)，它描述在固有属性$\theta$下，对于不同的样本点$x$​，其出现概率是多少。

**如果$x$是已知确定的， $\theta$​是变量，这个函数叫做似然函数(likelihood function), 它描述对于不同的模型参数，出现$x$这个样本点的概率是多少**。

> 版权声明：本文为CSDN博主「nebulaf91」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
> 原文链接：https://blog.csdn.net/u011508640/article/details/72815981

# 最大似然估计 MLE——频率派

在“**模型已定，但模型参数未知**”的情况下，基于你观测的数据来估计模型参数的一种思想或者方法。换句话说，解决的是**取怎样的模型参数可以使得产生已得观测数据的概率最大的问题** 。



$P(\theta|X)$​​即事件$X$​​发生后，这个概率值$\theta$​​​​​​是多少。



事实上，这个$\theta$​​的取值可以是任意的，也就是说似然函数$L_x(\theta)$​​是关于$\theta$​​​​​的一个函数。我们习惯于将似然函数写作$L(\theta| X)$​​ ,表示$X$​​发生后的似然值$\theta$​​ ,这里的$\theta$​​​不是概率，而是一个确定的值​​，是我们想要的是$\theta$​​是样本的固有属性。我们观测到的$X$​​是在事件固有属性$\theta$​​​​支配下发生的，也就是$P(X|\theta)$​​​,于是根据事件固有的属性我们有下式成立。


$$
L_x(\theta) = L(\theta|X)  = P(x|\theta)         \ \ \ (1)
$$


也就是说我们希望得到$\theta$值就是样本的固有属性。

但我们**只能根据我们观测的数据去推断**。我们相信**我们所观测到的现象就是事物表现出来的属性**（**即$P(\theta|X)=P(X|\theta)$​​​​​​​​** 。我们其实知道这个是不准确的，即($P(\theta|X) \neq P(X|\theta) $​)，比如投掷硬币，由于时间或者各种因素我们只能观测10次，通过这十次我们计算出正面朝上的频率是$7/10$，我们便认为投硬币正面朝上的概率就是$70\%$​​​​​​​​​​​​​。

**这就是频率派的思想，认为存在这样的固有属性，我们观测得到的结果就是由于这个固有属性产生的结果且当重复试验次数n次后我们可以用观测得到的样本的频率去表示事物本身的概率。**



我们**第二个假设**是**人为规定当前观测情形下的最大似然值$L(\theta|X)$就是这个概率$\theta$($\theta = arg _{*\theta}\ \ max L(\theta|X)$​​​ )**。怎么理解，就我们相信自己的观测结果出现就是事物本质的直接表现，认为只有事物出现这个观测的概率最大，所以我才能观测得到。

**既然有无数种分布可以选择,那让观测得到样本结果出现的可能性最大**



# 最大后验估计 MAP——贝叶斯派



由上式$(1)$我们知道似然函数$L(\theta|X)$​等于在固有属性$\theta$ 下 $X$的发生概率 $P(X|\theta)$​​ ,将其带入(2)，得到
$$
P(\theta | X) = \frac{ L( \theta |X )}{ P(X)} \cdot P(\theta) \ \ \ (3)
$$
在（3）式中,$L(\theta | X)$​​ 称为 似然度。

在（3）式中，我们要求的就是$\theta$​ ,不妨将其记为一个关于$\theta$的函数$f_x(\theta)$
$$
f_x(\theta) := P(\theta | X) = \frac{ L( \theta |X )}{ P(X)} \cdot P(\theta)
$$
和上面类似我们是想求$\theta$​​ , **我们使用$f_x(\theta)$​​取得最大值时的$\theta$​来代替**​​​​。我们可以观察式子的右端​，分母$P(X)$​​是与$\theta$​​​无关的，我们想要求最大值，只需求$L(\theta|X) \cdot P(\theta)$​的最大值即可。也就得到了我们的最大后延估计MAP



# 最大似然估计MLE 与最大后验估计 MAP

![image-20211108110434282](https://zuti.oss-cn-qingdao.aliyuncs.com/img/image-20211108110434282.png)

![image-20211108110812776](https://zuti.oss-cn-qingdao.aliyuncs.com/img/image-20211108110812776.png)



$ P(w)$​​是先验概率，也就是我们根据经验对于可能的分布的一个猜测。

可以看到，当假设分布服从常数分布时，$ logP(w)$​​​​是一个常数，可以忽略不计，最大后验估计退化为最大似然估计。还有就是我们不认为存在先验概率时，最大后验估计退化为最大似然估计。

当假设分布服从正态分布和拉普拉斯分布，分别得到L2正则化项和L1正则化项

![image-20211108111358800](https://zuti.oss-cn-qingdao.aliyuncs.com/img/image-20211108111358800.png)

# 从极大似然估计到KL散度

https://zhuanlan.zhihu.com/p/266677860

## 1.1 极大似然估计 要解决的问题

1给定一个数据分布 $P_{data(x)}$
2给定一个由参数 $\theta$ 定义的数据分布 $P_G(x;\theta)$
3我们希望求得参数$\theta$ 使得$P_G(x;\theta)$  尽可能接近 $P_{data(x)}$

可以理解成：

$P_G(x;\theta)$ 是某一具体的分布（比如简单的高斯分布），而 $P_{data(x)}$是未知的（或者及其复杂，我们很难找到一个方式表示它），我们希望通过**极大似然估计**的方法来确定 $\theta$ ，让$P_G(x;\theta)$能够大体表达$P_{data(x)}$

1. 从 $P_{data}$ 采样m个样本 $\{x^1,x^2,\dots,x^m\}$
2. 计算采样样本的似然函数 $L=\Pi_{i=1}^m P_G(x^i;\theta)$
3. 计算使得似然函数 $L$ 最大的参数 $\theta: \theta^* = arg \ \underset{\theta}{max} L=arg \ \underset{\theta}{max} \Pi_{i=1}^m P_G(x^i;\theta)$

> 这里再啰嗦一下**极大似然估计**为什么要这么做：
> $P_{data}$ 可以理解成是非常复杂的分布，不可能用某个数学表达精确表示，因此我们只能通过抽象，使用一个具体的分布模型  $P_G(x;\theta)$近似 $P_{data}$
> 所以，求 $P_G(x;\theta)$ 的参数 $\theta$的策略就变成了：
> 我们认为来自  $P_{data}$的样本  $\{x^1,x^2,\dots,x^m\}$ 在  $P_G(x;\theta)$分布中出现的概率越高，也就是 $L=\Pi_{i=1}^m P_G(x^i;\theta)$越大 , $P_G(x;\theta)$和 $P_{data}$ 就越接近。
> 因此，我们期待的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 就是使得 $L=\Pi_{i=1}^m P_G(x^i;\theta) $最大的  $\theta$.
> 即： $ \theta^* = arg \ max_\theta L=arg \ max_\theta \Pi_{i=1}^m P_G(x^i;\theta)$

咱们继续推导：


$$
\begin{aligned}
\theta^*  
&= arg \ \underset{\theta}{max} L \\ 
&=arg \ \underset{\theta}{max}  \Pi_{i=1}^m P_G(x^i;\theta) \\
&=arg \underset{\theta}{max} \ log  \Pi_{i=1}^m P_G(x^i;\theta) \\
&=arg \underset{\theta}{max} \   \sum_{i=1}^m  log P_G(x^i;\theta) \\
& \approx   arg \underset{\theta}{max} E_{x \sim P_{data}} [log P_G(x;\theta)] \\
&= arg \underset{\theta}{max} \int_{x} P_{data}(x)  log P_G(x;\theta) dx \\
&= arg \underset{\theta}{max} \int_{x} P_{data}(x)  log P_G(x;\theta) dx 
- \int_x P_{data}(x)logP_{data}(x)dx \\
&(我们求的是\theta，后面加上的一项与\theta 无关，这是为了将极大似然的式子推导为KL散度的表达)\\
&= arg \ \underset{\theta}{min} KL(P_{data}||P_{G}(x;\theta))
\end{aligned}
$$





> KL散度：
> ![[公式]](https://www.zhihu.com/equation?tex=KL%28P%7C%7CQ%29) 衡量P，Q这两个概率分布差异的方式：
> $KL(P||Q)=\int_x p(x)(log \ p(x)-log\ q(x))$





## **1.3 极大似然估计**的本质

找到 $\theta$ 使得 $P_G(x;\theta)$ 与目标分布 $P_{data}(x)$ 的KL散度尽可能低，也就是使得两者的分布尽可能接近，实现用确定的分布 $P_G(x;\theta)$ 极大似然 $P_{data(x)}$



# 感谢

感谢b站up主王木头的视频讲解



nebulaf91的博客http://blog.csdn.net/u011508640/article/details/72815981

