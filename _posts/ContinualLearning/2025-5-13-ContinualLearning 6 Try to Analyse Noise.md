---
layout: post
title:  Continual Learning  Try to analyze and understand Noise 1
categories: [Continual Learning,  math, MachineLearning, LoRA ]
description: 
keywords: [Continual Learning, math, MachineLearning, LoRA ]
---

# Continual Learning  Try to Analyze Noise 1



# 经验损失 与泛化误差



## 📌 问题设定（Problem Setting）

我们考虑一个典型的监督学习问题：

- 输入空间 $\mathcal{X}$，输出空间 $\mathcal{Y}$；

- 存在一个**未知的数据生成分布** $\mathcal{D}$，定义在 $\mathcal{X} \times \mathcal{Y}$ 上；

- 给定一个样本集：
  $$
  S = \{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)\} \sim \mathcal{D}^n
  $$
  即 $n$ 个样本是**独立同分布（i.i.d.）**地从 $\mathcal{D}$ 中采样得到的。

- 我们选择一个预测模型 $f$ 来近似从 $x$ 到 $y$ 的映射，$f \in \mathcal{F}$，其中 $\mathcal{F}$ 是假设空间；

- 定义一个**损失函数** $\ell: \mathcal{F} \times \mathcal{X} \times \mathcal{Y} \to \mathbb{R}_{\geq 0}$，度量 $f(x)$ 与真实标签 $y$ 的差异。

------

## 🧮 定义一：经验风险（Empirical Risk）

**经验风险**是指模型 $f$ 在训练样本 $S$ 上的平均损失，也称为训练误差：


$$
\hat{L}_S(f) = \frac{1}{n} \sum_{i=1}^n \ell(f, x_i, y_i)
$$

- 它可以被**直接计算**；
- 衡量了模型在训练数据上的拟合程度。

------

## 🌐 定义二：泛化误差（Generalization Error）

**泛化误差**是指模型 $f$ 在**真实数据分布 $\mathcal{D}$** 下的期望损失，也称为测试误差、期望风险：


$$
L(f) = \mathbb{E}_{(x, y) \sim \mathcal{D}} [\ell(f, x, y)]
$$


- 它**无法被直接观测**，因为我们不知道 $\mathcal{D}$；
- 但它才是我们真正关心的目标，因为它反映模型在**未来数据**上的表现。

------

## 🔗 两者的联系：泛化误差上界



据**大数定律（Law of Large Numbers）**，我们有以下重要关系：

> 当样本 $(x_i, y_i)$ 是从分布 $\mathcal{D}$ 上独立同分布（i.i.d.）采样时，**经验风险 $\hat{L}_n(f)$ 是泛化误差 $L(f)$ 的无偏估计，且在样本数 $n \to \infty$ 时几乎处处收敛于 $L(f)$**。



对于固定模型 $f$，当 $(x_i, y_i) \overset{i.i.d.}{\sim} \mathcal{D}$，
$$
\hat{L}_n(f) = \frac{1}{n} \sum_{i=1}^n \ell(f, x_i, y_i) \xrightarrow{\text{a.s.}} L(f) = \mathbb{E}_{(x, y) \sim \mathcal{D}}[\ell(f, x, y)]
$$
其中 $\xrightarrow{\text{a.s.}}$ 表示**almost surely convergence**（几乎处处收敛）。

也就是说 

> 随着样本数增大，经验风险 $\hat{L}_n(f)$ 会逐渐逼近泛化误差 $L(f)$。

![image-20250514134727447](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250514134727576.png)



实际上目前大多数机器学习做的事情就是 在给定一个数据的基础上，构建一个模型，然后设计一个损失函数，来最小化经验风险。也就是说我们的目标如下
$$
\min_{f \in \mathcal{F}}\hat{L}_S(f) =  \min_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^n \ell(f, x_i, y_i)
$$
我们想用上面目标来近似得到
$$
 \min_{f \in \mathcal{F}} L(f) =  \min_{f \in \mathcal{F}} \mathbb{E}_{(x, y) \sim \mathcal{D}} [\ell(f, x, y)]
$$
但这依旧存在一个问题就是从 经验风险到泛化误差是有差距的，即模型在训练集和测试集的表现可能不一致，而我们希望两者都表现特别好。我们更希望测试集效果好，也就是从训练集到测试集的泛化性。也就是我们优化上面的目标，并不能保证得到下面的结果。

因为如果只是在训练集表现地很好，但是在测试集表现很差，那么这时候就存在了我们所说的 过拟合，即模型实际并没有学到这个能力，只是在采样数据上表现很好，相当于只记住了采样数据。

所以我们的目标还是更希望得到两者的差异。统计学习的核心问题是：我们希望使用 $\hat{L}_S(f)$ 来估计 $L(f)$，并控制两者之间的差距。我们不应该只依赖大数定律，而是一个更确切的表述，也就是进行量化两者的差异。 



我们希望得到下面的表达，如果我们能用数学方法估计 $L(f)$ 的上界，并最小化它，就能更稳健地学习模型。



**通用形式的泛化上界：**
$$
L(f) \leq \hat{L}_S(f) + \underbrace{\text{复杂度惩罚项}}_{\text{dependent on } \mathcal{F}, n, \delta}
$$
不同理论有不同的“复杂度项”：

| 理论框架            | 泛化控制项                                                   |
| ------------------- | ------------------------------------------------------------ |
| VC理论              | $\sqrt{\frac{d \log(n/d)}{n}}$（$d$为VC维）                  |
| Rademacher复杂度    | $\mathcal{R}_n(\mathcal{F})$                                 |
| PAC-Bayes理论       | $\sqrt{ \frac{ \mathrm{KL}(\rho | \pi) + \ln(1/\delta) }{2n} }$ |
| Uniform Convergence | $\sup_{f \in \mathcal{F}}$                                   |

------

## 🎯 为什么我们更关注泛化误差？

因为**泛化误差 $L(f)$ 才是我们模型部署到现实世界时的实际表现**：

- 如果只优化经验风险 $\hat{L}_S(f)$，可能出现过拟合（memorizing training set）；
- 如果我们能用数学方法估计 $L(f)$ 的上界，并最小化它，就能更稳健地学习模型；
- 这就是 PAC、PAC-Bayes 等理论存在的意义：**量化从训练误差到测试误差的偏差**。

------

## 🧠 举个直观例子

假设你准备考试，只练了一套题：

- 你在这套题上得了满分 —— 经验风险为 0；
- 但考试时换了题，你不会 —— 泛化误差高；
- 所以我们更关心：你在**全部题目分布**（$\mathcal{D}$）上的平均表现，而不是仅在练习题上的表现。
- 然后我们希望从 经验误差，也就是你平时做题的表现来衡量你真实的水平，来估计未来考试的表现

------

## ✅ 总结

| 项目       | 经验风险 $\hat{L}_S(f)$           | 泛化误差 $L(f)$                       |
| ---------- | --------------------------------- | ------------------------------------- |
| 定义       | 在训练样本上的平均损失            | 在真实分布 $\mathcal{D}$ 上的期望损失 |
| 是否可观测 | ✅ 是                              | ❌ 否                                  |
| 衡量能力   | 拟合训练数据                      | 推广到未知样本                        |
| 目标价值   | 辅助评估与训练                    | 学习目标的最终指标                    |
| 关系       | $L(f) \leq \hat{L}_S(f) +$ 上界项 | 上界项取决于模型复杂度与样本量        |

> 📌 我们希望在训练时控制 $\hat{L}_S(f)$ 同时控制模型复杂度，从而推导出 $L(f)$ 的可靠上界。





# 使用经验风险估计 泛化误差 -PAC-Bayes 理论 ：看所有模型的平均表现

前述泛化误差的定义假设了对单一模型 $f$ 的分析，而 PAC-Bayes 理论将这一思想扩展到 **模型分布** 层面，从而提升了理论的灵活性与适用性。

我们现在从 **PAC-Bayesian 理论** 的角度出发，详细解释它是如何实现核心目标：

> **利用可观测的经验风险，去估计不可观测的泛化误差，并给出两者之间的差异的度量。**  



**通用形式的泛化上界：**
$$
L(f) \leq \hat{L}_S(f) + \underbrace{\text{复杂度惩罚项}}_{\text{dependent on } \mathcal{F}, n, \delta}
$$


------

## 🎯 一、核心问题：真实数据分布不可见

我们面对的问题是：

- 训练集上我们能计算 **经验风险** $\hat{L}_n(f)$；
- 但我们真正关心的是 **泛化误差** $L(f) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[\ell(f(x), y)]$；
- 然而 $\mathcal{D}$ 是未知的，我们不能直接计算 $L(f)$。

------

## 🧠 二、PAC-Bayes 的关键思想

PAC-Bayesian 理论解决这个问题的方法是：

> 不只评估单个模型 $f$ 的性能，而是评估**一个模型分布 $\hat{\rho}$ 上的加权平均性能**，即考虑**随机选择模型再做预测**的过程。

我们研究如下两个量的差异：

* 平均泛化误差：

  $$
  L(\hat{\rho}) := \mathbb{E}_{f \sim \hat{\rho}} [L(f)]
  $$

* 平均经验风险：

  $$
  \hat{L}_n(\hat{\rho}) := \mathbb{E}_{f \sim \hat{\rho}} [\hat{L}_n(f)]
  $$



PAC-Bayes bound 就是给出：

> **在高概率下，$L(\hat{\rho})$ 不会比 $\hat{L}_n(\hat{\rho})$ 大太多**。



## 定理（PAC-Bayesian 泛化界及其在高斯分布下的KL表达式）

设有一个训练样本集 $S = {(x_1, y_1), \dots, (x_n, y_n)}$，其中样本为从未知数据分布 $\mathcal{D}$ 上独立同分布采样。考虑定义在参数空间 $\mathcal{W}$ 上的先验分布 $P$ 与后验分布 $Q$，给定损失函数 $\ell(w, x, y) \in [0,1]$，定义如下：

- **经验风险（经验误差）**：
  $$
  L_S(w) = \frac{1}{n} \sum_{i=1}^{n} \ell(w, x_i, y_i)
  $$

- **泛化风险（测试误差）**：
  $$
  L_{\mathcal{D}}(w) = \mathbb{E}_{(x, y) \sim \mathcal{D}} [\ell(w, x, y)]
  $$

则根据 McAllester (1999) 和 Dziugaite & Roy (2017) 所建立的 PAC-Bayes 理论，有如下**泛化误差上界**：

------

### 📐 定理 1（PAC-Bayes 泛化界 (kwiller)）



![image-20250514213838893](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250514213839136.png)

![image-20250514214750455](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250514214750628.png)

对任意后验分布 $Q$ 和先验分布 $P$，有：
$$
\mathbb{E}_{w \sim Q} [L_{\mathcal{D}}(w)] \leq \mathbb{E}_{w \sim Q} [L_S(w)] + \sqrt{ \frac{ \mathrm{KL}(Q \| P) + \log \frac{n}{\delta} }{2(n - 1)} }
$$
该不等式以至少 $1 - \delta$ 的概率（对训练集 $S$ 的采样）成立。



- $\mathbb{E}_{w \sim Q} [L_{\mathcal{D}}(w)]$：不可观测的目标，我们真正想知道的；
- $\mathbb{E}_{w \sim Q} [L_S(w)]$：在训练集上可观测；
- $\mathrm{KL}(Q \| P)$：表示你在训练过程中偏离“先验知识”的程度；
- $\sqrt{\cdots}$：**调节项**，控制泛化误差可能高于经验风险的幅度。
- $\delta$ 是容忍失败的概率（即我们希望上界以至少 $1 - \delta$ 的概率成立）；
- $n$ 是样本数

PAC-Bayesian 理论的关键特点是：

- 它不对单个模型 $w$ 给出泛化保证；
- 而是对**模型分布** $Q$ 下的平均测试误差提供上界：

> ✅ 这就实现了我们想要的：“利用经验风险 + 偏离度，估计泛化误差”。

但PAC-Bayesian 的形式不只有上面这一个，还可以得到下面形式

![image-20250514214126588](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250514214157198.png)

![image-20250514213743686](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250514213743884.png)





当我们有了上面表达式之后，我们就有了用数学方法估计 $L(f)$ 的上界，，接下来我们只需要优化并最小化它，就能得到一个不只是在训练集而是更近似全局分布的模型。

如果给定先验概率 $P$ ,   训练数据集 $S$ ， 样本数量$n$ 和 容忍失败的概率 $\delta$ , 我们的优化目标就变成了最小化
$$
\min_{Q}\mathbb{E}_{w \sim Q} [L_S(w)]  +  \mathrm{KL}(Q \| P)
$$
这个公式揭示了 PAC-Bayes 的算法含义：我们要在**经验误差**与**模型复杂度**之间做出平衡。





------

### 📌 特例：先验和后验均为同构高斯分布

若先验 $P$ 和后验 $Q$ 为均值不同、协方差为单位矩阵的同构高斯分布，即：

- $P = \mathcal{N}(\mu_P, \sigma_P^2 I)$
- $Q = \mathcal{N}(\mu_Q, \sigma_Q^2 I)$

则它们之间的**KL 散度**可表示为：
$$
\mathrm{KL}(Q \| P) = \frac{1}{2} \left[ \frac{k \sigma_Q^2 + \|\mu_P - \mu_Q\|_2^2}{\sigma_P^2} - k + k \log \left( \frac{\sigma_P^2}{\sigma_Q^2} \right) \right]
$$
其中 $k$ 表示参数向量的维度。

> **直觉：**如果模型 $Q$ 和先验 $P$ 越接近（即 KL 小），那么我们越能信任训练集上的表现能推广到未知数据上。

------

## 📜 五、推导思路简要概述

PAC-Bayes bound 的推导思路如下：

1. **定义经验风险与泛化误差之间的差值**为随机变量；

2. 使用 Chernoff bound 或 Hoeffding inequality 对所有模型 $f$ 建立偏差概率控制；

3. 将上述控制推广到 **随机选择 $f$ 的情况**（即引入 $\hat{\rho}$）；

4. 使用 Donsker–Varadhan 变换或变换不等式将 KL 散度引入上界；

5. 得到如下不等式：

   $$
   \mathbb{P}_{(X,Y)^n \sim \mathcal{D}^n} \left[ \mathbb{E}_{f \sim \hat{\rho}} L(f) > \mathbb{E}_{f \sim \hat{\rho}} \hat{L}_n(f) + \epsilon \right] \leq \delta  
   $$

   并进一步解出  $\epsilon$  得到显式界限

------

## 🧩 六、优势与解释

- ✅ **数据依赖**：只需要训练集，不用留出验证集就能估计泛化误差；
- ✅ **模型依赖**：KL 散度可看作对模型复杂度的调节（Occam’s razor）；
- ✅ **灵活可控**：通过 $P$ 和 $Q$ 可以引入先验与结构偏好。



## ✅ 、小结

PAC-Bayes 框架提供了如下能力：

> 在训练集上观察模型的经验风险，再结合模型的复杂度（KL 散度），即可对未知分布下的泛化误差给出理论上界。

这使得我们在**缺乏测试集或真实分布可见性时**，依然能够：

- **评估模型的泛化能力**；
- **指导模型选择与正则项设计**；
- **在贝叶斯与频率学派之间建立桥梁**。

------



# PAC-Bayes 理论 和贝叶斯推理 有什么关系？

好的，下面对这段文字进行逐句解释，结合数学与直觉，帮助你理解它所传达的核心思想。

------

## 🧩 本节目标：构建 PAC-Bayesian 与 Bayesian 方法之间的桥梁

这段内容的主要目的是：**展示在使用负对数似然作为损失函数时，PAC-Bayes 泛化界限的最小化问题与贝叶斯边际似然（marginal likelihood）的最大化是等价的**。这是一种非常有价值的理论桥梁，连接了频率学派与贝叶斯学派的建模方式。

------

## 📚 第一步：回顾贝叶斯推理的基本构成

我们假设存在一个**模型参数空间** $\Theta$，对其中的参数 $\theta$，先验分布为 $p(\theta)$。每个 $\theta$ 决定一个**条件概率模型**：
$$
f_\theta(x) \mapsto p(y | x, \theta)
$$
也就是说，模型学习的是在给定输入 $x$ 的条件下，输出 $y$ 出现的概率。

**根据贝叶斯公式，后验分布为：**
$$
p(\theta | X, Y) = \frac{p(\theta) p(Y | X, \theta)}{p(Y | X)}
$$
其中，$p(Y | X) = \int p(\theta) p(Y | X, \theta) d\theta$ 是**边际似然**，即对所有可能参数下观测数据的平均解释能力。

------

## 📉 第二步：将似然函数转化为负对数损失

引入 **负对数似然损失函数**：
$$
\ell_{\text{nll}}(f_\theta, x, y) = - \log p(y | x, \theta)
$$
这样做有两个好处：

1. **损失函数形式统一**，可以与 PAC-Bayes 理论对接；
2. **概率越大 → 损失越小**，形成自然的优化目标。

对应的经验风险为：
$$
\hat{L}_S(\theta) = \frac{1}{n} \sum_{i=1}^n \ell(f_\theta, x_i, y_i) = -\frac{1}{n} \log p(Y \mid X, \theta)
$$
反过来写成指数形式：
$$
p(Y | X, \theta) = \exp\left(-n \hat{L}_S(\theta)\right)
$$
这就把**PAC-Bayes 中的经验损失**与**贝叶斯中的似然函数**联系了起来。

------

## 🎯 第三步：代入到 PAC-Bayes 最优后验公式中

PAC-Bayes 的 Gibbs 最优后验可写 $Q^*$ 为：
$$
Q^*(\theta) = \frac{\pi(\theta) \cdot \exp(-n \cdot \hat{L}_S(\theta))}{Z_{S}}
$$
而我们刚刚已经知道 $\exp(-n \cdot \widehat{L}_{S}(\theta)) = p(Y | X, \theta)$，所以就有：
$$
Q^*(\theta) = \frac{\pi(\theta) \cdot p(Y | X, \theta)}{p(Y | X)} = p(\theta | X, Y)
$$
这说明：**PAC-Bayes  的 Gibbs 的最优后验 $ \hat{\rho}^* $ 与贝叶斯后验是完全一致的**。

------

## 🔁 第四步：将 PAC-Bayes 优化目标转化为边际似然

 PAC-Bayes 的优化目标：
$$
\mathbb{E}_{\theta \sim Q^*} [\hat{L}_S(\theta)] + \frac{1}{n} \mathrm{KL}(Q^* \| P) = -\frac{1}{n} \log p(Y \mid X)
$$


> **最小化 PAC-Bayes 上界 = 最大化边际似然**

这个等价性是非常重要的，它使我们可以从两种学派中任选工具：

- 若你偏好贝叶斯方法，可以把 PAC-Bayes 当作泛化误差的频率上界；
- 若你偏好频率视角，可以用贝叶斯后验来指导模型设计与选择。

------

## 🧾 最后：Corollary 的含义总结

若假设 $-\log p(y | x, \theta) \in [a, b]$（即损失有界），那么 PAC-Bayes Bound 可以用边际似然表示为：

$\mathbb{E}_{\theta \sim \hat{\rho}^*} L_D(\theta) \leq a + \frac{b - a}{1 - e^{a-b}} \left[ 1 - \exp\left( \frac{a n}{p(Y | X)} \right) \right]$

**这个不等式表明**：边际似然越大，右侧上界越小，即泛化误差越小。

------

## ✅ 总结一句话

这段内容展示了 PAC-Bayes 泛化界与贝叶斯边际似然的数学等价性：**用贝叶斯方法进行推理，其实就是在最小化 PAC-Bayes 的泛化上界（前提是使用负对数似然损失）**。这为理解泛化、模型选择与不确定性建模之间的统一提供了坚实基础。



