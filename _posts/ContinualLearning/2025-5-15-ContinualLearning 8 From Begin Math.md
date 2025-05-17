---
layout: post
title:  Continual Learning  Try to analyze and understand Noise 1
categories: [Continual Learning,  math, MachineLearning, LoRA ]
description: 
keywords: [Continual Learning, math, MachineLearning, LoRA ]
---

# Continual Learning  Try to Analyze Noise 1



# 符号表述

我们考虑一个典型的**监督学习问题**，设定如下：

- 输入空间为 $\mathcal{X}$，输出空间为 $\mathcal{Y}$；

- 样本对 $(x, y) \in \mathcal{X} \times \mathcal{Y}$ 服从某个**未知的数据生成分布** $\mathcal{D}$；

- 给定一个**训练样本集**：
  $$
  S = \{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)\} \sim \mathcal{D}^n
  $$
  即 $n$ 个样本是独立同分布采样得到的。

------

我们考虑一个**假设空间** $\mathcal{F}$，其中每个 $f \in \mathcal{F}$ 是从输入到输出的预测模型。

我们定义一个**损失函数**：
$$
\ell: \mathcal{F} \times \mathcal{X} \times \mathcal{Y} \to [0, 1]
$$
其作用是评估某个模型 $f$ 在某个样本 $(x, y)$ 上的预测误差。

例如：

- 若 $\ell(f, x, y) = \mathbf{1}[f(x) \neq y]$，则为 $0$–$1$ 损失；
- 我们将其泛化为任意在区间 $[0, 1]$ 上有界的损失函数。

------

# 📌 定义经验风险与期望风险

对于一个固定的模型 $f \in \mathcal{F}$：

- **期望风险（泛化误差）**为：
  $$
  \mathcal{L}_{\mathcal{D}}(f) := \mathbb{E}_{(x, y) \sim \mathcal{D}}[\ell(f, x, y)]
  $$
  
- **经验风险（训练误差）**为：

- $$
  \mathcal{L}_S(f) := \frac{1}{n} \sum_{i=1}^n \ell(f, x_i, y_i)
  $$

  

#  PAC-Bayesian Model Selection

### 定理形式（PAC-Bayes “model selection folk theorem”）

对于所有的 $f_i \in \mathcal{F}$，以至少 $1 - \delta$ 的概率成立：
$$
\mathcal{L}_{\mathcal{D}}(f_i) \leq \mathcal{L}_S(f_i) + \sqrt{ \frac{1}{2m} \left( \log \frac{1}{P(f_i)} + \log \frac{1}{\delta} \right) }
\tag{1}
$$

------

### 📌 符号解释：

| 符号                             | 含义                                            |
| -------------------------------- | ----------------------------------------------- |
| $\mathcal{L}_{\mathcal{D}}(f_i)$ | 模型 $f_i$ 的泛化误差                           |
| $\mathcal{L}_S(f_i)$             | 模型 $f_i$ 在训练集上的经验误差                 |
| $P(f_i)$                         | 模型 $f_i$ 的先验概率，越小说明越不被“先验信任” |
| $\delta$                         | 置信水平（PAC 保证）                            |
| $m$                              | 训练样本数量                                    |

------



## ✅ 定理直观解释：

这个界限说明：

- 对于任何一个模型 $f_i$，其在**整个数据分布下的泛化误差**不会远离它在训练集上的误差；
- 偏差大小受两部分控制：
  1. $\log \frac{1}{P(f_i)}$：模型越不被先验信任，惩罚越大；
  2. $\log \frac{1}{\delta}$：我们想要越高置信度（越小 $\delta$），代价越大。

------

## 🤖 PAC-Bayes 模型选择算法

由不等式 (1) 引出一种简单的模型选择方法：

**选择一个使上界最小的模型 $f^*$：**
$$
f^* = \arg\min_{f_i \in \mathcal{F}} \left[ \mathcal{L}_S(f_i) + \sqrt{ \frac{1}{2m} \left( \log \frac{1}{P(f_i)} + \log \frac{1}{\delta} \right) } \right] 
\tag{2}
$$
这表明我们不仅仅选择训练误差最低的模型，而是要在训练误差与模型复杂度之间做权衡。复杂度通过**先验分布 $P$** 编码，也就是引入了一种**信息论意义上的 Occam’s Razor（奥卡姆剃刀）**。



# PAC-Bayesian Model Averaging （McAllester 1999）

在上一节我们提到的是**PAC-Bayes 模型选择**，即对每个单独模型 $f \in \mathcal{F}$ 给出一个风险上界。而本文的核心结果是将该框架推广到**概率分布 $Q$ 上的模型平均（model averaging）**。

## 📌 引入概率模型分布（PAC-Bayes 框架）

在 PAC-Bayes 框架下，我们不选定一个具体模型 $f$，而是考虑一个**后验分布 $Q$ 定义在 $\mathcal{F}$ 上**，即：

- 从 $Q$ 中采样一个模型 $f \sim Q$，然后用它进行预测；

- 这样我们定义：

  - **分布 $Q$ 的期望风险**为：
    $$
    \mathcal{L}_{\mathcal{D}}(Q) := \mathbb{E}_{f \sim Q} \left[ \mathcal{L}_{\mathcal{D}}(f) \right] \tag{3}
    $$
    
- **分布 $Q$ 的经验风险**为：
  
- $$
    \mathcal{L}_S(Q) := \mathbb{E}_{f \sim Q} \left[ \mathcal{L}_S(f) \right]  \tag{4}
    $$
  
  

我们还假设存在一个**先验分布 $P$（数据无关）**，也是定义在 $\mathcal{F}$ 上的概率分布，表示我们在观察数据之前对模型的信念。

我们在以下设定下建立主定理：

- $\mathcal{F}$ 为假设空间，$f \in \mathcal{F}$ 为预测函数；
- $P$ 是定义在 $\mathcal{F}$ 上的**先验分布**，在观察数据前给出；
- $Q$ 是在训练集 $S$ 给出后构建的**后验分布**；
- 损失函数 $\ell(f, x, y) \in [0, 1]$ 为有界可测函数；
- 样本集 $S = {(x_1, y_1), \dots, (x_n, y_n)} \sim \mathcal{D}^n$；
- 所有风险都以期望损失表示：
  - $\mathcal{L}_S(Q) = \mathbb{E}_{f \sim Q} \left[ \frac{1}{n} \sum_{i=1}^n \ell(f, x_i, y_i) \right]$；
  - $\mathcal{L}_{\mathcal{D}}(Q) = \mathbb{E}_{f \sim Q} \left[ \mathbb{E}_{(x,y) \sim \mathcal{D}} [\ell(f,x,y)] \right]$。

------

### 📌 KL 散度定义

为了度量后验 $Q$ 偏离先验 $P$ 的程度，我们使用**Kullback-Leibler 散度**：


$$
\mathrm{KL}(Q \| P) := \sum_{f \in \mathcal{F}} Q(f) \cdot \log \frac{Q(f)}{P(f)}  \tag{5}
$$
该项越大，表示模型 $Q$ 对训练集依赖越强（越复杂），在泛化界中起到“正则项”作用。

------

### ⚠️ 修剪（Pruned）分布的技术性定义

为了简化分析，本文仅考虑一类特殊的 $Q$：

> 如果某个模型 $f$ 满足 $Q(f) > 0$，则要求 $P(f) > 0$；否则 $Q(f) = 0$。

即：**后验分布的支持集必须包含在先验分布的支持集内。**

我们称这类 $Q$ 为**修剪分布（pruned distributions）**。这是为了避免在 KL 中出现 $\log \frac{Q(f)}{P(f)}$ 的除以 0 情况。

但理论作者指出：在实际应用中，这个限制**几乎不会影响主流后验分布的效果**，因为：

- 如语言建模或决策树学习等应用中，后验往往集中在少量高容量模型上；
- 这些模型虽然先验 $P(f)$ 非常小，但后验 $Q(f)$ 很大（因为数据支持强）；
- 所以对低概率模型进行“修剪”不会影响大部分 $Q$ 的质量。

------

## ✅ PAC-Bayes 上界定理

> **Theorem 1 (PAC-Bayesian Bound on Expected Loss of $Q$):**

对所有满足上述“修剪条件”的后验分布 $Q$，以至少 $1 - \delta$ 的概率，有：
$$
\mathcal{L}_{\mathcal{D}}(Q) \leq \mathcal{L}_S(Q) + \frac{1}{m - 1} \left[ \mathrm{KL}(Q \| P) + \log \frac{m}{\delta}  \right] 

\tag{6}
$$

> 注意上面公式$(2)$ 和之前的公式 $(1)$的区别 
> $$
> \mathcal{L}_{\mathcal{D}}(f_i) \leq \mathcal{L}_S(f_i) + \sqrt{ \frac{1}{2n} \left( \log \frac{1}{P(f_i)} + \log \frac{1}{\delta} \right) }
> \tag{1}
> $$
> 根据上面的定义$(3), (4)$，公式 $(6)$可以重新表述为
> $$
> \mathbb{E}_{f \sim Q} \left[ \mathcal{L}_{\mathcal{D}}(f) \right] 
> \leq 
> \mathbb{E}_{f \sim Q} \left[ \mathcal{L}_S(f) \right] + \frac{1}{n - 1} \left[ \mathrm{KL}(Q \| P) + \log \frac{1}{\delta} + 4 \log n + 8 \right] 
> 
> \tag{6-2}
> $$
> 我们就可以清晰地对比PAC-Bayesian Model Averaging和之前的模型选择的不同了， 这里描述的在某个分布$Q$下对模型$f$ 从 $\mathcal{F}$进行采样 的表现，而不是仅仅关心一个具体的$f$





### 📖 解释各部分含义：

| 项                             | 含义                                                   |
| ------------------------------ | ------------------------------------------------------ |
| $\mathcal{L}_{\mathcal{D}}(Q)$ | 后验 $Q$ 的期望泛化误差                                |
| $\mathcal{L}_S(Q)$             | 后验 $Q$ 的训练误差（经验风险）                        |
| $\mathrm{KL}(Q | P)$           | 后验对先验的偏离程度                                   |
| $\log \frac{1}{\delta}$        | 控制置信度的项，越小越稳健                             |
| $4 \log n + 8$                 | 来自对期望和采样的 concentration bound，具体推导见原文 |
| $n$                            | 样本数量                                               |

------

### 🧠 直观理解：

这一定理给出的结论是：

> **只要你选定的后验分布 $Q$ 不太复杂（KL 不大），并在训练集上表现好，那么它在整个分布上的泛化误差也可以得到控制。**

这为使用模型分布（而非单个模型）进行预测提供了坚实的理论基础。





