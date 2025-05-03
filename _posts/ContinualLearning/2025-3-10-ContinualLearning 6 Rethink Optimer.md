---
layout: post
title:  Continual Learning  6 LoRA think
categories: [Continual Learning,  math, MachineLearning, LoRA ]
description: 
keywords: [Continual Learning, math, MachineLearning, LoRA ]
---

# Continual Learning  6 LoRA think



一组最优参数 $\theta^*$使得模型$f_\theta(X)$在 数据集$D$上的表现最佳。而在概率统计的框架下，我们可以使用**最大似然估计（Maximum Likelihood Estimation, MLE）**或**贝叶斯推理（Bayesian Inference）**来描述这个优化过程

## 最大似然估计 MLE

在统计建模中，我们假设数据 $D$ 由某个概率分布生成，该分布的参数由 $\theta$ 控制。

因此，我们可以计算**数据在给定参数下的概率**，即 **似然函数 (Likelihood function)**：
$$
p(D | \theta)
$$
**最大似然估计 (MLE) 的目标**是找到最优参数 $\theta^*$，使得 $p(D | \theta)$ 最大化：
$$
\theta^* = \arg\max_{\theta} p(D | \theta)
$$
在深度学习的优化过程中，我们的目标通常是**最小化一个损失函数 (Loss function)**，而不是最大化一个概率。因此，我们可以定义**损失函数** $L(\theta)$ 为：
$$
L(\theta) = - \log p(D | \theta)
$$
这样，最小化 $L(\theta)$（损失最小化）等价于**最大化** $\log p(D | \theta)$（最大化对数似然）：
$$
\arg\min_{\theta} L(\theta) = \arg\max_{\theta} \log p(D | \theta)
$$
这就是为什么我们说：
$$
\log p(D | \theta) = - L(\theta)
$$
**直观理解**：

- **如果模型对数据$D$的拟合度越高（即$p(D | \theta)$越大）**，那么对数似然 $\log p(D | \theta)$ 也越大，损失 $L(\theta)$ 变小，说明模型表现更好。
- **反之，如果$p(D | \theta)$很小**（即模型无法很好地解释数据），那么 $\log p(D | \theta)$ 变小，导致损失 $L(\theta)$ 变大，说明模型拟合效果差。



## 贝叶斯定理 MAE

首先是贝叶斯定理

我们可以将$P(\theta |X)$​进行贝叶斯概率进行展开
$$
P(\theta | X) = \frac{ P(X| \theta)}{ P(X)} \cdot P(\theta)  \ \ \ (2)
$$
上式中$ P (\theta|X)$​​​称作​后验概率 ， $P(\theta)$​称作先验概率。$P(X|\theta)$叫做似然度，$P(X)$是边缘概率，与待估计参数$\theta$​无关，又叫做配分函数

然后对上面进行左右两端同时取对数 
$$
\log P(\theta | X) =  \log P(X| \theta) + \log P(\theta) - \log P(X)
$$

- **$\log p(D | \theta)$**：对数似然，即数据在参数 $\theta$ 下的概率。
- **$\log p(\theta)$**：参数的先验分布，表示我们对 $\theta$ 先验的信念，可以视为正则化项（如 L2 正则化）。
- **$\log p(D)$**：归一化项，与 $\theta$ 无关，在优化时可以忽略。

在优化过程中，我们希望**最大化后验概率**，即 $\log p(\theta | D)$

我们知道似然函数$L(\theta|X)$​等于在固有属性$\theta$ 下 $X$的发生概率 $P(X|\theta)$​​ ,将其带入(2)，得到
$$
P(\theta | X) = \frac{ L( \theta |X )}{ P(X)} \cdot P(\theta) \ \ \ (3)
$$
在上式中,$L(\theta | X)$​​ 称为 似然度。在上式中，我们要求的就是$\theta$​ ,不妨将其记为一个关于$\theta$的函数$f_x(\theta)$

$$
f_x(\theta) := P(\theta | X) = \frac{ L( \theta |X )}{ P(X)} \cdot P(\theta)
$$
和上面类似我们是想求$\theta$​​ , **我们使用$f_x(\theta)$​​取得最大值时的$\theta$​来代替**​​​​。我们可以观察式子的右端​，分母$P(X)$​​是与$\theta$​​​无关的，我们想要求最大值，只需求$L(\theta|X) \cdot P(\theta)$​的最大值即可。也就得到了我们的最大后延估计MAP

![image-20211108110434282](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250314160110835.png)

![image-20211108110812776](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250314160110953.png)



$ P(w)$​​是先验概率，也就是我们根据经验对于可能的分布的一个猜测。

可以看到，当假设分布服从常数分布时，$ logP(w)$​​​​是一个常数，可以忽略不计，最大后验估计退化为最大似然估计。还有就是我们不认为存在先验概率时，最大后验估计退化为最大似然估计。

当假设分布服从正态分布和拉普拉斯分布，分别得到L2正则化项和L1正则化项

![image-20231011104553987](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250314160110054.png)

------

##  Loss Function

1. **在神经网络训练中，我们通常最小化损失函数$L(\theta)$，而损失函数是负的对数似然$-\log p(D | \theta)$**。
2. **最小化损失函数$\arg\min L(\theta)$等价于最大化对数似然$\arg\max \log p(D | \theta)$**，从而让模型更符合数据。
3. **在贝叶斯学习中，优化目标不仅包括对数似然，还包括先验分布项$\log p(\theta)$，从而实现更稳健的学习**。

因此，从概率的角度来看，**深度学习中的损失函数 $L(\theta)$ 实际上就是负的对数似然 $-\log p(D | \theta)$，而优化过程等价于最大化数据的概率**。



# Continual learning

在持续学习的设置中，我们将一个数据集进行划分为多个task ，然后依次进行训练，我们期待这种训练效果要和一起训练 joint traing 的效果一致，接下来，我要进行表示和比较

首先，将一个数据 划分为不同的 task
$$
D= \{ T_1, T_2,T_3,\cdots,\}
$$
下面的思考过程，为了简便起见，我们只考虑划分为两个task，即
$$
D= \{ T_1, T_2\}
$$


##  Joint Traing

Joint Traing 就是把所有的数据都拿来一起训练，和上面的经典方法一模一样 ，我们采取贝叶斯视角
$$
\log P(\theta | X) =  \log P(X| \theta) + \log P(\theta) - \log P(X)
$$
最大化后验概率为
$$
\begin{aligned}
\log P(\theta | D) &=  \log P(D| \theta) + \log P(\theta) - \log P(D)  \\
\log P(\theta | \{ T_1, T_2\}) &=  \log P(\{ T_1, T_2\}| \theta) + \log P(\theta) - \log P(D)  \\


\end{aligned}
$$
如果我们假设 这里划分后的数据 $T_1,T_2$ 与整个模型 $D$是独立同分布的，于是我们可以将$\log P(\{ T_1, T_2\}| \theta)$ 进行分解如下
$$
\log P(\{ T_1, T_2\}| \theta) =   \log P( T_1| \theta) + \log P(  T_2| \theta)
$$
然后将上式带回之前的公式，我们将这样训练后得到的参数为$\theta^ \circ$
$$
\log P(\theta ^  \circ | \{ T_1, T_2\}) =   \log P( T_1| \theta) + \log P(  T_2| \theta) + \log P(\theta) - \log P(D)
$$
这里需要注意 $\theta$ 的初始值为我们最初认定的结果

## Spilt indiviual Traing

这里我们对于每一个任务都进行单独训练一个模型

对于任务1 ，我们有
$$
\log P(\theta^1 |  T_1) =   \log P( T_1| \theta)  + \log P(\theta) - \log P(T_1)
$$
对于任务2，我们同样有
$$
\log P(\theta^2 |  T_2) =   \log P( T_2| \theta)  + \log P(\theta) - \log P(T_2)
$$


Then merge

如果我们假定这里的 Task 是相同类别但是来自不同的来源，那么我们可以将其进行模型融合，即将单独训练后的 $\theta^1 ，\theta^2$ 融合得到 $\theta^{\diamond}$, 假设 $\theta^{\diamond}$ 能够很完美那么我们期待融合后的 $\theta^{\diamond}$ 同样能够在 $T_1,T_2$上起作用,我们同样可以视作这个融合后的 $\theta^{\diamond}$ 也是经过所有数据训练得到的


$$
\log P(\theta^{\diamond} | \{ T_1,T_2\}) = \log P( T_1| \theta)  + \log P( T_2| \theta) +   2  \cdot \log P(\theta) - \log P(T_1)   - \log P(T_2)
$$


如果不进行融合，
$$
f_{\theta}(x) =  
\left\{\begin{array}{lc}

f(\theta^1,x) \quad x\in  T_1\\
f(\theta^2,x)   \quad x\in  T_2\\
\end{array}\right.
$$
我们可以认为有一个选择函数 $I $，当不同输入的时候，可以进行选择对应的模型进行工作



此时对应的优化目标结果就是 $S \cdot \theta$
$$
\log P(I \cdot \theta) =
\left\{\begin{array}{lc}

 I \cdot [\log P( T_1| \theta)  + \log P(\theta) - \log P(T_1)] \quad x\in  T_1\\
I \cdot [ \log P( T_2| \theta)  + \log P(\theta) - \log P(T_2)]   \quad x\in  T_2\\
\end{array}\right.
$$


## Continual Traing

然后我们看一下在持续学习下的学习情况

首先是，对任务1进行训练，此时我们有
$$
\log P(\theta^1 |  T_1) =   \log P( T_1| \theta)  + \log P(\theta) - \log P(T_1)
$$
此时,经过训练后，得到了一个训练完的 $\theta^1$

然后进行任务2的训练
$$
\log P(\theta^2 |  T_1) =   \log P( T_2| \theta^1)  + \log P(\theta^1) - \log P(T_2)
$$
这个时候我们实际上在训练任务2的时候的参数初始化所采用的 $\theta^1$ 就是上式1 的结果，所以可以认为 $\log P(\theta^1)$ 就是  $\log P(\theta^1 |  T_1)$

按着上述的设定，将这个值带入得到
$$
\log P(\theta^2 |  T_1) =   \log P( T_2| \theta^1)  +  \log P( T_1| \theta)  + \log P(\theta) - \log P(T_1) - \log P(T_2)
$$
在 continual learning 中依次训练后的模型就是最终模型，我们可以记作 $\theta^ \circledast$ , 于是有
$$
\log P(\theta^ \circledast |  \{T_1,T_2\}) =     \log P( T_1| \theta)+ \log P( T_2| \theta^1)   + \log P(\theta) - \log P(T_1) - \log P(T_2)
$$



# EWC

根据EWC 的分析，在continual learning的设定和upper bound 也就是多任务学习中，区别如下


![image-20250330122447183](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250330122447347.png)



也就是在持续学习的训练进程中，我们已经是在任务1的训练结果的神经网络$\theta^1$的基础上来训练任务二的，那么可以这个神经网络中的某些权重经过训练会与任务一有关。也可以说神经网络的某些权重是能够反应任务一的数据分布情况。

我们想要是的网络$\theta$在任务一和任务二上都表现地很好，那就需要保护关于任务一的相关的重要参数，然后来继续训练优化在任务二上的损失。

那么我们就需要首先找出来这些对任务一起到重要作用的权重，在训练任务二的时候保护他们，可以是直接提供掩码，冻住这些权重，也可以就是通过损失函数来限制这些权重的更新程度。

好了，不管哪一种方法，首先需要时通过某种度量，找出来这些对任务一重要的权重。

在给定任务 1 的数据 $T_1$ 后，我们可以得到参数 $\theta$ 的**后验概率分布 **$p(\theta^1 |  T_1)$。这个分布携带了我们通过任务 A 所获得的所有信息，尤其是模型对参数的依赖性。这个后验分布 $p(\theta^1 | T_1)$ 在深度神经网络中是一个**高维、非凸、复杂形状**的分布，无法直接求解或表达。

因此我们采用一种常见的数学技巧——**拉普拉斯近似（Laplace Approximation）**，将复杂的分布近似为一个高斯分布（即正态分布）：

拉普拉斯近似下，我们对 $\log p(\theta | D)$ 进行二阶展开，近似为高斯：
$$
p(\theta | D) \approx \mathcal{N}(\theta^*_{\text{MAP}}, \Sigma) \approx \mathcal{N}(\theta^*_A, \text{diag}(F)^{-1})
$$
其中：
$$
\Sigma^{-1} = - \nabla^2 \log p(\theta | D) \Big|_{\theta = \theta^*_{\text{MAP}}}
$$
**在先验为高斯、似然为条件独立情形下，这个 Hessian 可以很好地被 Fisher 信息矩阵近似**，从而使得：
$$
\Sigma^{-1} \approx \mathcal{F}(\theta^*)
$$
这一部分也就是 EWC中提到的 使用Finshe信息矩阵来近似 Hessian矩阵。

- $\theta^*_A$ 是任务 A 学习完成后的最优参数（即后验概率的**最大值点**，也叫 MAP）；
- $\Sigma$ 是该高斯分布的**协方差矩阵**，表示不同参数的不确定性。

这意味着：

- 每个参数 $\theta_i$ 的方差近似为 $\sigma_i^2 = 1 / F_i$；
- 参数 $\theta_i$ 越重要（即 $F_i$ 越大），我们对它的不确定性就越小（方差越小）；
- 后续学习（例如任务 B）应尽可能避免改变这类重要参数。

------

## 在 EWC 中如何使用？

在任务 B 的训练中，损失函数被重新设计为：

$L(\theta) = L_B(\theta) + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta^*_{A,i})^2$

即：

- 第一项是任务 B 的标准损失；
- 第二项是一个来自任务 A 的“记忆保护项”，用于保留旧知识。这里的 $F_i$ 就起到了“记忆守护者”的作用。

EWC用 **Fisher 信息矩阵** 来**近似后验分布的曲率（Hessian）**，从而构建出一个可以“记住”旧任务的重要性结构，并在训练新任务时，以正则化的形式**惩罚那些试图偏离旧任务经验的参数变动**。

其中第二项是：
$$
\sum_i \frac{\lambda}{2} F_i (\theta_i - \theta^*_{A,i})^2
$$
这是一项**加权的平方惩罚项（weighted quadratic penalty）**，本质上是一个 **正则化项**，它的作用是惩罚当前参数 $\theta$ 偏离任务 A 学到的最优参数 $\theta^*_A$。

但重点在于：**每个参数 $\theta_i$ 的惩罚程度由 Fisher 信息 $F_i$ 决定**。

- 如果某个参数在任务 A 中非常关键（$F_i$ 很大）：则即使 $\theta_i$ 发生很小的变化，也会使惩罚项显著增加，优化器会自动 **抑制该参数的更新**，从而保持任务 A 的性能。
- 如果某个参数在任务 A 中不太重要（$F_i$ 很小）：则该参数可以在任务 B 中自由变化，以适应新任务，而不会对任务 A 的表现造成明显影响。

如果你从优化的角度来看，这个正则项会**改变任务 B 的损失函数的几何形状**，使得：

- 在靠近 $\theta^*_A$ 的区域形成一个陡峭的“谷底”；
- 该区域就是任务 A 的参数高性能区域；
- 优化器即使尝试优化任务 B，也很难将参数推离该区域太远。

这就确保了模型即使适应了任务 B，也不会遗忘任务 A 的表现。

![image-20250330192028922](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250330192029090.png)





#  LoRA


![image-20250405191847815](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250405191848102.png)





根据 LoRA的原始论文 
$$
h = W_0x + ∆W x = W_0x + BAx
$$
其中 在初始化的时候 $B=0$ ,$A$由正态分布产生， 其中 $B$ 的维度为$d \times r$ , $A$的维度为 $r \times d$



## 多个任务

当我们考虑到多个任务的学习时候，就是每一个任务 $T$ 都会训练得到一个对应的 $\Delta W_t = B_tA_t$ ,其中$A_t$是根据正态分布生成，对应的 $B_t$初始化为0， 每次训练后会更新这两部分的参数，于是在训练完$T$个任务后，就有
$$
\{(A_1B_1),(A_2B_2),(A_3B_3),\dots,(A_TB_T)\}
$$
然后我们在训练完$T$个任务之后，在推断的时候，怎么来使用或者选择这 $T$个专家呢？ 我们就需要一个 Router 或者 Select 来从 $T$中进行选择某一个，某几个，或者 使用它们的全部但是每一个分配不同的权重
$$
k^1,k^2,k^3,\cdots,k^T
$$


那么这个时候代码的最终结果是什么呢？
$$
\begin{aligned}
h &=  W_0 x + k^1 \Delta W_1 x + k^2\Delta W_2 x +\cdots + k^T\Delta W_T x\\
 & = W_0 x + k^1 (B_1A_1) x + k^2 (B_2A_2) x +\cdots + k^T (B_TA_T) x \\
 & = W_0 x + \sum_{i=1}^T k^i(B_iA_i)
\end{aligned}
$$
 我们来关注后面的这一项 ，我们可以将其写成矩阵形式，
$$
\sum_{t=1}^T k^t B_t A_t = \underbrace{\left[ B_1 \dots B_T \right]}_{d \times rT} \cdot \underbrace{\text{diag}(k^1 I_r, \dots, k^T I_r)}_{rT \times rT} \cdot \underbrace{\begin{bmatrix} A_1 \\ \vdots \\ A_T \end{bmatrix}}_{rT \times d}
$$
其中：

- $A = \text{block-diag}(A_1, \dots, A_T) \in \mathbb{R}^{T \cdot r \times  d}$
- $B = \text{block-diag}(B_1, \dots, B_T) \in \mathbb{R}^{d \times T \cdot r}$
- $\tilde{R} = \text{diag}(k^1 I_d, \dots, k^T I_d) \in \mathbb{R}^{T \cdot r \times T \cdot r}$

从而整个表达式变为：
$$
\begin{aligned}

h &= W_0 x + \left( B \tilde{R} A \right) x \\
&= \left(W_0 + B \tilde{R} A \right) x \\
 &= \left( W_0 + \left[ B_1 \dots B_T \right] \cdot \text{diag}(k^1 I_r, \dots, k^T I_r) \cdot \begin{bmatrix} A_1 \\ \vdots \\ A_T \end{bmatrix} \right) x
 
 \end{aligned}
$$


同样我们注意到其实每一个任务对应的LoRA部分也可以进行类似上面的拆解

## 单个任务的拆解

们可以将每个任务的 $A_t$ 和 $B_t$ 分解为 **多个专家子空间** 的组合，并通过路由矩阵动态加权。以下是分步拆解：

---

### **步骤 1：分解单任务的 $A_t B_t$**

$$
\begin{aligned}
\Delta W x_t \\
 = B^t_{(d \times r)}A^t_{(r \times d)} x

\end{aligned}
$$



将 $B_t$ 和 $A_t$ 分解为 **列向量** 和 **行向量**：
$$
B^t = \begin{bmatrix} b^t_{1} & \dots & b^t_{r} \end{bmatrix} ,\quad b^t_{j} \in \mathbb{R}^{d \times 1},0<j\leq r,B^t\in \mathbb{R}^{d \times r} \\


\quad A^t = \begin{bmatrix} a_{1}^{t\top} \\ \vdots \\ a_{r}^{t\top} \end{bmatrix} ,\quad a^t_{j} \in \mathbb{R}^{1 \times d},0<j\leq r,A^t\in \mathbb{R}^{d \times r}
$$
其中：

- $b^t_{ti} \in \mathbb{R}^{d \times 1}$（$B_t$ 的列），
- $a_{ti}^{t\top} \in \mathbb{R}^{1 \times d}$（$A_t$ 的行）。

$$
k^t=\text{diag}\Big( \underbrace{k^{t}_{1}I_1, \dots, k^{t}_{r}I_1}_{\text{task t}}\Big)
$$

则：
$$
\Delta W_t = k^tB^t A^t = \sum_{i=1}^r k^t_ib_{i}^t a_{i}^{t\top}
$$

- **物理意义**：每个 $b_{ti} a_{ti}^\top$ 对应一个 **低秩子空间**（秩为1），共 $r$ 个子空间





### **2. 多任务的全局矩阵构建**

假设有 \( T \) 个任务，每个任务贡献 \( r \) 个秩-1子空间（即专家），则：

#### **(1) 全局矩阵 \( B \) 的展开**

横向拼接所有任务的 \( B_t \)：
$$
B = \begin{bmatrix} B^1 & B^2 & \dots & B^T \end{bmatrix} = \begin{bmatrix} 
 \underbrace{b_{1}^1  \dots  b_{r}^1}_{\text{task 1}},
 
 & \dots ,
 & \underbrace{ b_{1}^t  \dots  b_{r}^t }_{\text{task t}},
\end{bmatrix} \in \mathbb{R}^{d \times Tr}
$$

#### **(2) 全局矩阵 \( A \) 的展开**

纵向堆叠所有任务的 \( A_t \)：
$$
A = \begin{bmatrix} A^1 \\ A^2 \\ \vdots \\ A^T \end{bmatrix} = \begin{bmatrix} 
\underbrace{ a_{1}^{1 \top}  \dots  a_{r}^{1\top}}_{task 1} \\ 
\underbrace{ a_{1}^{2 \top}  \dots  a_{r}^{2\top}}_{task 2} \\
\vdots  \\
\underbrace{a_{1}^{t\top}  \dots  a_{r}^{t\top}  }_{task t}
\end{bmatrix} \in \mathbb{R}^{Tr \times d}
$$

#### **(3) 路由矩阵 \( R \) 的展开**

定义块对角路由矩阵 \( R \in \mathbb{R}^{Tr \times Tr} \)，每个块为标量权重 $ k_{ti} $扩展的单位矩阵：
$$
R = \text{diag}\Big( \underbrace{k^{1}_{1}I_1, \dots, k^{1}_{r}I_1}_{\text{task 1}}, \dots, \underbrace{k_{1}^{t}I_1, \dots, k_{r}^tI_1}_{\text{task T}} \Big)  \\
= \text{diag}\Big( \underbrace{k^{1}_{1}, \dots, k^{1}_{r}}_{\text{task 1}}, \dots, \underbrace{k_{1}^{t}, \dots, k_{r}^t}_{\text{task T}} \Big)
$$
其中：

- $ k^t \in \mathbb{R} $ 是任务 $ t $ 中专家的权重，
- $ I_1$ 是 $ 1 \times 1 $ 单位矩阵（标量扩展为对角块）。
- $k_{ti} \in \mathbb{R}$ is the weight of the $i$-th expert in task $t$,
- $I_1$ is the $1 \times 1$ identity matrix (scalar as diagonal block).

---

### **3. 整体增量权重的矩阵形式**

所有任务的加权增量权重为：
$$
\begin{aligned}
\Delta W 
&=  B \cdot R \cdot A \quad \in \mathbb{R}^{d \times d} \\
&= \sum_{t=1}^T \sum_{i=1}^r \left( \underbrace{k_i^t}_{weight}  \cdot\underbrace{b_i^t  }_{\text{ column vector}} \cdot \underbrace{a_i^{t\top}}_{\text{row vector}} \right) \\
& 
= \sum_{t=1}^T \sum_{i=1}^r k_i^t \begin{bmatrix}
b_i^t(1) a_i^{t}(1) & \dots & b_i^t(1) a_i^{t}(d) \\
\vdots & \ddots & \vdots \\
b_i^t(d) a_i^{t}(1) & \dots & b_i^t(d) a_i^{t}(d)
\end{bmatrix}
 
 \end{aligned}
$$

$$
\Delta W(m,n)  &= \sum_{t=1}^T \sum_{i=1}^r k_i^t \cdot b_i^t(m) a_i^t(n) \quad (1 \leq m,n \leq d) \\
$$




## 讨论不同的设置

###  任务 \( t \) 内所有子空间共享权重  $ k^{t}_{1} = k^{t}_{2} = \dots = k^{t}_{r} = k^t $

在应用LorA作为 不同的 Expert 的时候，其实我们应该注意到，每一个任务级的权重是共享的，也就是对于任务 $t$内的 $r$个专家，其权重就是这个任务的权重

#### **(1) 任务级权重表示**

退化后的输出为：
$$
h = W_0 x +  \Delta W x_t\\

= W_0 x + \sum_{t=1}^T k^t B^t A^t x
$$

#### **(2) 专家子空间级权重表示**

路由矩阵 \( R \) 中任务 \( t \) 的块为  $ k_t I_r $（而非 $ k_{ti} I_1 $）：
$$
R = \text{diag}(k^1 I_r, \dots, k^T I_r) \\
 = \text{diag}(\underbrace{k^{1}I_1, \dots, k^{1}I_1}_{\text{task } 1}, \dots, \underbrace{k^{t}I_1, \dots, k^{t}I_1}_{\text{task } t}, \dots, )
$$
则：
$$
\Delta W = B R A = \sum_{t=1}^T k^t B^t A^t = \sum_{t=1}^T k^t \sum_{i=1}^r  b^t_{i} a_{i}^{t\top}{}
$$
**结论**：两种表示等价，退化为 **任务级加权求和**，即每个任务的子空间共享相同权重。

#### 总结

**任务内共享权重**  

- 专家子空间级权重表示可严格退化为任务级权重表示，验证了路由矩阵设计的灵活性。

### 仅任务 $ t $ 的权重 $ k^t = 1$， $k^{-t} = 0$
#### **(1) 任务级权重表示**
退化后的输出为：
$$
h = W_0 x + k^t B^t A^t x = W_0 x + B^t A^t x
$$

#### **(2) 专家子空间级权重表示**
路由矩阵 $ R $ 中只有任务 $ t $ 的块非零：
$$
R = \text{diag}(0I_1, \dots, I_t, \dots, 0I_T) \\
= \text{diag}(\underbrace{0I_1, \dots, 0I_1}_{\text{task } 1}, \dots, \underbrace{k^{t}_{1}I_1, \dots, k^t_{r}I_1}_{\text{task } t}, \dots, \underbrace{0I_1, \dots, 0I_1}_{\text{task } T}) \\
= \text{diag}(0, \dots, \underbrace{1, \dots, 1}_{\text{task } t}, \dots, 0)
$$
若 $ k_{ti} = 1 \, (\forall i) $，则：
$$
\Delta W = B R A = B^t A^t =  \sum_{i=1}^r  b^t_{i} a_{i}^{t\top}
$$
对应的路由矩阵为
$$
R = \text{diag}(0, \dots, \underbrace{1, \dots, 1}_{\text{task } t}, \dots, 0)
$$

**结论**：两种表示均退化为 **单任务 LoRA**，即仅激活任务 \( t \) 的增量权重。



**单任务激活**  

- 两种表示均退化为单任务 LoRA，无参数混合。

  

---

### **2. 所有任务权重  $ k_t = 1 $**
#### **(1) 任务级权重表示**
退化后的输出为：
$$
h = W_0 x + \sum_{t=1}^T B^t A^t x
$$

#### **(2) 专家子空间级权重表示**
路由矩阵 \( R \) 中所有块的权重  $ k_{ti} = 1 $：
$$
R = \text{diag}(I_1, \dots, I_1) = I_{Tr}
$$
则：
$$
\Delta W = B R A = B A = \sum_{t=1}^T B^t A^t =\sum_{t=1}^T  1 \sum_{i=1}^r  b^t_{i} a_{i}^{t\top}
$$
**结论**：两种表示均退化为 **所有任务增量权重的直接相加**，即并行使用所有 LoRA 适配器。



全任务激活**  
- 两种表示等价于所有 LoRA 适配器的直接相加，可能引发维度爆炸或干扰，需谨慎使用。



### **共享与任务特定的混合参数表示**
当 LoRA 的矩阵 $ A $ 和 $B $部分共享、部分任务特有时，可以通过以下方式表示：

#### **(1) 参数分解**
- **共享参数**：定义全局共享的低秩矩阵
  $$
    A_{\text{shared}} \in \mathbb{R}^{r_{\text{shared}} \times d} , B_{\text{shared}} \in \mathbb{R}^{d \times r_{\text{shared}}} 
  $$
  
- **任务特定参数**：每个任务 $ t $定义额外的低秩矩阵 
  $$
  A^t \in \mathbb{R}^{r_{\text{task}} \times d} , B^t \in \mathbb{R}^{d \times r_{\text{task}}}
  $$
  

#### **(2) 联合矩阵构建**
全局参数矩阵横向/纵向拼接共享和任务特定部分：
$$
B = \begin{bmatrix} B_{\text{shared}} & B^1 & \dots & B^T \end{bmatrix} \in \mathbb{R}^{d \times (r_{\text{shared}} + T r_{\text{task}})}
$$
$$
A = \begin{bmatrix} A_{\text{shared}} \\ A^1 \\ \vdots \\ A^T \end{bmatrix} \in \mathbb{R}^{(r_{\text{shared}} + T r_{\text{task}}) \times d}
$$

#### **(3) 路由矩阵设计**
路由矩阵 \( R \) 控制共享参数和任务特定参数的激活权重：
- **共享部分权重**：\( k_{\text{shared}} \in \mathbb{R} \)，
- **任务特定权重**：\( k_t \in \mathbb{R} \)（每个任务 \( t \) 独立权重）。

路由矩阵为块对角形式：
$$
R = \text{diag}\Big( \underbrace{k_{\text{shared}} I_{r_{\text{shared}}} }_{\text{share}}, \underbrace{k_1 I_{r_{\text{task}}}, \dots, k_T I_{r_{\text{task}}} }_{\text{task-specific}} \Big)
$$

#### **(4) 增量权重表达式**
最终增量权重为共享和任务特定部分的加权组合：
$$
\Delta W = B R A = \underbrace{k_{\text{shared}} B_{\text{shared}} A_{\text{shared}}}_{\text{shared}} + \sum_{t=1}^T \underbrace{k^t B^t A^t}_{\text{task-specific}}
$$

---

### **分层的任务特定 LoRA 适配**
如果任务 $ t $仅在神经网络的某些层添加 LoRA 适配器，可以通过分层路由机制实现：

#### **(1) 分层参数定义**
假设网络有 $ L $层，每层 $ l $ 的权重为 $ W_0^{(l)} $，任务 $ t $ 在层 $ l $ 的 LoRA 适配器为 $ B_t^{(l)} A_t^{(l)} $。

 The LoRA adapter at layer $ l $ is $ B_t^{(l)} A_t^{(l)} $  in task $ t $  。

#### **(2) 分层路由矩阵**
定义分层路由矩阵 $ R^{(l)} $，控制任务$ t $ 在层 $ l $ 的激活权重：
$$
R^{(l)} = \text{diag}\Big( k_{1}^{(l)} I_{r}, \dots, k_{T}^{(l)} I_{r} \Big)
$$
其中：
- $ k_{t}^{(l)} \in \mathbb{R} $：任务 $t $ 在层  $l $ 的权重，
- $ k_{t}^{(l)} = 0 $ 表示任务 $ t $ 不在层 $ l$ 添加适配器。

#### **(3) 分层增量权重**
层 \( l \) 的增量权重为：
$$
\Delta W^{(l)} = \sum_{t=1}^T k_{t}^{(l)} B_t^{(l)} A_t^{(l)} = B^{(l)} R^{(l)} A^{(l)}
$$
其中：
- $ B^{(l)} = \begin{bmatrix} B_1^{(l)} & \dots & B_T^{(l)} \end{bmatrix} \in \mathbb{R}^{d \times Tr} $,
- $ A^{(l)} = \begin{bmatrix} A_1^{(l)} \\ \vdots \\ A_T^{(l)} \end{bmatrix} \in \mathbb{R}^{Tr \times d} $.

#### **(4) 整体输出表达式**
网络第 \( l \) 层的输出为：
$$
h^{(l)} = \left( W_0^{(l)} + B^{(l)} R^{(l)} A^{(l)} \right) x^{(l)}
$$

---

**关键结论**

1. **混合共享与任务特定参数**  
   - 共享部分通过全局权重 \( k_{\text{shared}} \) 控制，任务特定部分通过独立权重 \( k_t \) 激活。
   - 物理意义：共享子空间捕捉跨任务的通用特征，任务特定子空间适配个性化需求。

2. **分层任务适配**  
   - 不同任务在不同层激活适配器，通过分层路由矩阵 \( R^{(l)} \) 实现稀疏控制。
   - 物理意义：任务可能依赖不同层级的特征（浅层提取基础特征，深层处理高级语义）。

3. **统一框架的灵活性**  
   - 两种场景均可通过路由矩阵的块对角设计和参数拼接统一表示。
   - 代码实现中可通过掩码（Mask）或条件判断动态选择激活的块。

###   在添加新的 LoRA 的时候添加限制



#### O-LoRA

原来的表述
$$
\begin{aligned}
\Delta W 
&=  B \cdot R \cdot A \quad \in \mathbb{R}^{d \times d} \\
&= \sum_{t=1}^T \sum_{i=1}^r \left( \underbrace{k_i^t}_{weight}  \cdot\underbrace{b_i^t  }_{\text{ column vector}} \cdot \underbrace{a_i^{t\top}}_{\text{row vector}} \right) \\
& 
= \sum_{t=1}^T \sum_{i=1}^r k_i^t \begin{bmatrix}
b_i^t(1) a_i^{t}(1) & \dots & b_i^t(1) a_i^{t}(d) \\
\vdots & \ddots & \vdots \\
b_i^t(d) a_i^{t}(1) & \dots & b_i^t(d) a_i^{t}(d)
\end{bmatrix}
 
 \end{aligned}
$$
论文中提到 将 $B^t$视作 基向量与空间，而将$A^t$视作 系数

子空间：
$$
B^t=[b^t_{1},b^t_{2},\cdots,b^t_{r}], b^t_{j} \in \mathbb{R}^{d \times 1},0<j\leq r,B^t\in \mathbb{R}^{d \times r} \\
$$
特征系数
$$
\quad A_t = \begin{bmatrix} a_{1}^{t\top} \\ \vdots \\ a_{r}^{t\top} \end{bmatrix},  a^t_{j} \in \mathbb{R}^{1 \times d},0<j\leq r,A^t\in \mathbb{R}^{r \times d}
$$


正交性约束的分析
$$
B^i=[b^i_{1},b^i_{2},\cdots,b^i_{r}], b^i_{j} \in \mathbb{R}^{d \times 1},0<j\leq r,B^i\in \mathbb{R}^{d \times r} \\

B^t=[b^t_{1},b^t_{2},\cdots,b^t_{r}], b^t_{j} \in \mathbb{R}^{d \times 1},0<j\leq r,B^t\in \mathbb{R}^{d \times r} \\
$$

$$
\begin{aligned}
O_{i.t} &= B^{iT}B^t \\


\end{aligned}
$$

$$
\begin{aligned}
 L_{orth}(B_i,B_t) &=  \sum_{j,k} || O_{i,t}[j,k] ||^2 =  \sum_{j,k} || B^T_iB_t[j,k] ||^2   \\
 &= \sum_{j,k} (b_{ij}^Tb_{tk})^2

\end{aligned}
$$
where $O_{i,t}[j, k]$ denotes the element at the $j-th$ row and $k-th$ column of $O_{i,t}$


$$
\lambda_1 \sum_{i=1}^t L_{\text{orth}}(B_i, B_t)
$$


进行扩展推导理解

两个矩阵乘积的理解
$$
\begin{aligned}
O_{i.t} &= B^T_iB_t  \in \mathbb{R}^{r \times r} \\
&= [b^i_{1},b^i_{2},\cdots,b^i_{r}]^T[b^t_{1},b^t_{2},\cdots,b^t_{r}] \\
&= \begin{bmatrix}
b^i_{1T} \\
b^i_{2T} \\
\vdots \\
b^i_{rT}
\end{bmatrix}
\begin{bmatrix}
b^t_{1},b^t_{2},\cdots,b^t_{r}
\end{bmatrix} \\
&= 
\begin{bmatrix}
b^{iT}_{1}b^t_{1} & b^{iT}_{1}b^t_{2} & \cdots & b^{iT}_{1}b^t_{r} \\
b^{iT}_{2}b^t_{1} & b^{iT}_{2}b^t_{2} & \cdots & b^{iT}_{2}b^t_{r} \\
\vdots & \vdots & \ddots & \vdots \\
b^{iT}_{r}b^t_{1} & b^{iT}_{r}b^t_{2} & \cdots & b^{iT}_{r}b^t_{r}
\end{bmatrix}
\end{aligned}
$$


正交化约束理解
$$
\begin{aligned}
 L_{orth}(B_i,B_t) &=  \sum_{j,k} || O_{i,t}[j,k] ||^2 =  \sum_{j,k} || B^T_iB_t[j,k] ||^2   \\
 
&=  \sum_{j,k} ||b^i_jb^t_k||^2
= \sum_{j,k} (b^i_jb^t_k)^2

\\
&= \| B^T_i B_t \|_F^2 
= 
\begin{bmatrix}
b^{iT}_{1}b^t_{1} & b^{iT}_{1}b^t_{2} & \cdots & b^{iT}_{1}b^t_{r} \\
b^{iT}_{2}b^t_{1} & b^{iT}_{2}b^t_{2} & \cdots & b^{iT}_{2}b^t_{r} \\
\vdots & \vdots & \ddots & \vdots \\
b^{iT}_{r}b^t_{1} & b^{iT}_{r}b^t_{2} & \cdots & b^{iT}_{r}b^t_{r}
\end{bmatrix} \\



\end{aligned}
$$



$$
t \neq i, 
\begin{aligned}
\quad L_{\text{orth}}(B^i, B^t) = O \ 
& \Longleftrightarrow \ B^i \bot B^t \  \\
 \sum_{j,k} ||b^{i}_{j}b^{t}_{k}||^2=0 & \Longleftrightarrow \sum_{j,k} b^{i}_{j}b^{t}_{k} =0  \Longleftrightarrow
\{b^{i}_{j}\} \bot \{b^{t}_{k}\} \quad (j,k=1,\dots,r) \\
where \quad
 b^{i}_{j} b^{t}_{k} =0 &\Longleftrightarrow \sum_{m=1}^d b^{i}_{j}(m)b^{t}_{k}(m) =0

\end{aligned}
$$
我们发现，当任务过多的时候，上述正交性不再满足
$$
t > t_{\text{max}} = \left\lfloor \frac{d}{r} \right\rfloor \quad \Rightarrow \quad \lambda_1 \sum_{i=1}^t L_{\text{orth}}(B^i, B^t) \neq 0
$$


最终结论：所以加入L2限制限制后，对应的约束条件 为
$$
\sum_{i=1}^TL(B^i,B^t)=\sum_{i=1}^T \sum_{j,k}^r b^i_{j}b^t_{k} 
= \sum_{i=1}^T \sum_{j,k}^r \sum_{m=1}^db^i_{j}(m)b^t_{k}(m)

= O
$$




### N -LoRA



举例说明
$$
\Delta W^1 = B^1 = \begin{bmatrix}1 & 1 \\ -1 & -1 \end{bmatrix}, \quad \Delta W^2 = B^2 = \begin{bmatrix}1 & -1 \\ 1 & -1 \end{bmatrix}  \\

 \Delta W^{1\top} \Delta W^2 = B^1B^2 = \begin{bmatrix}0 & 0 \\ 0 & 0 \end{bmatrix}=O \quad  \\ 

\Delta W^1 \odot \Delta W^2 = \begin{bmatrix}
1 \cdot 1 & 1 \cdot (-1) \\
(-1) \cdot 1 & (-1) \cdot (-1)
\end{bmatrix} = \begin{bmatrix}
1 & -1 \\
-1 & 1
\end{bmatrix} \neq 0
$$

$$ { }
\Delta W^1 = B^1= \begin{bmatrix}
a & 0 \\
0 & 0
\end{bmatrix}, \quad
\Delta W^2 = B^2=\begin{bmatrix}
0 & 0 \\
0 & b
\end{bmatrix}, \quad a, b \neq 0  \\

\Delta W^1 \odot \Delta W^2 = \begin{bmatrix}
a \cdot 0 & 0 \cdot 0 \\
0 \cdot 0 & 0 \cdot b
\end{bmatrix} = \begin{bmatrix}
0 & 0 \\
0 & 0
\end{bmatrix} \\

\Delta W^{1\top} \Delta W^2  = \begin{bmatrix}
0 & 0 \\
0 & 0
\end{bmatrix}
$$


分析
$$
B^i=[b^i_{1},b^i_{2},\cdots,b^i_{r}], b^i_{j} \in \mathbb{R}^{d \times 1},0<j\leq r,B^i\in \mathbb{R}^{d \times r} \\

B^t=[b^t_{1},b^t_{2},\cdots,b^t_{r}], b^t_{j} \in \mathbb{R}^{d \times 1},0<j\leq r,B^t\in \mathbb{R}^{d \times r} \\
$$




约束条件
$$
\begin{aligned}
\Delta W^i \odot \Delta W^t &= O \\
B^i \odot B^t &=  O \\
b_{j}^{i}(m) = 0  \or  b_{j}^{t}(m)   &= 0
\end{aligned}
$$

$$
\mathcal{L}_{\text{collision}} = \sum_{i \neq t} \sum_{a,b} \mathbf{1}\left\{ \Delta W^1[a,b] \neq 0 \ \land \cdots \land\ \Delta W^T[a,b] \neq 0 \right\} \\
= \sum_{i \neq t} \sum_{j} \mathbf{1}\left\{ B^1\neq 0 \ \land \cdots \land\ B^T \neq 0 \right\} \\
= \sum_{i \neq t} \sum_{j} \mathbf{1}\left\{ b^1_{j}(m)\neq 0 \ \land \cdots \land\ b^T_{j}(m) \neq 0 \right\}
$$

$$
\mathcal{L}_{\text{sparse}} = \lambda \sum_{i=1}^n \|\Delta W_i\|_1 = \lambda \sum_{i=1}^n \sum_{a,b} |\Delta W_i[a,b]| \\
=  \lambda \sum_{i=1}^T \sum_j |B^i| =  \lambda \sum_{i=1}^T   \sum_{j=1}^r \sum_{m=1}^d|b^i_j(m)|
$$


当什么时候失效呢？
$$
t > t_{\text{max}} = \left\lfloor \frac{d^2}{r} \right\rfloor \quad \Rightarrow \quad  \exists i \neq j, \quad \text{Supp}(\Delta W_i) \cap \text{Supp}(\Delta W_j) \neq \emptyset
$$





# SAM 



SAM~\citep{foretSharpnessAwareMinimizationEfficiently2021} seeks to find parameters that lie in flat regions of the loss landscape by solving:

$$
\begin{equation}
\min_\theta \max_{\|W\|_2 \leq \rho} L(W + \epsilon),
\end{equation}
$$
where $\rho$ controls the perturbation radius.

SAM can be summarized as solving the following bilevel optimization:
$$
\begin{aligned}
\textbf{Outer minimization:} \quad & \min_{W} \mathcal{L}(W + \epsilon^*(W)), \\
\textbf{Inner maximization:} \quad & \epsilon^*(W) = \arg\max_{\|\epsilon\|_2 \leq \rho} \mathcal{L}(W + \epsilon).
\end{aligned}
$$
This approach ensures that the update direction accounts for local sharpness, promoting parameter solutions that generalize better.



## Calcualte Step 

SAM proceeds by two steps per update:

### **Step 1 Compute adversarial perturbation direction**

calculate the  most sharp point $w + \epsilon$ in the perturbation radius  $\rho$  with the input $\mathcal{B}$ ,where $\mathcal{B}$ denotes a batch of training data

**1.1** calculater the  origin model's loss and gradient 
$$
\mathcal{L}(W; \mathcal{B}) \\
g=\nabla_{W} \mathcal{L}(W; \mathcal{B})
$$
**1.2** Then, it calculates the normalized perturbation $\epsilon$ in the direction of the gradient:
$$
\begin{equation}
    \epsilon^* = 
    \rho \frac{g}{\|g\|_2}
    
    =\rho \frac{\nabla_{W} \mathcal{L}(W;\mathcal{B})}{\|\nabla_{W} \mathcal{L}(W;\mathcal{B})\|_2}.
    \end{equation}
$$
If adaptive SAM is used, the perturbation is scaled element-wise based on parameter magnitude:
$$
\begin{equation}
\epsilon_i = \rho \cdot \frac{ \| W_i \| \cdot g_i }{ \| \|W \| \cdot g \|_2 },
\end{equation}
$$
where $|\cdot|$ denotes the element-wise norm and scaling.

**1.3** Add noise $\epsilon $ on the origin model 
$$
W_{adv} =  W + \epsilon^*
$$

### **Step 2  Gradient descent at the adversarial point**

**2.1** Keeping $W_{\text{adv}}$ fixed, SAM evaluates the loss again
$$
\mathcal{L}(W+\epsilon^*; \mathcal{B})
$$
**2.2** Computes the gradient at the perturbed point
$$
\begin{equation}
   g_{\text{adv}}= \nabla_{W} \mathcal{L}(W + \epsilon^*; \mathcal{B}).
\end{equation}
$$
**2.3** Finally, SAM restores the original parameters $W$, and applies a standard optimizer step using $g_{\text{adv}}$:
$$
\begin{equation}
W \leftarrow W - \eta \cdot g_{\text{adv}},
\end{equation}
$$
where $\eta$ is the learning rate.



## Varient when apply in DDP 









## SAM -LoRA

LoRA Mode
$$
W=W_0+\Delta_{\text{LoRA}}W=W_0 + BA
$$
where $W_0 \in \mathbb{R}^{d \times k}$ is the pre-trained and frozen weight, while $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ are trainable low-rank matrices with rank $r \ll \min(d, k)$. This reduces trainable parameters and enables efficient fine-tuning.

As a result, the loss function $\mathcal{L}(W; \mathcal{B})$, where $\mathcal{B}$ denotes a batch of training data, is optimized only with respect to the LoRA parameters:
$$
g = \nabla_{BA} \mathcal{L}(W; \mathcal{B}).
$$


### **Step 1: Compute adversarial perturbation direction**

**1.1** At the current parameters $W = W_0 + \Delta{W}$, we compute the gradient:
$$
g = \nabla_{BA} \mathcal{L}(W; \mathcal{B}) = (\nabla_{A} \mathcal{L}(W; \mathcal{B}),\nabla_{B} \mathcal{L}(W; \mathcal{B}))
$$
We then construct a pseudo-gradient in the full parameter space, where frozen parameters have zero gradient, to compute the SAM perturbation:
$$
g = 
\begin{cases}
g_i, & \text{if } \theta_i \text{ is trainable (LoRA)} \\
0, & \text{otherwise}
\end{cases}
$$

$$
g_{B} = \nabla_{B} \mathcal{L}(W; \mathcal{B}),  \\
g_{A} =\nabla_{A} \mathcal{L}(W; \mathcal{B})
$$



**1.2** The perturbation is defined as:
$$
\epsilon = \rho \cdot \frac{\tilde{g}}{\|\tilde{g}\|_2},
$$
the result is 
$$
\epsilon_B =  \rho \cdot \frac{\nabla_{B} \mathcal{L}(W; \mathcal{B})}{\|\nabla_{B} \mathcal{L}(W; \mathcal{B})\|_2}
$$

$$
\epsilon_A =  \rho \cdot \frac{\nabla_{A} \mathcal{L}(W; \mathcal{B})}{\|\nabla_{A} \mathcal{L}(W; \mathcal{B})\|_2}
$$

**1.3** then add noise $\epsilon $ on the origin model 

the result is 
$$
B_{\text{adv}} = (B + \epsilon_{B}) , A_{\text{adv}} = (A + \epsilon_{A})
$$

$$
W_{adv} = W_0+ B_{\text{adv}}A_{\text{adv}} =W_0 +  (B + \epsilon_{B})(A +  \epsilon_{B})
$$


### **Step 2: Evaluate and update**

2.1  Keeping $W_{\text{adv}}$ fixed, SAM evaluates the loss again

We evaluate the loss at the perturbed point:
$$
\mathcal{L}_{\text{adv}} = \mathcal{L}(W_{\text{adv}}; \mathcal{B}) 
= \mathcal{L} ( W +   (B + \epsilon_{B})(A +  \epsilon_{B});\mathcal{B})
$$
**2.2** computes the gradient at the perturbed point

Then, compute the gradient only with respect to $\Delta_{\text{LoRA}}$:
$$
g_{\text{adv}}  
= \nabla_{BA} \mathcal{L}(W_{\text{adv}}; \mathcal{B}) 
= \nabla_{BA} \mathcal{L}( W +   (B + \epsilon_{B})(A +  \epsilon_{B}); \mathcal{B})
$$
so 
$$
g_{(B,\ \text{adv})} = \nabla_{B} \mathcal{L}(W +   (B + \epsilon_{B})(A +  \epsilon_{B}); \mathcal{B}) \\

g_{(A,\ \text{adv})} = \nabla_{A} \mathcal{L}(W +   (B + \epsilon_{B})(A +  \epsilon_{B}); \mathcal{B})
$$


2.3 Finally, we perform a parameter update:
$$
\Delta_{\text{LoRA}} \leftarrow \Delta_{\text{LoRA}} - \eta \cdot g_{\text{adv}},
$$
where $\eta$ is the learning rate.
$$
B \leftarrow  B - \eta \cdot  \nabla_{B} \mathcal{L}(W +   (B + \epsilon_{B})(A +  \epsilon_{B}); \mathcal{B}), \\
A \leftarrow  A - \eta \cdot  \nabla_{A} \mathcal{L}(W +   (B + \epsilon_{B})(A +  \epsilon_{B}); \mathcal{B}),
$$
 here , the  noise are only affect the local LoRA paramter rather than the whole model









# RWP 

## Intro 

RWP~\citep{du2023efficient} introduces an expectation-based smoothing loss by perturbing model weights with Gaussian noise.



The smoothed (Bayesian) loss is defined as:
$$
\mathcal{L}_{\text{Bayes}}(\theta) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, \sigma^2)} [\mathcal{L}(\theta + \epsilon)]
$$
A mixed loss is then proposed:
$$
\mathcal{L}_m(\theta) = \lambda \cdot \mathcal{L}_{\text{Bayes}}(\theta) + (1 - \lambda) \cdot \mathcal{L}(\theta)
$$
where $\lambda \in [0,1]$ controls the trade-off between robustness and task-specific learning.

the first LBayes(w) provides a smoothed landscape that biases the network towards flat region, while the second L(w) helps recover the necessary local information and better locates the minima that contributes to high performance.



## Calculate Step 

### Step 1: Compute clean gradient

**1.1** Forward and backward on clean weights:
$$
g_0 = \nabla_\theta \mathcal{L}(\theta; \mathcal{B})
$$

------

### Step 2: Compute perturbed gradient

**2.1** Sample noise:
$$
\epsilon \sim \mathcal{N}(0, \sigma^2 \cdot \|\theta\|)
$$
**2.2** Evaluate perturbed loss and gradient:
$$
g_1 = \nabla_\theta \mathcal{L}(\theta + \epsilon;\mathcal{B})
$$

------

### Step 3: Mixed gradient update

**3.1** Combine gradients:
$$
g = \lambda g_1 + (1 - \lambda) g_0 
 = \lambda \nabla_\theta \mathcal{L}(\theta + \epsilon;\mathcal{B}) + (1 - \lambda) \nabla_\theta \mathcal{L}(\theta; \mathcal{B})
$$
**3.2** Update parameters:
$$
\theta \leftarrow \theta - \eta \cdot g
$$

------

## Varient from paper - 

To enhance the optimization process, we utilize two distinct batches of data, namely B1 and B2, for the two gradient steps involved.
$$
g = \lambda g_1 + (1 - \lambda) g_0 
 = \lambda \nabla_\theta \mathcal{L}(\theta + \epsilon;\mathcal{B_1}) + (1 - \lambda) \nabla_\theta \mathcal{L}(\theta; \mathcal{B_2})
$$


## Varient when apply in DDP 

calcualate the N step 's gradient, then do one time gradient update 
$$
g = \lambda g_N + (1 - \lambda) \sum_{i=1}^{N-1} g_0 
 = \lambda \nabla_\theta \mathcal{L}(\theta + \epsilon;\mathcal{B_N}) +  \sum_{i=1}^{N-1}(1 - \lambda) \nabla_\theta \mathcal{L}(\theta; \mathcal{B_i})
$$






## RWP in LoRA Setting



LoRA Mode
$$
W=W_0+\Delta_{\text{LoRA}}W=W_0 + BA
$$
where $W_0 \in \mathbb{R}^{d \times k}$ is the pre-trained and frozen weight, while $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ are trainable low-rank matrices with rank $r \ll \min(d, k)$. This reduces trainable parameters and enables efficient fine-tuning.

As a result, the loss function $\mathcal{L}(W; \mathcal{B})$, where $\mathcal{B}$ denotes a batch of training data, is optimized only with respect to the LoRA parameters:
$$
g = \nabla_{BA} \mathcal{L}(W; \mathcal{B})  \\
g_A = \nabla_{A} \mathcal{L}(W; \mathcal{B})=\nabla_{A} \mathcal{L}(W_0 + BA; \mathcal{B}) \\
g_B = \nabla_{B} \mathcal{L}(W; \mathcal{B})=\nabla_{B} \mathcal{L}(W_0 + BA; \mathcal{B})
$$




## 1 Add nosiy only on the lora part



### Step 1: Compute clean gradient

**1.1** Forward and backward on clean weights:
$$
g_0 = \nabla_\theta \mathcal{L}(\theta)
$$

$$
g_{(0,A)} = \nabla_{A} \mathcal{L}(W; \mathcal{B})=\nabla_{A} \mathcal{L}(W_0 + BA; \mathcal{B}) \\
g_{(0,B)} = \nabla_{B} \mathcal{L}(W; \mathcal{B})=\nabla_{B} \mathcal{L}(W_0 + BA; \mathcal{B})
$$


------

### Step 2: Compute perturbed gradient

**2.1** Sample noise:
$$
\epsilon \sim \mathcal{N}(0, \sigma^2 \cdot \|\theta\|)
$$
**2.2** Add noise , here only add on Lora part
$$
A_{adv} =  A + \epsilon, B_{adv} = B + \epsilon \\
W_{advLoRA} =  W_0 + (B + \epsilon ) (A + \epsilon)
$$


**2.3** Evaluate perturbed loss and gradient:
$$
g_1 = \nabla_\theta \mathcal{L}(\theta + \epsilon) \\
$$

$$
g_{(1,A)} = \nabla_{A} \mathcal{L}(W_{advLoRA}; \mathcal{B})=\nabla_{A} \mathcal{L}(W_0 +(B + \epsilon ) (A + \epsilon); \mathcal{B}) \\
g_{(1,B)} = \nabla_{B} \mathcal{L}(W_{advLoRA}; \mathcal{B})=\nabla_{B} \mathcal{L}(W_0 +(B + \epsilon ) (A + \epsilon); \mathcal{B})
$$





------

### Step 3: Mixed gradient update

**3.1** Combine gradients:
$$
g = \lambda g_1 + (1 - \lambda) g_0
$$

$$
\begin{aligned}
g_A 
&= a \cdot  g_{(0,A)} + b \cdot g_{(1,A)}  \\ 
&= a \nabla_{A} \mathcal{L}(W; \mathcal{B})+ b \nabla_{A} \mathcal{L}(W_{advLoRA}; \mathcal{B}) \\
&= a \nabla_{A} \mathcal{L}(W_0 + BA; \mathcal{B}) + b  \nabla_{A} \mathcal{L}(W_0 +(B + \epsilon ) (A + \epsilon); \mathcal{B})  \\

\end{aligned}
$$


$$
\begin{aligned}
g_B 
&= a \cdot  g_{(0,B)} + b \cdot g_{(1,B)}  \\ 
&= a \nabla_{B} \mathcal{L}(W; \mathcal{B})+ b \nabla_{B} \mathcal{L}(W_{advLoRA}; \mathcal{B}) \\
&= a \nabla_{B} \mathcal{L}(W_0 + BA; \mathcal{B}) + b  \nabla_{B} \mathcal{L}(W_0 +(B + \epsilon ) (A + \epsilon); \mathcal{B})  \\

\end{aligned}
$$


**3.2** Update parameters:
$$
\theta \leftarrow \theta - \eta \cdot g
$$

$$
A \leftarrow A - \eta \cdot g_A \\
B \leftarrow B - \eta \cdot g_B 
$$







## 1 Add nosiy  the whole  model 



### Step 1: Compute clean gradient

**1.1** Forward and backward on clean weights:
$$
g_0 = \nabla_\theta \mathcal{L}(\theta)
$$

$$
g_{(0,A)} = \nabla_{A} \mathcal{L}(W; \mathcal{B})=\nabla_{A} \mathcal{L}(W_0 + BA; \mathcal{B}) \\
g_{(0,B)} = \nabla_{B} \mathcal{L}(W; \mathcal{B})=\nabla_{B} \mathcal{L}(W_0 + BA; \mathcal{B})
$$


------

### Step 2: Compute perturbed gradient

**2.1** Sample noise:
$$
\epsilon \sim \mathcal{N}(0, \sigma^2 \cdot \|\theta\|)
$$
**2.2** Add noise , try to on the whole model 
$$
A_{adv} =  A + \epsilon, B_{adv} = B + \epsilon \\
W_{advFull} =  (W_0+\epsilon) + (B + \epsilon ) (A + \epsilon) 
\sim W_0+ BA + \epsilon   
$$


**2.3** Evaluate perturbed loss and gradient:
$$
g_1 = \nabla_\theta \mathcal{L}(\theta + \epsilon) \\
$$

$$
g_{(1,A)} = \nabla_{A} \mathcal{L}(W_{advFull}; \mathcal{B})=\nabla_{A} \mathcal{L}(W_0 +BA + \epsilon); \mathcal{B}) \\
g_{(1,B)} = \nabla_{B} \mathcal{L}(W_{advFull}; \mathcal{B})=\nabla_{B} \mathcal{L}(W_0 +BA + \epsilon); \mathcal{B})
$$





------

### Step 3: Mixed gradient update

**3.1** Combine gradients:
$$
g = \lambda g_1 + (1 - \lambda) g_0
$$

$$
\begin{aligned}
g_A 
&= a \cdot  g_{(0,A)} + b \cdot g_{(1,A)}  \\ 
&= a \nabla_{A} \mathcal{L}(W; \mathcal{B})+ b \nabla_{A} \mathcal{L}(W_{advFull}; \mathcal{B}) \\
&= a \nabla_{A} \mathcal{L}(W_0 + BA; \mathcal{B}) + b  \nabla_{A} \mathcal{L}(W_0 +BA + \epsilon; \mathcal{B})  \\

\end{aligned}
$$


$$
\begin{aligned}
g_B 
&= a \cdot  g_{(0,B)} + b \cdot g_{(1,B)}  \\ 
&= a \nabla_{B} \mathcal{L}(W; \mathcal{B})+ b \nabla_{B} \mathcal{L}(W_{advLoRA}; \mathcal{B}) \\
&= a \nabla_{B} \mathcal{L}(W_0 + BA; \mathcal{B}) + b  \nabla_{B} \mathcal{L}(W_0 +BA + \epsilon; \mathcal{B})  \\

\end{aligned}
$$


**3.2** Update parameters:
$$
\theta \leftarrow \theta - \eta \cdot g
$$

$$
A \leftarrow A - \eta \cdot g_A \\
B \leftarrow B - \eta \cdot g_B 
$$































RWP 提出的目标函数是期望的 Bayes loss（对权重扰动的期望）：
$$
L_{\text{Bayes}}(w) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, \sigma^2)}[\mathcal{L}(w + \epsilon)]
$$

### 合损失函数（Mixed RWP）：

$$
L_m(w) = \lambda L_{\text{Bayes}}(w) + (1 - \lambda) L(w) \tag{12}
$$

其中：

- $L(w)$ 是标准的原始损失（non-perturbed）
- $L_{\text{Bayes}}(w)$ 是加扰动后、从高斯采样平均而来的平滑 loss
- $\lambda \in [0, 1]$ 控制这两部分的权重

**Step 1（g₀）**：
$$
g_0 = \nabla_\theta \mathcal{L}_t(\theta)
$$
**Step 2（g₁）**：
$$
\theta' = \theta + \epsilon,\quad \epsilon \sim \mathcal{N}(0, \sigma^2 \cdot \|\theta\|) \\
g_1 = \nabla_\theta \mathcal{L}_t(\theta')
$$
**混合梯度更新**：
$$
g = \lambda g_1 + (1 - \lambda) g_0 \\
\theta \leftarrow \theta - \eta g
$$


## RPW- local LoRA





2 . RPW- full model 

是的，在你当前的实现中，**全局增加扰动确实是**生效的，具体来说：

------

### ✅ **扰动是否“生效” —— 分析过程如下：**

#### 1. **扰动发生的位置**

你明确地在 `with torch.no_grad()` 作用域内对模型所有参数执行了：

```python
param.data.add_(noise)
```

这一步 **修改了参数本身的值**，即：

$\theta \leftarrow \theta + \epsilon$

这会直接作用于 **模型 forward 时的计算图**。即模型在执行 `loss_noisy = self.compute_loss(model, inputs)` 时，是使用扰动后的权重进行前向传播的。

------

#### 2. **loss 是基于扰动后的参数计算的**

这一步：

```python
with self.compute_loss_context_manager():
    loss_noisy = self.compute_loss(model, inputs)
```

确实是基于扰动后的 $w + \epsilon$ 计算损失：

- 所有主干参数 `W` 和 LoRA 参数 `θ_\text{LoRA}` 都临时变成了：`原值 + 噪声`

- 因此 `loss_noisy` 对应的是扰动后参数的 loss：

  L(w+ϵ)L(w + \epsilon)

------

#### 3. **扰动是否参与反向传播？**

扰动后的参数（虽然包含了主干参数 W 的改动）在反向传播时：

- PyTorch 只会计算 `requires_grad=True` 的参数的梯度；

- 所以虽然 `loss_noisy` 依赖于主干参数 $W + \epsilon$，但 **只会生成可训练参数 LoRA 的梯度**：

  ∇θLoRAL(w+ϵ)\nabla_{\theta_\text{LoRA}} L(w + \epsilon)

这正是 RWP 在 **LoRA 微调设定**中最核心的做法 ——

> 用“扰动整个网络”的方式获得更鲁棒的梯度，但只更新轻量参数。

------

### 🧠 结论：扰动是“生效”的，尽管更新仅限于 LoRA

| 项目                              | 是否成立 | 说明                                                   |
| --------------------------------- | -------- | ------------------------------------------------------ |
| **扰动参数被真实修改**            | ✅        | `param.data.add_(noise)` 修改了实际 forward 使用的权重 |
| **loss 反映的是扰动后模型性能**   | ✅        | `loss_noisy` 是在 $w + \epsilon$ 下计算的              |
| **backward 影响了扰动产生的梯度** | ✅        | 得到的是 $\nabla_{\theta_\text{LoRA}} L(w + \epsilon)$ |
| **主干权重不会更新**              | ✅        | 因为其 `requires_grad=False`，不会产生 `.grad`         |

------

如你所见，你的策略实际上很好地结合了 **loss landscape smoothness（通过扰动实现）** 与 **轻量更新（仅 LoRA）** 的目标。

是否还希望进一步测试扰动是否对 loss 有提升泛化的作用（如对比 $\mathcal{L}(w)$ 和 $\mathcal{L}(w + \epsilon)$ 的 flatness）？
