---
layout: post
title:  Continual Learning  7 SAM-LoRA
categories: [Continual Learning,  math, MachineLearning, LoRA ]
description: 
keywords: [Continual Learning, math, MachineLearning, LoRA ]
---

# Continual Learning  6 SAM-LoRA



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

**1.1** compute the  origin model's loss and gradient 


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


where $\cdot$ denotes the element-wise norm and scaling.



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

When using **gradient accumulation** with accumulation steps $N$, the effective update at each global optimization step is based on the cumulative gradient:


$$
g = \sum_{i=1}^{N} \nabla_{W} \mathcal{L}(W; \mathcal{B}_i),
$$


where each $\mathcal{B}_i$ denotes a mini-batch of data.

In practice, to reduce computational and communication overhead in DDP, **SAM only perturbs the weights during the last accumulation step (step $N$)**, using **only the final mini-batch $\mathcal{B}_N$**.



1 calculate the gradient in the N step 


$$
\mathcal{L}(W; \mathcal{B}) \\
g = \sum_{i=1}^{N} \nabla_{W} \mathcal{L}(W; \mathcal{B_i})
$$


**1.2** Then, it calculates the normalized perturbation $\epsilon$ in the direction of the gradient:


$$
\begin{equation}
    \epsilon^* = 
    \rho \frac{g}{\|g\|_2}
    
    =\rho \frac{\sum_{i=1}^{N} \nabla_{W} \mathcal{L}(W; \mathcal{B_i})}{\|\sum_{i=1}^{N} \nabla_{W} \mathcal{L}(W; \mathcal{B_i})\|_2}.
    \end{equation}
$$


**1.3** Add noise $\epsilon $ on the origin model 
$$
W_{adv} =  W + \epsilon^*
$$

### **Step 2  Gradient descent at the adversarial point**

**2.1** Keeping $W_{\text{adv}}$ fixed, SAM evaluates the loss again,only using the last batch


$$
\mathcal{L}(W+\epsilon^*; \mathcal{B}_N)
$$


**2.2** Computes the gradient at the perturbed point


$$
\begin{equation}
   g_{\text{adv}}= \nabla_{W} \mathcal{L}(W + \epsilon^*; \mathcal{B}_N).
\end{equation}
$$


**2.3** Finally, SAM restores the original parameters $W$, and applies a standard optimizer step using $g_{\text{adv}}$:


$$
\begin{equation}
W \leftarrow W - \eta \cdot g_{\text{adv}},
\end{equation}
$$


where $\eta$ is the learning rate.







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



# LPF-SGD 



## Intro 

![image-20250503193124458](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250503193124755.png)



![image-20250503193229933](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250503193230204.png)



## Imply Step 

### Step 1 Generate Noise 



**1  Uniform Noise (Gaussian distribution)** 


$$
K \sim \mathcal{N}(0, \gamma \sum)
$$


$\mathcal{N}(0, \gamma \sum)$ in the algorithm stands for a sample from a Gaussian distribution, which means 


$$
\epsilon_{ij} \sim \mathcal{N}(0,  \gamma  \cdot \sigma^2)
$$


**2 filter-noise**

To be more specific,  $K \sim \mathcal{N}(0, \gamma \sum)$   and we set the matrix $\sum $ to be proportional to the norm of the parameters in each filter of the network, i.e., we set $Σ = diag(||θ^t_1||, ||θ^t_2|| · · · ||θ^t_k||)$, where $θ^t_k$ is  the weight matrix of the $k^{th}$ filter at iteration $t$ and $γ$ is the LPF radius.

Finally, we increase $γ$ during network training to progressively increase the area of the loss surface that is explored at each gradient update. This is done according to the following rule: 
$$
 γ_t = γ_0(α/2(− cos(tπ/T ) + 1) + 1)
$$
 where $T$ is total number of iterations (epochs * gradient updates per epoch) and $α$ is set such that $\gamma_{T} = (α + 1) ∗ γ_0$. This policy can be justified by the fact that the more you progress with the training, the more you care to recover flat regions in the loss landscape.


$$
K \sim \mathcal{N}(0, \gamma_t \cdot  diag(\|W_{1,:}\|, \|W_{2,:}\| \dots, \|W_{k,:}\|))
$$

$$
\epsilon_{ij} \sim \mathcal{N}\left(0,  \gamma_t  \cdot \right \|W_{i,:}\|_2)
$$


where $W_{i,:}$  is  the $i$-th row of $W$ (i.e., the $i$-th filter)

Noise scale depends on the norm of each filter. Filters with higher L2 norm receive proportionally larger noise.



### Step 2  Add noise  and calculate the Loss


$$
\mathcal{L}(\theta + \epsilon; \mathcal{B})
$$


### Step3 Calculate the gradient 

$$
g = \nabla_\theta \mathcal{L}(\theta + \epsilon; \mathcal{B})
$$









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
\epsilon \sim \mathcal{N}(0, \sigma^2 )
$$


**2.2** Add noise to the model and evaluate perturbed loss


$$
\mathcal{L}(\theta + \epsilon;\mathcal{B})
$$


 **2.3** Calculate  perturbed gradient:
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
g = \lambda g_N + (1 - \lambda) \sum_{i=1}^{N} g_i 
 = \lambda \nabla_\theta \mathcal{L}(\theta + \epsilon;\mathcal{B}_N) +  \sum_{i=1}^{N}(1 - \lambda) \nabla_\theta \mathcal{L}(\theta; \mathcal{B_i})
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


Sample independent noise for LoRA components:


$$
\epsilon_A \sim \mathcal{N}(0, \sigma^2 \cdot \|A\|_2), \quad \epsilon_B \sim \mathcal{N}(0, \sigma^2 \cdot \|B\|_2)
$$



$$
\epsilon_A \sim \mathcal{N}(0, \sigma^2 ), \quad \epsilon_B \sim \mathcal{N}(0, \sigma^2 )
$$


也可以使用更细粒度的 element-wise 噪声版本（如果需要）：


$$
[\epsilon_A]_{ij} \sim \mathcal{N}(0, \sigma^2 \cdot |[A]_{ij}|^2), \quad [\epsilon_B]_{ij} \sim \mathcal{N}(0, \sigma^2 \cdot |[B]_{ij}|^2)
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
Sample noise for full model parameters, including frozen $W_0$:


$$
\epsilon_W \sim \mathcal{N}(0, \sigma^2 \cdot \|W_0 + BA\|_2)
$$
若将 $\epsilon_W$ 分解为对应结构的噪声项，可写作：


$$
\epsilon_{W_0} \sim \mathcal{N}(0, \sigma^2 \cdot \|W_0\|_2), \quad
\epsilon_A \sim \mathcal{N}(0, \sigma^2 \cdot \|A\|_2), \quad
\epsilon_B \sim \mathcal{N}(0, \sigma^2 \cdot \|B\|_2)
$$



$$
\epsilon_{W_0} \sim \mathcal{N}(0, \sigma^2 ), \quad
\epsilon_A \sim \mathcal{N}(0, \sigma^2 ), \quad
\epsilon_B \sim \mathcal{N}(0, \sigma^2 )
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









# Flat -LoRA



purpose 




$$
\begin{equation}
\min _{\mathbf{A}, \mathbf{B}} \underset{\epsilon \sim \mathcal{N}\left(0, \sigma^2 \mathbf{I}\right)}{\mathbb{E}} L(\mathbf{W}+s \cdot \mathbf{B} \mathbf{A}+\boldsymbol{\epsilon})
\end{equation}
$$


which is equal to 
$$
\min _{\mathbf{A}, \mathbf{B}}  \mathcal{L}_{\text{Bayes}}(W) 
= 
\min _{\mathbf{A}, \mathbf{B}}  \mathbb{E}_{\epsilon \sim \mathcal{N}(0, \sigma^2)} [\mathcal{L}(W_0+AB + \epsilon)]
$$


## Calculate Step 

### Step1 Generate noise

the noise inject 

​	
$$
{ 
\epsilon \sim \mathcal{N}
\left(0, \frac{\sigma^2}{n} \cdot \mathrm{diag}\left(\|W'_{1,:}\|_2^2, \dots, \|W'_{m,:}\|_2^2\right) \cdot \mathbf{1}_{m \times n} \right)

}
$$


即对每个输出 filter $i$，整个 row $W'_{i,:}$ 的噪声满足：


$$
\epsilon_{i,:} \sim \mathcal{N}\left(0, \frac{\sigma^2}{n} \cdot \|W'_{i,:}\|_2^2 \cdot \mathbf{1}_{n} \right)
$$
1 **Filter-aware**：对每个输出通道（filter）赋予不同扰动幅度，$|W'_i|_2^2$ 越大，对其加的噪声越强；

2 **Input-dim-aware**：所有元素的总扰动幅度不随输入维度 $n$ 增加，采用 $1/n$ 缩放。



**Step2  Add noise on the whole model**


$$
A_{adv} =  A + \epsilon, B_{adv} = B + \epsilon \\
W_{advFull} =  (W_0+\epsilon) + (B + \epsilon ) (A + \epsilon) 
\sim W_0+ BA + \epsilon
$$


### Step 3   **Calculate perturbed gradient**


$$
g_{A} = \nabla_{A} \mathcal{L}(W_{advFull}; \mathcal{B})=\nabla_{A} \mathcal{L}(W_0 +BA + \epsilon); \mathcal{B}) \\
g_{B} = \nabla_{B} \mathcal{L}(W_{advFull}; \mathcal{B})=\nabla_{B} \mathcal{L}(W_0 +BA + \epsilon); \mathcal{B})
$$


# Noise Inject 

Absolutely. Here's the **complete English version** of the unified summary for the noise injection strategies discussed above, presented in a structured and consistent format suitable for inclusion in a paper’s **method section** or **appendix**.

------

# ✅ Unified Formulation of Noise Injection Strategies

This section presents a unified mathematical formulation for various random weight perturbation (RWP) strategies, including filter-aware, element-wise, and Fisher-scaled variants.

------

## 🔁 Notation

Let:

- $W \in \mathbb{R}^{m \times n}$: a weight matrix, with $m$ filters (rows) and $n$ input dimensions per filter;
- $W_{i,:}$: the $i$-th row of $W$ (i.e., the $i$-th filter);
- $\epsilon \in \mathbb{R}^{m \times n}$: the injected perturbation;
- $\sigma$: base noise scale hyperparameter;
- $\gamma$: filter-aware scaling factor;
- $\eta$: Fisher regularization strength;
- $F_{ij}$: Fisher information at position $(i,j)$;
- $|W|_F$: Frobenius norm of $W$;
- $\mathcal{N}(0, v^2)$: Gaussian distribution with mean 0 and variance $v^2$;
- $\odot$: element-wise (Hadamard) product.

------

## 🧮 General Expression

All noise injection schemes can be unified under the following general form:

$\boxed{ \epsilon = Z \odot S, \quad \text{where } Z_{ij} \sim \mathcal{N}(0, 1) }$

Here, $S$ defines the **scaling matrix**, controlling the standard deviation of the perturbation at each position. We specify $S$ differently for each method below.

------

## 🧩 Strategy 1: Standard RWP (Uniform Noise)

$S_{ij} = \sigma \quad \Rightarrow \quad \epsilon_{ij} \sim \mathcal{N}(0, \sigma^2)$

> Every parameter receives equal, isotropic noise.

------

## 🧩 Strategy 2: Matrix-norm-aware Scaling

$S_{ij} = \sigma \cdot \|W\|_F \quad \Rightarrow \quad \epsilon_{ij} \sim \mathcal{N}(0, \sigma^2 \cdot \|W\|_F^2)$

> Entire matrix shares a uniform noise scale based on its overall magnitude.

------

## 🧩 Strategy 3: Element-wise Scaling

$S_{ij} = \sigma \cdot |W_{ij}| \quad \Rightarrow \quad \epsilon_{ij} \sim \mathcal{N}(0, \sigma^2 \cdot W_{ij}^2)$

> Each parameter receives a noise scale proportional to its own absolute value.

------

##  Strategy 4: **Filter-wise Gaussian distribution (LPF-SGD)**

$S_{ij} = \sqrt{\gamma_t} \cdot \|W_{i,:}\|_2 \quad \Rightarrow \quad \epsilon_{ij} \sim \mathcal{N}\left(0, \gamma_t \cdot \|W_{i,:}\|_2^2\right)$

> Noise scale depends on the norm of each filter. Filters with higher L2 norm receive proportionally larger noise.



## 🧩 Strategy 5: **Adaptive Random Weight Perturbation** **(RWP)**

This can be used **in combination** with any of the above strategies to reduce noise in highly sensitive directions.

Let $S_{ij}^{\text{base}}$ be the original scaling (from strategies 1–4). Then:



$S_{ij}^{\text{fisher}} = \frac{S_{ij}^{\text{base}}}{\sqrt{1 + \eta \cdot F_{ij}}} \quad \Rightarrow \quad \epsilon_{ij} \sim \mathcal{N}\left(0, \frac{(S_{ij}^{\text{base}})^2}{1 + \eta \cdot F_{ij}} \right)$



> Fisher information suppresses perturbation in directions with high sensitivity.





## 🧩 Strategy 6: **Effective Random Perturbation** **(Flat-****LoRA****)**

$S_{ij} = \frac{\sigma \cdot \|W_{i,:}\|_2}{\sqrt{n}} \quad \Rightarrow \quad \epsilon_{ij} \sim \mathcal{N}\left( 0, \frac{\sigma^2 }{n} \|W_{i,:}\|_2^2  \right)$

> Noise scale depends on the norm of each filter. Filters with higher L2 norm receive proportionally larger noise.
>
> 



------

## ✅ Summary Table

| Strategy              | Scaling Term $S_{ij}$                                  | Distribution $\epsilon_{ij}$ | Structure-aware | Fisher-aware                |
| --------------------- | ------------------------------------------------------ | ---------------------------- | --------------- | --------------------------- |
| Standard RWP          | $\sigma$                                               | $\mathcal{N}(0, \sigma^2)$   | ❌ No            | ❌ No                        |
| Filter-aware RWP      | $\dfrac{\gamma \cdot |W_{i,:}|_2}{n}$                  | $\mathcal{N}(0, (\cdot)^2)$  | ✅ Row-wise      | ❌ No                        |
| Matrix-norm-aware     | $\sigma \cdot |W|_F$                                   | $\mathcal{N}(0, (\cdot)^2)$  | ✅ Global        | ❌ No                        |
| Element-wise scaling  | $\sigma \cdot                                          | W_{ij}                       | $               | $\mathcal{N}(0, (\cdot)^2)$ |
| Fisher-scaled variant | $\dfrac{S_{ij}^{\text{base}}}{\sqrt{1 + \eta F_{ij}}}$ | $\mathcal{N}(0, (\cdot)^2)$  | ✅ (Inherited)   | ✅ Yes                       |

------

This unified formulation and notation enables a clear comparison of noise injection mechanisms and can be directly adopted in your method section or appendix.

Would you like a LaTeX-ready version of this table and equations?
