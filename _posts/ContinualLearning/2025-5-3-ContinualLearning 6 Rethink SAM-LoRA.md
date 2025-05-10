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
\min_W \max_{ \|\epsilon\| \leq \rho} L(W + \epsilon),
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
g = a g_0 + b g_1
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
g = a g_0 + b g_1
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



## Add noise only on the origin one

Even we can try add noise only on the origin model 

$$
W_{advOrigin} =  (W_0+\epsilon) + B   A  
\sim W_0+ BA + \epsilon
$$

### Step 2: Compute perturbed gradient

**2.1** Sample noise:
$$
\epsilon \sim \mathcal{N}(0, \sigma^2 \cdot \|\theta\|)
$$
Sample noise for only origin pertrained model parameters, that is frozen $W_0$:


$$
\epsilon_W \sim \mathcal{N}(0, \sigma^2 \cdot \|W_0 + BA\|_2)
$$
若将 $\epsilon_W$ 分解为对应结构的噪声项，则只有预训练模型的权重 添加噪声


$$
\epsilon_{W_0} \sim \mathcal{N}(0, \sigma^2 \cdot \|W_0\|_2)
$$



$$
\epsilon_{W_0} \sim \mathcal{N}(0, \sigma^2 ),
$$


**2.2** Add noise , try to on the whole model 


$$
W_{advOrigin} =  (W_0+\epsilon) + B  A  
\sim W_0+ BA + \epsilon
$$



**2.3** Evaluate perturbed loss and gradient:


$$
g_1 = \nabla_\theta \mathcal{L}(\theta + \epsilon) \\
$$

$$
g_{(1,A)} = \nabla_{A} \mathcal{L}(W_{advOrigin}; \mathcal{B})=\nabla_{A} \mathcal{L}(W_0 +BA + \epsilon); \mathcal{B}) \\
g_{(1,B)} = \nabla_{B} \mathcal{L}(W_{advOrigin}; \mathcal{B})=\nabla_{B} \mathcal{L}(W_0 +BA + \epsilon); \mathcal{B})
$$





------

### Step 3: Mixed gradient update

**3.1** Combine gradients:
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

很好，以下是你要求的**与代码高度契合、数学表达清晰统一**、带有**统计解释与设计思想**的完整英文版本。该版本采用你给出的统一框架：

------

## 🎯 **Unified Noise Injection Formulation**

All noise injection strategies in our framework can be unified under the following general form:

$\boxed{ \boldsymbol{\epsilon} = Z \odot S, \quad \text{where } Z_{ij} \sim \mathcal{N}(0, 1) }$

Here, $S$ is a **scaling matrix** with the same shape as the parameter tensor $W$ and defines the standard deviation of noise injected into each element $W_{ij}$. Different strategies specify $S$ differently to reflect structural, statistical, or task-aware properties.

------

## 🧩 Strategy 1: **Gauss_standard** — Uniform Noise

$S_{ij} = \sigma \quad \Rightarrow \quad \epsilon_{ij} \sim \mathcal{N}(0, \sigma^2)$

> Each parameter receives equal-magnitude isotropic noise, regardless of its position or magnitude.

**Statistical meaning:** Adds i.i.d. Gaussian perturbations to all weights, simulating uniform stochasticity across the model.

------

## 🧩 Strategy 2: **Gauss_matrix** — Matrix-Norm-Aware Noise

$S_{ij} = \sigma \cdot \|W\|_F \quad \Rightarrow \quad \epsilon_{ij} \sim \mathcal{N}(0, \sigma^2 \cdot \|W\|_F^2)$

> All parameters in the same matrix share a global scaling factor proportional to the Frobenius norm $|W|_F$.

**Statistical meaning:** Respects global parameter magnitude, injecting stronger noise in high-capacity layers.

------

## 🧩 Strategy 3: **Gauss_element** — Element-Wise Magnitude Scaling

$S_{ij} = \sigma \cdot |W_{ij}| \quad \Rightarrow \quad \epsilon_{ij} \sim \mathcal{N}(0, \sigma^2 \cdot W_{ij}^2)$

> Each weight receives noise proportional to its own magnitude.

**Statistical meaning:** Models heteroscedastic uncertainty at the parameter level.

------

## 🧩 Strategy 4: **lpf_sgd** — Row-Wise L2-Norm Scaling

$S_{ij} = \sigma \cdot \|W_{i,:}\| \quad \Rightarrow \quad \epsilon_{ij} \sim \mathcal{N}(0, \sigma^2 \cdot \|W_{i,:}\|^2)$

> Parameters in the same row share a noise scale, computed as the L2 norm of that row.

**Statistical meaning:** Regularizes whole feature directions or filter channels (rows), especially suitable for LoRA-A or linear layers.

------

## 🧩 Strategy 5: **mARWP_fisher** — Fisher-Aware Row-Wise Scaling

$S_{ij} = \frac{\sigma \cdot \|W_{i,:}\|}{\sqrt{1 + \eta \cdot F_i}} \quad \Rightarrow \quad \epsilon_{ij} \sim \mathcal{N}\left(0, \frac{\sigma^2 \cdot \|W_{i,:}\|^2}{1 + \eta \cdot F_i}\right)$

> Noise is modulated by both parameter magnitude (via row norms) and task sensitivity (via Fisher information), suppressing noise in highly confident directions.

------

### 🔬 Fisher Information Estimation

To compute the Fisher score $F_i$ for the $i$-th row, we use a diagonal approximation updated via exponential moving average (EMA) over squared gradients:

$\text{Fisher}_{ij}^{(t)} = \lambda \cdot \text{Fisher}_{ij}^{(t-1)} + \left(\frac{\partial \mathcal{L}}{\partial W_{ij}}\right)^2$

with momentum factor $\lambda \in [0, 1)$. The row-wise Fisher score is then aggregated as:

$F_i = \sum_{j=1}^d \text{Fisher}_{ij}$

and used in scaling:

$S_{ij} = \frac{\sigma \cdot \|W_{i,:}\|}{\sqrt{1 + \eta \cdot F_i}}$

**Statistical meaning:** Injects less noise in important task-relevant directions (high Fisher), while exploring more in uncertain directions — promoting robustness and sharpness control.





# To explore 

we using RWP  under DDP ,



if only inject noise in lora part 
$$
\begin{aligned}
g_A 
&= a \cdot  g_{(0,A)} + b \cdot g_{(1,A)}  \\ 
&= a   \frac{1}{N}\sum_{i=1}^N \nabla_{A} \mathcal{L}(W; \mathcal{B_i})+ b \nabla_{A} \mathcal{L}(W_{advLoRA}; \mathcal{B_N}) \\
&= a  \frac{1}{N}\sum_{i=1}^N\nabla_{A} \mathcal{L}(W_0 + BA; \mathcal{B_i}) + b  \nabla_{A} \mathcal{L}(W_0 +(B + \epsilon ) (A + \epsilon); \mathcal{B_N})  \\

\end{aligned}
$$

$$
\begin{aligned}
g_A 
&= a \cdot  g_{(0,A)} + b \cdot g_{(1,A)}  \\ 
&= a  \frac{1}{N}\sum_{i=1}^N\nabla_{A} \mathcal{L}(W; \mathcal{B}_i)+ b \nabla_{A} \mathcal{L}(W_{advFull}; \mathcal{B}_N) \\
&= a \frac{1}{N}\sum_{i=1}^N \nabla_{A} \mathcal{L}(W_0 + BA; \mathcal{B}_i) + b  \nabla_{A} \mathcal{L}(W_0 +BA + \epsilon; \mathcal{B}_N)  \\

\end{aligned}
$$




## Q1  Is  full-noise better than only lora-noise?

Obs :  Full better than SAM

We using DDP setting , n=2,  noise is the same using Gaussian Noise



hyparmater 

n=2

a b =0.5

Common setting

```sh
#lora setting 
--lora_strategy Inclora \
--lora_dim 8 \
# traing setting 
--learning_rate 1e-03 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--num_train_epochs 1 \
--optimizer_type adamw_hf \
# noise setting
--rwp_a 0.5\
--rwp_b 0.5 \
--rwp_noise_type Gauss_standard \
--noise_std 0.0015 \
```



Different 

```sh
--rwp_type RWP_full \
```





```sh
--rwp_type RWP_lora \
```



## Q2 which part Loss is important ?



  Obs :  Full better than SAM

We using DDP setting , n=2,  noise is the same using Gaussian Noise



hyparmater 

n=2

a b =0.5

Common setting



```sh
#lora setting 
--lora_strategy Inclora \
--lora_dim 8 \
# traing setting 
--learning_rate 1e-03 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--num_train_epochs 1 \
--optimizer_type adamw_hf \
# noise setting
--rwp_type RWP_full \
--rwp_noise_type Gauss_standard \
--noise_std 0.0015 \
```



Different 

```sh

--rwp_a 0.5\
--rwp_b 0.5 \

```











## Q3  which noise is better in full-RWP?





# Describe Flat





The object of the flat Loss : 




$$
\begin{equation}
\min_W \max_{ \|\epsilon\|_p \leq \rho} L(W + \epsilon;\mathcal{B}),
\end{equation}
$$

$$
\begin{equation}
\min _{W} 
\underset{\epsilon \sim \mathcal{N}\left(0, \sigma^2 \mathbf{I}\right)}
{\mathbb{E}} L(W+\boldsymbol{\epsilon})
\end{equation}
$$


How to visual the loss land 

Letting $v$ be a direction in weight space, it is normalized as
$$
\bar{v} = \frac{v}{\|v\|} \cdot \|W\|
$$
here ,  Regularizes whole feature directions or filter channels (rows) in paramter
$$
\|w_{ij}\| =  \|W_{i,:}\| =\sqrt{\sum_{k}^n w_{i,k}^2}
$$
Specificlly, for a linear layer $l = Wx+B$ ,  the weight is $m \times n$ , the input is $m$ , the output is $n$ , the random direction $V$  in this layer is  also $m \times n $ , it is generated as follows: 
$$
v_{ij} \sim \mathcal{N}(0, \sigma^2 \cdot \|W_{i,:}\|^2)
$$


Then, the loss landscape is visualized in discrete steps along this direction, 
$$
L(W+V \cdot s;\mathcal{B})
$$
where $s \in [-1,1]$ is the step  along the perturbation direction 

and the input is the test dataset $\mathcal{B}$  , it reflect the model's genera ablity in  test dataset



**Average-Case / Random Flatness:** 



Considering random weight perturbations  $V$ within the $\rho $  neighborhood of $W$, average-case flatness is computed as
$$
\begin{equation}
\begin{aligned}
\underset{V \sim \mathcal{N}\left(0, \sigma^2 \mathbf{I}\right)}
{\mathbb{E}}

\left[ 


\mathcal{L}( W+s\cdot V ; \mathcal{B}) -\quad  \mathcal{L}(W;\mathcal{B}) \right]  

\end{aligned}

\end{equation}
$$
where $V$ is just a random direction, and $s$ is the step , with constracion
$$
 \| s \cdot V \|  \leq \rho 
$$


in fact we can't get the all random direction related with matrix weight $W$ , we just smaple $N$ times to MC similar it ,as follows: 
$$
\sim  \frac{1}{M}\sum^M_{i} 
\  \left[
  \mathcal{L}( W+  V_i ; \mathcal{B}) -\   \mathcal{L}(W;\mathcal{B}) \right]
$$


**Worst-Case**  flatness  change in the worst  direction , which is also the define of sharpness:


$$
\  \left[
\max_{V_i \sim \mathcal{N}\left(0, \sigma^2 \mathbf{I}\right)}
  \mathcal{L}( W+  V_i ; \mathcal{B}) -\   \mathcal{L}(W;\mathcal{B}) \right]


\sim  \max_{i\in \{1,\dots,M\}}
\  \left[
  \mathcal{L}( W+  V_i ; \mathcal{B}) -\   \mathcal{L}(W;\mathcal{B}) \right]
$$


and the other defination is as follow: 







After we get one most worst direction $V_i$ , we can do  Regularization to get another direction  $V_i$ ,  and $V_i \perp V_j$



so we can  show  the lossland in 2-direciton 
$$
L=(W +a \times V_i+b\times V_j; \mathcal{B})
$$




非常好，以下是**润色后的英文表述**，采用连贯流畅的学术论文风格，避免列点结构，适合直接作为论文中“方法”章节的一部分。整体结构依然遵循您要求的三部分框架：扰动空间定义、平坦性刻画、景观可视化。

------

## Mathematical Characterization and Visualization of Loss Flatness and Landscape

Understanding the geometry of the loss landscape is essential for analyzing the generalization and stability of deep neural networks. In this section, we present a unified framework to define weight perturbations, formally characterize the flatness of the loss surface, and visualize its geometric structure. The formulation draws upon both optimization-theoretic and Bayesian interpretations to provide a comprehensive view.

### 1. Perturbation Neighborhood in Weight Space

From a mathematical perspective, the neighborhood of a model parameter $W$ can be interpreted through two complementary lenses. In the optimization view, the neighborhood is defined as a norm-constrained region centered around $W$, denoted as:

$\mathcal{N}_\rho(W) = \{ W' \in \mathbb{R}^n \mid \|W' - W\|_p \leq \rho \},$

where $\rho > 0$ specifies the radius of the neighborhood and the choice of norm $p$ determines the shape of the constraint region.

Alternatively, from a Bayesian or geometric standpoint, a perturbed weight can be viewed as the result of applying a step along a stochastic direction. This formulation expresses the perturbed weights as:

$W' = W + s \cdot V,$

where $V$ is a random direction sampled from a predefined distribution, such as an isotropic or structured Gaussian, and $s \in \mathbb{R}$ is a scalar that controls the perturbation magnitude. To ensure that comparisons across directions remain meaningful, normalization is typically applied to $V$. A global normalization rescales the direction as $\bar{V} = \frac{V}{|V|} \cdot |W|$, maintaining the same scale as the original weights. In practice, especially for layers with matrix-shaped parameters, a filter-wise normalization may be used, where each row of the weight matrix $W \in \mathbb{R}^{m \times n}$ is perturbed independently, with variance scaled by the norm of that row:

$v_{ij} \sim \mathcal{N}\left(0, \sigma^2 \cdot \|W_{i,:}\|^2 \right).$

This directional decomposition forms the basis for both theoretical analysis and empirical evaluation of the loss surface.

### 2. Defining Flatness of the Loss Landscape

Flatness refers to the sensitivity of the loss function to small perturbations in the model parameters. Intuitively, a flatter region in the loss landscape corresponds to a wider local minimum where the loss remains stable under perturbations, while a sharper region indicates higher sensitivity and potential overfitting.

The first perspective considers the **average-case flatness**, which quantifies the expected increase in loss under stochastic perturbations. Formally, it is defined as:

$\phi_{\text{avg}}(W) = \mathbb{E}_{V \sim \mathcal{N}(0, \sigma^2 I)} \left[ \mathcal{L}(W + s \cdot V; \mathcal{B}) - \mathcal{L}(W; \mathcal{B}) \right],$

where $\mathcal{B}$ denotes the evaluation dataset and $s$ is the perturbation step size. In practical implementations, this expectation is approximated by Monte Carlo sampling over $M$ realizations:

$\phi_{\text{avg}}(W) \approx \frac{1}{M} \sum_{i=1}^M \left[ \mathcal{L}(W + V_i; \mathcal{B}) - \mathcal{L}(W; \mathcal{B}) \right].$

This formulation can be interpreted as a Bayesian marginalization over local weight uncertainty and captures the typical sensitivity of the model.

In contrast, the **worst-case flatness**, or sharpness, evaluates the maximal increase in loss over the neighborhood. This corresponds to the adversarial sensitivity of the model, and is defined as:

$\phi_{\text{worst}}(W) = \max_{\|V\| \leq \rho} \left[ \mathcal{L}(W + V; \mathcal{B}) - \mathcal{L}(W; \mathcal{B}) \right].$

Since exact maximization is intractable, it is approximated by sampling and selecting the worst-performing perturbation:

$\phi_{\text{worst}}(W) \approx \max_{i \in \{1, \dots, M\}} \left[ \mathcal{L}(W + V_i; \mathcal{B}) - \mathcal{L}(W; \mathcal{B}) \right].$

This quantity serves as a practical surrogate for the leading eigenvalue of the loss Hessian and has been widely adopted as a sharpness metric (Keskar et al., 2017).

While average-case and worst-case formulations capture different aspects of local geometry, both are closely related under quadratic loss approximations. In such cases, the worst-case flatness relates to the spectral norm of the Hessian, whereas the average-case flatness corresponds to its trace, offering complementary insights into the curvature structure of the loss landscape.

### 3. Visualization of the Loss Landscape

To better interpret the geometric structure of the loss surface, it is helpful to visualize how the loss value changes when perturbations are applied in specific directions. Such visualization not only supports theoretical analysis but also provides empirical insights into model robustness and generalization behavior.

In one-dimensional visualization, a single normalized direction $V$ is selected, and the loss is evaluated along a range of perturbation magnitudes:

$L(s) = \mathcal{L}(W + s \cdot V; \mathcal{B}), \quad s \in [-1, 1].$

This produces a scalar-valued curve that reflects the curvature of the loss function along the selected direction. A flat curve indicates robustness to perturbations, while a steep curve suggests sensitivity.

For a more comprehensive view, two-dimensional visualization can be performed by choosing two orthogonal directions, $V_1$ and $V_2$, and plotting the loss surface over a perturbation grid:

$L(a, b) = \mathcal{L}(W + a \cdot V_1 + b \cdot V_2; \mathcal{B}), \quad a, b \in [-1, 1].$

The resulting surface can be rendered as either a contour plot or a 3D surface plot, providing visual evidence of the sharpness or flatness of local minima. This technique is especially informative when comparing optimization methods or regularization strategies.

To ensure the visualization reflects generalization behavior rather than overfitting, it is customary to use a held-out dataset (e.g., validation or test set) as the input $\mathcal{B}$ for loss evaluation. Empirically, flatter minima in the visualized landscape are often associated with better performance on unseen data, highlighting the practical value of this analysis.

------

### Summary

This section provides a unified mathematical formulation for characterizing and visualizing loss landscape geometry. By decomposing weight perturbations into direction and magnitude, formalizing flatness through both expectation and extremal analysis, and visualizing loss responses in low-dimensional subspaces, we offer a framework that is both theoretically grounded and practically applicable. These tools serve as diagnostic instruments for understanding model robustness and guiding the design of more generalizable learning systems.

------

如需，我可以将上述内容转化为高质量的 LaTeX 格式，带公式编号、图示建议和引用结构，是否需要继续？





# LoRA - Noise


$$
\mathbf{W}_{\text{noisy}} = \mathbf{W} + \mathbf{Z} \odot \mathbf{S} \\

Z∼\mathcal{N}(0,1)^{m×n} , S \in R^{m\times n}
$$



$$
\mathbf{W}_{\text{noisy}} = \mathbf{W} + \mathbf{Z} \odot \left( \text{std} \cdot R \right) \\
$$




## LPF-SGD

$$
R = \mathbf{r} \cdot \mathbf{1}_n^\top \\

\mathbf{r} = 
\begin{bmatrix}
r_1 \\
r_2 \\
\vdots \\
r_m
\end{bmatrix}
\in \mathbb{R}^{m \times 1} , r_i = \left\| \mathbf{W}_i \right\|_2 = \sqrt{ \sum_{j=1}^n W_{ij}^2 }, \quad \forall i = 1, 2, \dots, m
$$



## 2 Gaussian Noise 

$$
\mathbf{W}_{\text{noisy}} = \mathbf{W} + \sigma \cdot \mathbf{Z}, \quad \mathbf{Z} \sim \mathcal{N}(0, 1)^{m \times n}
$$





To better understand the role of LoRA in model adaptation, we contrast it with a commonly used noise injection strategy. A general form of additive noise perturbation can be written as:
$$
\begin{equation}
\mathbf{W}_{\text{noisy}} = \mathbf{W}_0 + \mathbf{Z} \odot \mathbf{S},
\end{equation}
$$
where $\( \mathbf{W}_0 \in \mathbb{R}^{d_o \times d_i} \)$ is the original weight matrix, $ \( \mathbf{Z} \sim \mathcal{N}(0, 1)^{d_o \times d_i} \)$ is element-wise Gaussian noise, and $\( \mathbf{S} \in \mathbb{R}^{d_o \times d_i} \)$ is a scaling matrix that modulates the variance, often structured based on weight norms or other heuristics.

In contrast, the weight update in LoRA is formulated as a low-rank decomposition:
$$
\begin{equation}
\Delta \mathbf{W} = k \cdot \mathbf{B} \mathbf{A},
\end{equation}
$$
where  $\( \mathbf{B} \in \mathbb{R}^{d_o \times r} \) $,  $\( \mathbf{A} \in \mathbb{R}^{r \times d_i} \)$, and $ \( k \in \mathbb{R} \) $ is a scalar or diagonal scaling matrix. Notably, $\( \mathbf{A} \)$ is initialized with Gaussian noise, while $\( \mathbf{B} \) $is initialized to zero, making$ \( \Delta \mathbf{W} \)$ initially a structured random perturbation.

This formulation reveals a key connection: **\textit{LoRA can be reinterpreted as injecting structured, low-rank noise into the parameter space}**, where the low-rank structure imposes an implicit constraint on the perturbation directions. Specifically, the matrix $\( \mathbf{A} \) $projects the input into a latent subspace where random noise is introduced, and$ \( \mathbf{B} \) $acts as a decoder that maps this noise back into the output space. During training, both$ \( \mathbf{A} \) $ and $ \( \mathbf{B} \) $are updated to capture task-relevant variations within this constrained space. The scaling factor $ \( k \)  $plays a role analogous to the standard deviation  $ \( \sigma \)  $in noise-based methods, modulating the overall magnitude of the perturbation.

From this perspective, LoRA does not merely act as a parameter-efficient adaptation mechanism—it can also be seen as a **\textit{learnable structured perturbation scheme}**, where the perturbation space is both informed by prior noise and optimized for downstream performance. This interpretation bridges LoRA with sharpness-aware or noise-injection optimization techniques, offering a unified view on generalization-driven fine-tuning.





We reinterpret the application of Gaussian noise in multi-task continual LoRA settings as a form of structured perturbation over different subspaces of the model. In this setup, each task maintains a separate low-rank adapter $\Delta W_t = B_t A_t$, and previously learned adapters are frozen to mitigate forgetting. We investigate three distinct noise injection strategies and analyze their impact:

1. **LoRA-only noise injection**: Perturbations are added only to the trainable LoRA components $A$ and $B$. This corresponds to:
   $$
   W_{\text{advLoRA}} = W_0 + (B + \epsilon)(A + \epsilon),
   $$
   which effectively models noise constrained to the low-rank adaptation subspace. However, this setup ignores the base model, limiting the perturbation’s influence on broader representations.

2. **Full-model noise injection**: All parameters, including the frozen base model $W_0$, are perturbed:
   $$
   W_{\text{advFull}} = (W_0 + \epsilon) + (B + \epsilon)(A + \epsilon) \approx W_0 + BA + \epsilon.
   $$
   This introduces full-rank noise across the model space, potentially improving generalization but also increasing instability due to unconstrained perturbations.

3. **Origin-only noise injection**: Noise is injected exclusively into the frozen pre-trained weights:
   $$
   W_{\text{advOrigin}} = (W_0 + \epsilon) + BA,
   $$
   which preserves the structure of task-specific LoRA parameters while enhancing the robustness of the base model. Empirically, this strategy yields the best performance, suggesting that selectively perturbing the fixed representation space is most effective in continual learning settings.

These results support the view that LoRA serves as a structured, learnable perturbation module, while noise injection over $W_0$ functions as a form of “frozen-space regularization.” Combining both enables robust adaptation while maintaining task-specific expressiveness. 





\paragraph{Gaussian Noise Injection.}
In this setting, each element in the parameter matrix is perturbed by an independent Gaussian variable with a fixed standard deviation:
$$
\begin{equation}
\mathbf{W}_{\text{noisy}} = \mathbf{W} + \sigma \cdot \mathbf{Z}, \quad \mathbf{Z} \sim \mathcal{N}(0, 1)^{m \times n}
\end{equation}
$$
This corresponds to element-wise additive white Gaussian noise (AWGN), where all weights are equally disturbed, regardless of their magnitude or structure.

\paragraph{LPF-SGD (Structure-Aware Noise).}
As a comparison, LPF-SGD introduces row-wise scaled Gaussian noise proportional to the L2-norm of each row:
$$
\begin{align}
\mathbf{r}_i &= \left\| \mathbf{W}_i \right\|_2, \quad \forall i = 1, \dots, m \\
\mathbf{R} &= \mathbf{r} \cdot \mathbf{1}_n^\top \in \mathbb{R}^{m \times n} \\
\mathbf{W}_{\text{noisy}} &= \mathbf{W} + \mathbf{Z} \odot \left( \sigma \cdot \mathbf{R} \right), \quad \mathbf{Z} \sim \mathcal{N}(0, 1)^{m \times n}
\end{align}
$$
This structured noise respects the per-row sensitivity of parameters and has been shown to improve generalization and robustness.



\paragraph{Mode 1: Standard Gaussian Noise (Unstructured).}

Each parameter is perturbed by i.i.d. Gaussian noise with a fixed standard deviation \( \sigma \), independent of the parameter value:

$$
\begin{equation}
\epsilon_{ij} \sim \mathcal{N}(0, \sigma^2), \quad 
\mathbf{W}_{\text{noisy}} = \mathbf{W} + \epsilon
\end{equation}
$$
This formulation does \emph{not} consider the magnitude or structure of \( \mathbf{W} \), and serves as a baseline for uniform perturbation.



 分支 SAM 和 Gaussian 区别 




$$
\mathbf{W}_{\text{noisy}} = \mathbf{W} + \mathbf{Z} \odot \mathbf{S} \\

Z∼\mathcal{N}(0,1)^{m×n} , S \in R^{m\times n} \\
$$





$$
\epsilon_{ij} \sim \mathcal{N}(0, \sigma^2)
$$
