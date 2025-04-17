---
layout: post
title:  Paper Reading 17 LLM finetuning and forgetting - 6 Flat Minima
categories: [Paper Reading,  LLM, Continual Learning,] 
description:  [Learn or Recall? Revisiting Incremental Learning with Pre-trained Language Models]
keywords: [Paper Reading,  LLM, Continual Learning,  Flat Minima] 
---



# Paper Reading 17  LLM finetuning and forgetting - 6 Flat Minima





# Flat minima Paper List 





| 分类        | 序号 | 标题                                                         | 翻译                             | 期刊/会议 | 年份 | 理由                                       |
| ----------- | ---- | ------------------------------------------------------------ | -------------------------------- | --------- | ---- | ------------------------------------------ |
| Origin      | [1]  | Sharpness-Aware Minimization for Efficiently Improving Generalization | 锐度感知最小化提升泛化能力       | ICLR      | 2021 | SAM 原始论文，奠定研究基础                 |
| Origin      | [2]  | Averaging Weights Leads to Wider Optima and Better Generalization | 权重平均提高泛化能力             | ArXiv     | 2018 | 提出 SWA 方法，为平坦极小值研究奠基        |
| Improvement | [3]  | Towards Efficient and Scalable Sharpness-Aware Minimization  | 高效可扩展 SAM                   | CVPR      | 2022 | 提出 LookSAM，降低 SAM 计算成本            |
| Improvement | [4]  | Make Sharpness-Aware Minimization Stronger: A Sparsified Perturbation Approach | 稀疏扰动增强 SAM                 | NeuIPS    | 2022 | 提出 SSAM，通过稀疏扰动优化计算效率        |
| Improvement | [5]  | Enhancing Sharpness-Aware Minimization by Learning Perturbation Radius | 学习扰动半径增强 SAM             | ECML PKDD | 2024 | 提出 LETS 方法，自适应优化 SAM             |
| Improvement | [6]  | SWAD: Domain Generalization by Seeking Flat Minima           | 寻找平坦极小值以优化域泛化       | NeurIPS   | 2021 | 提出 SWAD，结合 SWA 进行域泛化优化         |
| Analysis    | [7]  | When Do Flat Minima Optimizers Work?                         | 平坦极小值优化器的适用性分析     | NeurIPS   | 2022 | 系统对比 SWA 与 SAM，探讨适用场景          |
| Analysis    | [8]  | Towards Understanding Sharpness-Aware Minimization           | SAM 的数学分析                   | ICML      | 2022 | 从优化角度分析 SAM 泛化能力                |
|             |      | Why Does Sharpness-Aware Minimization Generalize Better Than SGD? |                                  | NIPS      | 2023 |                                            |
| Analysis    | [9]  | On the Duality Between Sharpness-Aware Minimization and Adversarial Training | SAM 与对抗训练的对偶性           | ArXiv     | 2023 | 分析 SAM 提高对抗鲁棒性的机制              |
|             |      |                                                              |                                  |           |      |                                            |
| Analysis    | [10] | Low-Pass Filtering SGD for Recovering Flat Optima            | 低通滤波 SGD 搜索平坦极小值      | AISTATS   | 2022 | 提出 LPF-SGD 方法，引导 SGD 寻找平坦区域   |
|             |      |                                                              |                                  |           |      |                                            |
| Add-on LoRA | [11] | Flat-LoRA: Low-Rank Adaptation over a Flat Loss Landscape    | 平坦极小值优化 LoRA              | ArXiv     | 2024 | 在 LoRA 适配中优化平坦极小值以提升泛化能力 |
| Add-on LoRA | [12] | Flat Minima Generalize for Low-Rank Matrix Recovery          | 平坦极小值提升低秩矩阵恢复泛化   | IMA       | 2024 | 理论分析平坦极小值在低秩优化中的作用       |
| Add-on LoRA | [13] | Implicit Regularization of Sharpness-Aware Minimization for Scale-Invariant Problems | SAM 在尺度不变问题中的隐式正则化 | NeurIPS   | 2024 | 探索 SAM 在低秩优化中的隐式正则化效果      |

本综述总结了深度学习优化中的平坦极小值研究进展，涵盖了 SAM、SWA 及其改进方法，并讨论了其理论分析、计算效率优化及在 LoRA 适配中的应用。希望本综述能为相关研究者提供有价值的参考。

------

# **Flat Minima Paper Research History**

## **1. Origin Papers**

### **Sharpness-Aware Minimization (SAM)**



SAM 是近年来提出的一种优化方法，核心思想是**最小化模型参数邻域内的最坏损失**，以此引导模型收敛至平坦极小值[1]。该方法显著提升了深度神经网络的泛化能力，并在多个数据集上展现出优越性能。

**关键贡献：**

- 通过对参数扰动进行最大化，平滑损失景观，从而增强模型泛化能力[1]。
- 提供了一种新的优化范式，在计算成本可接受的前提下，提升模型的鲁棒性和抗噪能力[1]。

### **Stochastic Weight Averaging (SWA)**

SWA 采用**对优化轨迹上多个权重点进行平均**的方法，从而找到更平坦的极小值，提高泛化能力[2]。相比于 SAM，SWA 计算量更低，适用于更广泛的深度学习任务。

**关键贡献：**

- 通过简单的权重平均实现对泛化能力的增强[2]。
- 提出了一种与 SAM 互补的优化方法，适用于训练稳定性要求较高的任务[2]。

------

## **2. Improvement**

### **优化 SAM 计算效率**

尽管 SAM 提高了泛化能力，但其计算代价较高，因此提出了一些改进方法：

- **LookSAM**：仅周期性计算梯度上升步骤，大幅降低计算成本，同时保持泛化性能[3]。
- **Sparse SAM (SSAM)**：采用稀疏扰动策略，只对部分参数施加扰动，减少计算需求[4]。
- **LETS**：提出基于双层优化的学习扰动半径方法，以适应不同任务需求[5]。



论文信息：





### **扩展 SWA 适用范围**

SWA 主要用于计算机视觉任务，研究者提出了其在其他领域的改进：

- **SWAD**：结合 SWA 进行域泛化优化，使模型在未知领域的泛化能力更强[6]。

------

## **3. Analysis**

### **理论分析 SAM 和 SWA 的泛化能力**

- **对比分析**：研究发现 SWA 和 SAM 各有优势，SWA 计算量小，但 SAM 在抗噪能力上更胜一筹[7]。
- **数学分析**：通过 Hessian 迹和变分法，深入研究 SAM 及其对优化动态的影响[8]。
- **对抗训练关系**：揭示 SAM 与对抗训练（AT）之间的对偶性，即 SAM 也能提升模型的对抗鲁棒性[9]。
- **低通滤波 SGD（LPF-SGD）**：提出一种基于低通滤波的 SGD 变种，主动搜索平坦极小值，提高优化效率[10]。

------

## **4. Add-on LoRA**

LoRA 作为一种低秩适配（Low-Rank Adaptation）方法，在适配预训练大模型时得到了广泛应用。最近的研究探索了 SAM 与 LoRA 的结合：

- **Flat-LoRA**：提出在 LoRA 低秩子空间内优化平坦极小值，以提升微调后的模型泛化能力[11]。
- **Flat Minima Generalize for Low-Rank Matrix Recovery**：研究发现，在低秩矩阵恢复任务中，**平坦极小值可以更好地重建真实数据结构**，并在多个矩阵优化任务（如矩阵补全、鲁棒 PCA）上验证了这一结论[12]。
- **Implicit Regularization of Sharpness-Aware Minimization for Scale-Invariant Problems**：研究表明，**SAM 在低秩优化问题中具有隐式正则化效应**，尤其在尺度不变问题上能够自适应地找到更优的优化路径[13]。

**关键贡献：**

- **平坦极小值在 LoRA 低秩优化中的重要性**：研究表明，SAM 和其他平坦极小值优化方法能够有效改善 LoRA 的泛化能力[11,12]。
- **理论支持**：通过 Hessian 分析和优化收敛性研究，进一步验证了 SAM 在低秩优化中的作用[12,13]。

------















# Flat minima Concept 



总结阅读几篇论文与 Flat Minima 相关 ， 首先需要介绍 Flat Minima的基本概念 

## **Flat Minima: 概念、研究范围与应用**

### **1. Flat Minima 的概念**
Flat Minima（平坦极小值）是机器学习优化中的一个核心概念，主要用于描述损失函数在参数空间中的极小值的几何特性。

### **(1) 定义**
假设神经网络的损失函数为 $L(\theta)$，其中 $\theta$ 是模型的参数向量。优化目标是找到参数 $\theta^*$ 使得 $L(\theta)$ 取得极小值：
$$
\theta^* = \arg\min_{\theta} L(\theta)
$$
Flat Minima 指的是**损失函数在局部极小点附近的变化较缓慢，即 Hessian 矩阵的特征值较小**：
- 如果 **Hessian 矩阵 $H = \nabla^2 L(\theta)$ 的最大特征值 $\lambda_{\max}$ 小**，则表示该点附近的损失函数曲率较小，极小值较“平坦”。
- 相反，如果 $\lambda_{\max}$ 较大，表示该极小值的曲面陡峭，被称为 **Sharp Minima（尖锐极小值）**。

### **(2) 直观理解**
- **Flat Minima**（平坦极小值）：模型在该区域的损失曲面较平坦，意味着参数的微小变化不会导致损失函数显著增大。这通常与 **更好的泛化能力（generalization）** 相关。
- **Sharp Minima**（尖锐极小值）：模型在该区域的损失曲面陡峭，意味着对参数变化敏感，可能导致 **过拟合（overfitting）**。

---

### **3. Flat Minima 在神经网络优化中的研究方向**
在神经网络优化领域，Flat Minima 是近年来深度学习泛化研究的重要方向，主要集中在以下几个方面：

### **(1) Flat Minima 与泛化能力**
研究表明，**平坦的极小值通常对应于较好的泛化能力**，即：
- **Sharp Minima** 可能导致模型在训练集上表现良好，但在测试集上表现较差（过拟合）。
- **Flat Minima** 由于损失曲面更平缓，即使输入数据略有变化，模型输出也不会发生剧烈变化，从而提高泛化能力。

### **(2) 通过优化方法寻找 Flat Minima**
不同的优化方法在损失曲面上的搜索方式不同，可能会影响最终收敛到的极小值：
- **标准 SGD**：
  - 可能会收敛到 Sharp Minima，特别是在学习率较小的情况下。
- **SWA（Stochastic Weight Averaging）**：
  - 通过权重平均，使优化解偏向平坦区域。
  - 适用于计算机视觉、图神经网络等任务。
- **SAM（Sharpness-Aware Minimization）**：
  - 通过显式最大化局部损失，找到更平坦的极小值。
  - 适用于 NLP 任务，提高 Transformer 结构的泛化能力。

### **(3) 计算 Flat Minima 的度量方法**
为了定量分析 Flat Minima，研究人员提出了多种度量方法：
- **Hessian 最大特征值**：
  - 计算 Hessian 矩阵的最大特征值 $\lambda_{\max}$，特征值越小，表示损失曲面越平坦。
- **Loss Landscape Visualization（损失曲面可视化）**：
  - 通过绘制损失曲面，观察不同优化器找到的极小值形态。
- **线性插值实验**：
  - 在非平坦解和平坦解之间进行线性插值，分析损失变化趋势。

---

### **4. Flat Minima 在神经网络中的应用**
### **(1) 计算机视觉**
- 使用 **SWA** 在 CNN 训练过程中找到更平坦的极小值，提高分类任务的鲁棒性。
- 例如，在 CIFAR-10 和 ImageNet 上，SWA 能够提高模型的测试集表现。

### **(2) 自然语言处理**
- **SAM 优化 Transformer** 结构，使其在 NLP 任务（如 GLUE Benchmark）中表现更好。
- SAM 通过寻找平坦的极小值，提升 RoBERTa 和 T5 的泛化能力。

### **(3) 图神经网络**
- **SWA 在 GNN 任务（如 OGB-Proteins）中提升泛化能力**，使节点预测和图分类任务更稳定。

---

### **5. 结论**
- **Flat Minima 是优化过程中损失曲面的重要几何特性，与模型的泛化能力密切相关。**
- **不同优化方法（如 SWA, SAM）能够引导模型找到更平坦的极小值，提高泛化能力。**
- **Flat Minima 在计算机视觉、NLP、GNN 以及统计优化等多个领域具有重要应用。**





### **Flat Minima: 概念、数学依据及在神经网络中的研究与应用**

## **1. Flat Minima 概念**

**Flat minima**（扁平极小值）是机器学习和优化领域的一个重要概念，主要用于描述损失函数局部最小值的几何特性。Flat minima 指的是那些在参数空间中较宽广、较平缓的最小值，而不是陡峭的极小值（Sharp minima）。

在神经网络优化中，找到一个**flat minimum**通常意味着模型对训练数据的小扰动更具鲁棒性，也能更好地泛化到未见数据上。因此，优化过程中希望找到的是**flat minima**，而不是**sharp minima**。

### **2. 数学研究依据**

从数学角度来看，flat minima 可以用**Hessian 矩阵**（损失函数的二阶导数矩阵）来衡量：

- **Sharp Minima（陡峭极小值）**: Hessian 矩阵的特征值较大，意味着损失函数在该点附近变化剧烈，泛化能力较弱。
- **Flat Minima（平缓极小值）**: Hessian 矩阵的特征值较小，意味着损失函数在该点附近变化较缓，对噪声的鲁棒性更强。

具体数学描述： 假设我们有一个损失函数 L(θ)L(\theta)，其在某点 θ∗\theta^* 处的二阶泰勒展开为：
$$
L(\theta) \approx L(\theta^*) + \frac{1}{2} (\theta - \theta^*)^T H (\theta - \theta^*)
$$
其中，$H = \nabla^2 L(\theta^*)$ 是 Hessian 矩阵。如果 H 的特征值较小，则意味着损失函数在该点附近变化较缓，说明是 flat minimum。

研究中常用的 flatness 量化指标：

1. **Trace(H)**（Hessian 迹）：衡量所有特征值的总和。
2. **λmax⁡(H)\lambda_{\max}(H)**（Hessian 最大特征值）：衡量损失曲面的最大陡峭程度。
3. **det(H)\text{det}(H)**（行列式）：衡量局部曲率。

### **Flat Minima 与泛化性（Generalization）**

有研究表明，选择 flat minima 可以提高模型的泛化能力（Keskar et al., 2017）。其核心理论依据包括：

- **PAC-Bayesian Bounds**: 说明 flat minima 关联于较低的泛化误差上界。
- **随机梯度下降（SGD）的偏差**: SGD 由于噪声梯度的影响，往往倾向于找到更平坦的最小值，而非陡峭的最小值。

### **3. 在神经网络领域的研究方向**

Flat minima 的研究在深度学习领域引起了广泛关注，主要涉及以下几个方面：

### **(1) 训练方法**

- **Stochastic Gradient Descent (SGD)**: 由于 SGD 具有噪声性质，容易逃离 sharp minima，找到更扁平的极小值。
- **Entropy-SGD** (Chaudhari et al., 2017): 通过在损失函数上加入熵项，使优化倾向于寻找平坦的极小值。
- **Sharpness-Aware Minimization (SAM)** (Foret et al., 2021): 直接在优化过程中最小化“局部最大扰动损失”，鼓励找到 flat minima。

### **(2) 泛化能力提升**

- **Flat minima 与模型鲁棒性**: 研究表明，flat minima 可提升模型对噪声数据、对抗攻击（adversarial attacks）等的鲁棒性。
- **Wide Networks vs. Narrow Networks**: 宽度更大的神经网络（如 Transformer 或 ResNet）更倾向于找到 flat minima，而窄网络可能更容易陷入 sharp minima。

### **(3) 结构化优化**

- **Loss Surface Analysis（损失面分析）**: 通过研究神经网络的损失面几何结构，分析 flat minima 对优化动态的影响。
- **Bayesian Deep Learning**: 通过在损失函数上引入先验（如 Laplace 近似、Gaussian Processes），探索 flat minima 与贝叶斯不确定性估计的关系。

### **4. Flat Minima 的实际应用**

Flat minima 在多个深度学习任务中被广泛应用，如：

- **计算机视觉**: 目标检测、图像分类（ResNet、ViT 等模型的训练策略）
- **自然语言处理（NLP）**: 预训练语言模型（如 BERT、GPT）的优化
- **强化学习（Reinforcement Learning）**: 通过寻找 flat minima 来增强策略鲁棒性

### **5. 相关研究与参考文献**

- Keskar, N. S., Nocedal, J., Mudigere, D., Smelyanskiy, M., & Tang, P. T. P. (2017). **On large-batch training for deep learning: Generalization gap and sharp minima**. *International Conference on Learning Representations (ICLR)*. [[Paper](https://arxiv.org/abs/1609.04836)]
- Chaudhari, P., Choromanska, A., Soatto, S., LeCun, Y., Baldassi, C., Borgs, C., ... & Zecchina, R. (2017). **Entropy-SGD: Biasing gradient descent into wide valleys**. *International Conference on Learning Representations (ICLR)*. [[Paper](https://arxiv.org/abs/1611.01838)]
- Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021). **Sharpness-aware minimization for efficiently improving generalization**. *International Conference on Learning Representations (ICLR)*. [[Paper](https://arxiv.org/abs/2010.01412)]
- Hochreiter, S., & Schmidhuber, J. (1997). **Flat minima**. *Neural Computation*, 9(1), 1-42. [[Paper](https://direct.mit.edu/neco/article/9/1/1/6105/Flat-Minima)]

------

### **总结**

Flat minima 是神经网络优化的重要概念，代表了泛化能力更强、鲁棒性更好的解。数学上可以通过 Hessian 矩阵的特征值衡量，研究方法 Hessian 矩阵包括 SGD、Entropy-SGD 和 SAM 等。Flat minima 影响深度学习模型的训练、优化和泛化能力，在计算机视觉、NLP、强化学习等多个领域有重要应用。

🚀 **希望这篇介绍对你有所帮助！如果你对某个具体方向感兴趣，可以深入讨论！**





# 如何描述，度量，刻画 Flat minima



# 从不同的视角进行定义





# 不同的度量方式

# **Filter Normalization Method: Step-by-Step Process**

## **Overview**

The **Filter Normalization** method is a technique proposed to enhance the visualization of neural network loss landscapes by eliminating scale invariance issues. This method allows for fair comparisons of loss landscapes across different network architectures and training settings.

------

## **Step-by-Step Process of Filter Normalization**

### **Step 1: Generate a Random Direction Vector**

- Create a random Gaussian direction vector $d$ with the same dimensions as the neural network parameters  $\theta$:
  $$
  d \sim \mathcal{N}(0, I)
  $$
   $I$ is the identity matrix representing a Gaussian distribution with zero mean and unit variance.

------

### **Step 2: Normalize Each Filter Individually**

- For each layer in the neural network:

  - Identify the **filters** (sets of weights that operate over input channels).

  - Compute the Frobenius norm of each filter in the parameter tensor $\theta$:
    $$
    \|\theta_{i,j}\| = \sqrt{\sum_{k} \theta_{i,j,k}^2}
    $$

  - Scale the corresponding filter in the direction vector $d$:
    $$
    d_{i,j} \gets \frac{d_{i,j}}{\|d_{i,j}\|} \cdot \|\theta_{i,j}\|
    $$
    
- This ensures that the perturbation maintains the relative scale of each filter, removing artificial differences in visualization due to weight magnitude variations.

------

### **Step 3: Construct the Loss Landscape Projection**

- Define two independent **random direction vectors** $\xi$ and $\eta$ after applying filter-wise normalization.

- Perturb the original network parameters in these directions:
  $$
  \theta(\alpha, \beta) = \theta^* + \alpha \xi + \beta \eta
  $$
  where:

  - $\theta^*$ is the reference set of parameters (e.g., a trained model),
  - $\alpha, \beta$ are scalar coefficients controlling the magnitude of the perturbation.

- Compute the loss function values over a grid of $(\alpha, \beta)$ values:
  $$
  f(\alpha, \beta) = L(\theta(\alpha, \beta))
  $$
  

------

### **Step 4: Visualize the Loss Landscape**

- Generate **2D contour plots** based on $f(\alpha, \beta)$, ensuring that the normalization method is applied consistently across different architectures and training methods.

  

- This visualization allows for a **fair comparison of sharpness and flatness** between different minimizers.

------

## **Advantages of Filter Normalization**

| Feature                                | Traditional Visualization                  | Filter Normalization                   |
| -------------------------------------- | ------------------------------------------ | -------------------------------------- |
| **Effect of Weight Scaling**           | Distorted due to varying weight magnitudes | Eliminates scale dependence            |
| **Comparability Across Architectures** | Limited due to varying norms               | Standardized comparisons across models |
| **Accuracy of Sharpness Estimation**   | Can be misleading                          | Correlates well with generalization    |

------

## **Final Notes**

- **Batch normalization layers should not be perturbed** to avoid instability in visualization.
- **Applicable to both convolutional and fully connected layers**, treating FC layers as $1 \times 1$ convolutions.
- **Computational cost** is slightly higher than traditional methods but provides **more reliable insights** into neural network training dynamics.

🚀 **Future Improvements**:

- Explore **higher-dimensional projections** for more accurate landscape representation.
- Apply the method to **transformers and non-CNN architectures**.
- Investigate **real-time visualization techniques** during training.



# **Explanation of Figure 4: 3D Loss Landscape Visualization**

## **Overview**

Figure 4 in the paper presents **3D loss landscape visualizations** for different neural network architectures, specifically **ResNet-110 without skip connections** and **DenseNet-121** trained on CIFAR-10. These 3D visualizations provide insights into the non-convexity and sharpness of the loss surface, helping to explain why certain architectures are more trainable than others.

------

## **Steps to Generate the 3D Loss Landscape**

The 3D loss landscape is created by **projecting the high-dimensional loss function** onto a 2D plane using two independent random direction vectors. The following steps outline the process:

### **Step 1: Select a Reference Point**

- Choose a **trained model parameter set** $\theta^*$ (e.g., the final converged weights after training).
- This serves as the **center point** of the visualization.

### **Step 2: Generate Random Direction Vectors**

- Generate two independent random Gaussian direction vectors 

  $\xi$

   and 

  $\eta$

   with the same dimension as 

  $\theta^*$

  :

  ```
  $$\xi \sim \mathcal{N}(0, I), \quad \eta \sim \mathcal{N}(0, I)$$
  ```

- Apply 

  filter-wise normalization

   to ensure comparability:

  ```
  $$\xi_{i,j} \gets \frac{\xi_{i,j}}{\|\xi_{i,j}\|} \cdot \|\theta_{i,j}\|
  $$
  ```

  ```
  $$\eta_{i,j} \gets \frac{\eta_{i,j}}{\|\eta_{i,j}\|} \cdot \|\theta_{i,j}\|
  $$
  ```

- This ensures that different architectures have a comparable loss landscape visualization.

### **Step 3: Perturb the Model Parameters**

- Modify the parameters along the two directions to obtain new weight configurations:

  ```
  $$\theta(\alpha, \beta) = \theta^* + \alpha \xi + \beta \eta$$
  ```

- Here, $\alpha$ and $\beta$ are scalar perturbations that define the movement in the loss landscape.

### **Step 4: Compute the Loss Values**

- Evaluate the loss function 

  $L(\theta(\alpha, \beta))$

   on the validation dataset for a range of values:

  ```
  $$f(\alpha, \beta) = L(\theta^* + \alpha \xi + \beta \eta)$$
  ```

- This produces a **grid of loss values** that represents the loss surface.

### **Step 5: Plot the 3D Loss Surface**

- Construct a 3D mesh plotwith:
  - **X-axis:** $\alpha$ (perturbation along direction $\xi$)
  - **Y-axis:** $\beta$ (perturbation along direction $\eta$)
  - **Z-axis:** $f(\alpha, \beta)$ (loss function value)
- The surface is rendered using a colormap to indicate different loss values.

------

## **Key Observations from Figure 4**

1. **ResNet-110 without skip connections**:
   - The loss landscape is **highly chaotic** with steep valleys and large regions of high curvature.
   - The sharp changes in the loss surface indicate **poor trainability**.
   - This explains why deep networks without skip connections often struggle to converge.
2. **DenseNet-121**:
   - The loss landscape is **much smoother and more convex**.
   - The loss variations are more gradual, suggesting **better trainability and generalization**.
   - This supports the idea that well-designed architectures (such as DenseNets) provide more stable optimization surfaces.

------

## **Why is This Visualization Important?**

- **Provides empirical evidence** on how architecture choices affect optimization difficulty.
- **Reveals the transition from convex to chaotic behavior** in deep networks.
- **Supports the argument that skip connections improve trainability** by smoothing the loss landscape.

🚀 **Future Work**:

- Apply this technique to **transformer architectures**.
- Investigate **dynamic loss landscape evolution** during training.
- Use **higher-dimensional embeddings** to visualize more complex optimization trajectories.



# SAM 论文阅读



# Origin paper

# Paper1  SAM:SHARPNESS-AWARE MINIMIZATION FOR EFFICIENTLY  IMPROVING GENERALIZATION





[SAM解析：Sharpness-Aware Minimization for Efficiently Improving Generalization-CSDN博客](https://blog.csdn.net/qq_40744423/article/details/121570423)





## **Paper Analysis and Summary**

### 1. Scope/Setting
This paper—**“Sharpness-Aware Minimization for Efficiently Improving Generalization” (ICLR 2021)**—focuses on improving generalization performance of large, over-parameterized neural networks. It does so by exploring how **loss landscape geometry**, particularly “sharp” vs. “flat” minima, impacts the final model’s ability to generalize beyond the training set. The authors propose a new training procedure called **Sharpness-Aware Minimization (SAM)**.

### 2. Purpose
- **Motivation**: Merely minimizing training loss in large neural networks does not necessarily yield good test (generalization) performance. Sharp minima often correlate with poorer generalization.
- **Objective**: The authors aim to design an efficient and scalable algorithm that simultaneously minimizes loss value and “sharpness,” guiding the model toward broader (flatter) minima associated with better generalization.

### 3. Key Idea
1. **Min-Max Formulation**: Instead of simply minimizing the training loss \( L_S(w) \), SAM solves:
   $$
   \min_{w} \Bigl[\, \max_{\|\epsilon\|_p \le \rho} L_S(w + \epsilon) \Bigr],
   $$
   where \( \rho \) is a hyperparameter controlling the size of the local neighborhood around \( w \), and \( p \) typically equals 2 in the paper’s experiments.
2. **Flatness Criterion**: If even the “worst-case” small perturbation \(\epsilon\) fails to significantly increase the loss, it implies the loss landscape is relatively flat in that region, which tends to improve generalization.

### 4. Method
1. **Approximate Inner Maximization**  
   - The **inner maximization** $ \max_{\|\epsilon\|\le\rho} L_S(w + \epsilon)  $is approximated with a first-order Taylor expansion.  
   - This yields a perturbation $ \epsilon^*(w) $that aligns with the local gradient direction, scaled to norm \(\rho\).
2. **Gradient Computation**  
   
   - After obtaining $ \epsilon^*(w)$ , they compute the gradient of the loss at $\( w + \epsilon^*(w) \)$.  
   - **Second-order terms** are dropped for computational efficiency.
   - The final update is performed via:
     $$
     w \leftarrow w - \eta \, \nabla_w \Bigl[ \max_{\|\epsilon\|\le\rho} L_S(w + \epsilon) \Bigr]
     \approx w - \eta \, \nabla_w \, L_S\bigl(w + \epsilon^*(w)\bigr).
     $$
   
3. **Implementation**  
   - Practically requires **two forward-backward passes** each iteration (one for finding $\( \epsilon^*\)$ and one for the final gradient).  
   - Integrated easily into standard frameworks (PyTorch, TensorFlow, JAX, etc.).

### 5. Contributions
1. **Algorithmic Novelty**: Introduces a practical way to **explicitly penalize sharp minima** via a tractable min-max optimization—**SAM**.
2. **Theoretical Justification**: Shows a PAC-Bayesian-style bound linking **training loss in a small neighborhood** of the parameters to **generalization**.
3. **Empirical Validation**: Demonstrates consistent improvements in test accuracy across **CIFAR-10**, **CIFAR-100**, **ImageNet**, **SVHN**, **Fashion-MNIST**, and numerous finetuning tasks.  
4. **Label-Noise Robustness**: SAM inherently provides robustness to noisy labels on par with specialized noisy-label algorithms.

### 6. Difference and Innovation
- **Sharpness-Aware** vs. Standard Minimization: Unlike vanilla SGD/Adam, SAM **explicitly** incorporates loss curvature (sharpness) into optimization.  
- **Contrast with Prior Work**:  
  - **Entropy-SGD, Chaudhari et al.** also attempt to smooth the landscape, but require more complex sampling-based computations. SAM is simpler to implement and scales better.  
  - **Weight Averaging** methods (e.g., SWA, Izmailov et al.) also lead to flatter solutions, but do so by averaging weights over multiple iterations. SAM directly **adapts** each update step to avoid sharp minima.

### 7. Results and Conclusion
- **CIFAR-10/100**: State-of-the-art (or near SOTA) error rates using WideResNet, PyramidNet, ShakeShake, etc. Notably, error on CIFAR-100 can drop from ~14–16% to ~10–12%.
- **ImageNet**: Significant improvement (e.g., ResNet-152 top-1 error from 20.3% down to 18.4%).
- **Finetuning**: SAM yields better performance than standard finetuning (e.g., EfficientNet-L2 improvements on multiple datasets).
- **Robustness to Label Noise**: Outperforms or matches specialized noisy-label methods.
- **Key Conclusion**: SAM is a straightforward, effective, and theoretically motivated way to consistently improve generalization by prioritizing **flat** minima.

#### 8. Discussion
- **m-Sharpness**: An interesting twist is that splitting the dataset (in minibatches or across accelerators) yields a notion of “m-sharpness,” which can predict generalization even better than the global measure.  
- **Hessian Spectrum**: SAM-trained models exhibit reduced Hessian eigenvalues, confirming the approach’s link to flatter minima.  
- **Future Directions**:  
  - More thorough analysis of second-order terms.  
  - Extensions to tasks beyond image classification (e.g., NLP, RL).  
  - Combining or comparing with other data-augmentation/regularization approaches.

---

## **Table: Base Models and Datasets**

Below is a concise summary of **base models** (approximate size) and **datasets** from the paper’s main experiments. Parameter counts are approximate or taken from common references.

| **Model**                        | **Params (approx.)** | **Datasets**                            |
| -------------------------------- | -------------------- | --------------------------------------- |
| **WideResNet-28-10**             | ~36.5M               | CIFAR-10, CIFAR-100, SVHN, F-MNIST      |
| **Shake-Shake (26 2x96d)**       | ~26M                 | CIFAR-10, CIFAR-100, SVHN, F-MNIST      |
| **PyramidNet+ShakeDrop**         | ~26–30M+             | CIFAR-10, CIFAR-100                     |
| **ResNet-50**                    | ~25.6M               | ImageNet                                |
| **ResNet-101**                   | ~44.5M               | ImageNet                                |
| **ResNet-152**                   | ~60M                 | ImageNet                                |
| **EfficientNet-B7**              | ~66M                 | Finetuning on CIFAR-10, CIFAR-100, etc. |
| **EfficientNet-L2**              | ~480M                | Large-scale pretraining + finetuning    |
| **Additional** (e.g., ResNet-32) | ~0.47M (small)       | Noisy-label experiments (CIFAR-10)      |

- **Note**: Param sizes vary slightly depending on implementations, residual connections, and shape definitions.  
- **Datasets**:  
  - *CIFAR-10/CIFAR-100*: Standard 32×32 images, 50k train, 10k test.  
  - *ImageNet*: 1.28M training images, 50k validation.  
  - *SVHN*: Digit classification, 73k+531k images.  
  - *Fashion-MNIST*: 28×28 grayscale clothing images.  
  - *Finetuning tasks*: Flowers, Stanford Cars, Birdsnap, etc.  
  - *Noisy-label setups*: CIFAR-10 with artificially corrupted labels.

---

**References**  
- Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021). *Sharpness-Aware Minimization for Efficiently Improving Generalization*. ICLR 2021.



## **Detailed Analysis of the Method (SAM)**

### 1. Motivation and Purpose of the Method

Modern deep neural networks are heavily **over-parameterized**, often capable of memorizing training sets without necessarily achieving optimal generalization. Traditional training **minimizes only the training loss** $L_S(w)$, which can converge to minima that are “sharp”—regions in the loss landscape where slight parameter perturbations significantly increase the loss, potentially leading to poorer test performance.

- **Objective**: **Sharpness-Aware Minimization (SAM)** simultaneously **reduces loss** and **penalizes sharp minima**, thereby guiding the optimizer toward “flatter” minima with better generalization performance.

---

### 2. Overview of the Method

### 2.1 Key Idea: Min-Max Formulation

The paper formulates SAM as a **min-max** optimization problem:

$$
\min_{w} \; \max_{\|\epsilon\|\_p \le \rho} \; L\_S\bigl(w + \epsilon\bigr).
$$

- **Outer Minimization**: Optimize $w$ to ensure the **worst-case local perturbation** $\epsilon$ in the parameter space does **not** raise the loss too much.  
- **Inner Maximization**: Identify the **most “damaging” direction** in a small neighborhood ($\|\epsilon\|\_p \le \rho$) around $w$.

By explicitly **punishing** solutions whose **local region** is sharp, SAM encourages **flat** minima.

### 2.2 Algorithmic Steps

1. **Approximate the Inner Problem**  
   - Use a **first-order Taylor expansion** around $w$ to find:
     $$
     \epsilon^\*(w) \approx \rho \,\frac{\nabla L\_S(w)}{\|\nabla L\_S(w)\|\_q}, 
     \quad \text{where } \tfrac{1}{p} + \tfrac{1}{q} = 1.
     $$
   - In practice, $p=2$ (i.e., Euclidean norm) is commonly used, so $\|\epsilon\| \le \rho$ means $\epsilon^\*$ is simply the gradient direction scaled to norm $\rho$.

2. **Compute the SAM Gradient**  
   - Evaluate the gradient at the **perturbed** point $(w + \epsilon^\*(w))$:
     $$
     \nabla\_w L\_{S}^{\text{SAM}}(w) \approx \nabla\_w \, L\_S\bigl(w + \epsilon^\*(w)\bigr).
     $$
   - **Second-order terms** (involving Hessian) are **omitted** for efficiency.

3. **Parameter Update**  
   - Perform **gradient descent** with the above gradient:
     $$
     w \leftarrow w \;-\; \eta \,\nabla\_w L\_{S}^{\text{SAM}}(w).
     $$

### 2.3 Key Implementation Details

- **Double Forward-Backward Pass**: Each step needs (1) one pass to compute $\epsilon^\*$, (2) another pass to compute the “perturbed” gradient.  
- **Practical Hyperparameters**: 
  - **$\rho$** controls how large the local perturbation can be. Typical values range from $0.01$ to $0.2$ in the experiments.  
  - **$p$** is usually 2 (Euclidean).  

### 2.4 Tips / Observations

- **Batch-Level or Accelerator-Level**: In practice, SAM is often computed per mini-batch (or even per “sub-batch” across multiple GPUs/TPUs).  
- **Stability**: Reducing $\rho$ when faced with instability or extremely large parameter gradients can help convergence.  
- **No Extra Regularization**: Although the paper uses $L^2$ weight decay, SAM already acts as a form of regularization that favors smoother regions.

---

### 3. Dataset Usage, Benchmarks, and Evaluation Metrics

- **Datasets**:  
  1. **CIFAR-10 / CIFAR-100**: Standard benchmarks (32×32 color images).  
  2. **ImageNet**: Large-scale (1.28M training images, 50k validation).  
  3. **SVHN**: Street View House Numbers (digit classification).  
  4. **Fashion-MNIST**: Clothing images (28×28 grayscale).  
  5. **Finetuning Tasks**: CIFAR-10/100 (for transfer), Flowers, Stanford Cars, etc.  
  6. **Noisy-Label Scenarios**: CIFAR-10 with synthetic label corruption.

- **Benchmarks / Metrics**:
  - **Classification Accuracy** (or error rate) on test/validation sets is the main metric.  
  - For ImageNet: **Top-1** and **Top-5** error rates.  
  - Some experiments measure **robustness** to label noise, comparing final test accuracy on clean data.

---

### 4. Experimental Setup

### 4.1 Domains and Base Models

1. **Standard Image Classification**  
   - **WideResNet** (WRN-28-10, WRN-40-10), **ResNet** (ResNet-50, -101, -152), **Shake-Shake**, **PyramidNet** families.  
   - **Data Augmentations**: Basic flips/crops vs. advanced methods (AutoAugment, Cutout).  

2. **Large-Scale Classification**  
   - **ImageNet** with ResNet-50/101/152 (comparing SAM vs. standard SGD across up to 400 epochs).

3. **Finetuning**  
   - **EfficientNet** variants (B7, L2) pretrained on ImageNet or additional unlabeled data.  
   - Finetuned on smaller classification tasks (Flowers, Stanford Cars, etc.).

4. **Noisy Labels**  
   - ResNet-32 on CIFAR-10 with artificially corrupted labels (20–80% noise rates).

### 4.2 Optimizers and Main Settings

- **Primary Optimizer**: Stochastic Gradient Descent (**SGD**) + momentum (0.9).  
- **Learning Rate Schedules**: Commonly **cosine** or step-based decays.  
- **Batch Size**: Typically 256 for CIFAR, sometimes 4096 for ImageNet.  
- **Weight Decay**: e.g., $10^{-4}$, tuned per model.  
- **Number of Epochs**:  
  - CIFAR: 200–1800 epochs.  
  - ImageNet: up to 400 epochs.  
  - If SAM uses 200 epochs, a baseline might run 400 epochs to ensure fair comparison of total computational cost.

### 4.3 Main Experimental Points of Discussion

1. **Impact of $\rho$**: The paper shows that moderate values (0.05) often yield strong results.  
2. **Comparison vs. “Vanilla” Training**: They compare SAM’s final test accuracy vs. baseline (same architecture, same training steps, or 2× steps for baseline).  
3. **Comparison vs. Other Techniques**:
   - Weight averaging (SWA),  
   - Data augmentation (Mixup, AutoAugment),  
   - Noisy-label specialized methods (MentorMix, etc.).  

4. **Robustness Studies**: Label noise experiments highlight how SAM’s “worst-case” approach naturally mitigates noisy targets.  

---

### 5. Experimental Results

1. **CIFAR-10/100**  
   - SAM consistently **lowers error rates** (e.g., from 2.2% to 1.6% on CIFAR-10 with WRN-28-10).  
   - New **state-of-the-art** on CIFAR-100 with certain model-augmentation combos (down to ~10.3% error).

2. **ImageNet**  
   - Training for 200–400 epochs with SAM significantly improves top-1 accuracy.  
   - For instance, **ResNet-152** top-1 error from 20.3% down to **18.4%**.

3. **Finetuning**  
   - EfficientNet-B7/L2 with SAM achieves **better** or **state-of-the-art** performance on multiple smaller tasks (e.g., ~0.3% error on CIFAR-10, ~3.92% on CIFAR-100).

4. **Noisy Label Experiments**  
   - On corrupted CIFAR-10, SAM matches or **exceeds** specialized noisy-label approaches by simply handling local perturbations in parameter space.

### 5.1 Authors’ Analysis

- **Loss Landscape Insights**: Hessian spectra show significantly **lower maximum eigenvalues** under SAM, confirming flatter minima.  
- **m-Sharpness**: Splitting data across minibatches yields an even more fine-grained notion of sharpness with better correlation to generalization gaps.
- **Performance Gains**: The results underscore that **explicitly discouraging “sharp minima”** often yields noticeable improvements in test accuracy, especially in high-capacity networks.

### 5.2 Key Conclusions

- SAM is a **simple extension** of standard training pipelines (needing two gradient calculations per step).  

- Consistently outperforms standard optimizers in achieving **low test error** and better **robustness**.  

- Encourages further exploration of **flatness** measures (e.g., $m$-sharpness) as a tool to predict or improve generalization.

  

---



# Paper2 SWA： Averaging Weights Leads to Wider Optima and Better Generalization

| **作者:** Pavel Izmailov; Dmitrii Podoprikhin; Timur Garipov; Dmitry Vetrov; Andrew Gordon Wilson; |
| ------------------------------------------------------------ |
| **期刊: , 2019.**                                            |
| **期刊分区:**                                                |
| **本地链接:**                                                |
| **DOI:** [10.48550/arXiv.1803.05407](https://doi.org/10.48550/arXiv.1803.05407) |
| **摘要:** *Deep neural networks are typically trained by optimizing a loss function with an SGD variant, in conjunction with a decaying learning rate, until convergence. We show that simple averaging of multiple points along the trajectory of SGD, with a cyclical or constant learning rate, leads to better generalization than conventional training. We also show that this Stochastic Weight Averaging (SWA) procedure finds much flatter solutions than SGD, and approximates the recent Fast Geometric Ensembling (FGE) approach with a single model. Using SWA we achieve notable improvement in test accuracy over conventional SGD training on a range of state-of-the-art residual networks, PyramidNets, DenseNets, and Shake-Shake networks on CIFAR-10, CIFAR-100, and ImageNet. In short, SWA is extremely easy to implement, improves generalization, and has almost no computational overhead.* |
| **标签:** # SWA ,                                            |
| **笔记日期:** 2025/2/22 20:29:52                             |

## 📜 研究核心

------

> Tips: 做了什么，解决了什么问题，创新点与不足？



### ⚙️ 内容



### 💡 创新点



### 🧩 不足



## 🔁 研究内容

------



### 💧 数据



### 👩🏻‍💻 方法



### 🔬 实验



### 📜 结论



## 🤔 个人总结

------

> Tips: 你对哪些内容产生了疑问，你认为可以如何改进？



### 🙋‍♀️ 重点记录



### 📌 待解决



### 💭 思考启发





### **论文解析：SWA（Stochastic Weight Averaging）—— 通过权重平均找到更宽的极小值并提升泛化能力**

------

## **1. 研究背景与范围**

### **研究背景（Scope）**

- 现代深度神经网络（DNN）通常使用 **随机梯度下降（SGD）** 变体进行训练，配合 **学习率衰减** 来优化损失函数。
- 研究表明，**SGD 可能会收敛到较尖锐（sharp）的局部极小值，而不是更宽（flat）的最优解**，从而影响模型泛化能力。
- **SWA 方法的核心目标**：通过 **权重平均（Weight Averaging）** 来找到**更宽的极小值（Flat Minima）**，进而提升泛化能力。

### **研究范围（Setting）**

- SWA 方法适用于各种 **深度神经网络架构（ResNet, DenseNet, VGG, PyramidNet等）**。
- 主要在 **CIFAR-10、CIFAR-100、ImageNet** 等数据集上进行实验。
- **比较对象**：SGD、FGE（Fast Geometric Ensembling），探讨 SWA 相对于其他优化方法的优势。

------

## **2. 核心思想（Key Idea）**

- **SGD 的问题**：
  - 仅使用学习率衰减优化，可能导致权重最终收敛到**尖锐极小值（sharp minima）**，而这些解的泛化能力较差。
  - 在高维优化空间中，SGD 可能只探索到局部最优，而未能找到全局最优区域。
- **SWA 的核心思想**：
  - **在 SGD 训练过程中对多个权重进行平均**，以找到更宽的极小值，使模型更加稳健。
  - SWA 通过 **固定或周期性学习率（cyclical/constant learning rate）**，让模型在**高泛化能力区域来回探索**，然后对这些权重求平均。

------

## **3. 方法（Method）**

### **SWA 训练过程**

1. **初始训练（Standard SGD Training）**

   - 先使用常规 SGD 训练模型，直到模型收敛或达到一定的训练轮数。

2. **采样权重（Weight Sampling）**

   - 在训练过程中，使用 **固定学习率（constant learning rate）** 或 **周期性学习率（cyclical learning rate）** 采样多个权重 $\theta_i$。

3. **权重平均（Weight Averaging）**

   - 对采样的权重进行均值计算：
     $$
     \theta_{\text{SWA}} = \frac{1}{N} \sum_{i=1}^{N} \theta_i
     $$
     
   - 该均值权重更可能落在**较平坦的损失区域（flat minima）**，从而提升泛化能力。
   
4. **最终模型（Final Model）**

   - 采用 $\theta_{\text{SWA}}$ 作为最终模型，并使用 batch normalization 进行最终调整。

### **计算复杂度**

- SWA **几乎没有额外计算成本**：
  - 仅需在训练过程中维护一个**权重均值**，相比于 SGD **计算开销极小**。

------

## **4. 贡献（Contribution）**

1. **提出 SWA 方法**，通过权重平均找到更宽的极小值，提高泛化能力。
2. **揭示了 SGD 训练的轨迹特性**，解释了为何 SGD 可能收敛到不理想的解。
3. **SWA 可作为简单的“插件”方法**，无需更改模型结构，可直接应用到各种 DNN 训练流程中。
4. **提供了对 FGE（Fast Geometric Ensembling）方法的近似**，但比 FGE **计算效率更高**。

------

## **5. SWA 相较于其他方法的不同与创新（Difference & Innovation）**

| 方法    | 训练方式                    | 目标             | 计算开销 | 泛化能力                 |
| ------- | --------------------------- | ---------------- | -------- | ------------------------ |
| **SGD** | 传统梯度下降                | 最小化损失       | 低       | 可能收敛到尖锐极小值     |
| **FGE** | 采样多个模型并做预测平均    | 提升泛化能力     | 高       | 需要多个模型，计算成本大 |
| **SWA** | 仅对 SGD 轨迹中的权重做平均 | 找到更宽的极小值 | 低       | 提升泛化能力，计算成本低 |

- **创新点**：
  - **SWA 不需要多个模型**（如 FGE），而是直接**对单个模型的不同阶段的权重进行平均**，达到类似 FGE 的泛化能力提升效果。
  - **SWA 在计算开销上几乎与 SGD 相同**，但能够大幅提升泛化能力。

------

## **6. 主要实验结果（Results & Conclusion）**

### **实验数据集**

- **CIFAR-10**、**CIFAR-100**、**ImageNet**
- 深度神经网络架构：**VGG-16、ResNet-164、Wide ResNet-28-10、DenseNet-161、Shake-Shake Net**

### **实验结果**

| 方法                | CIFAR-100  | CIFAR-10   | ImageNet   |
| ------------------- | ---------- | ---------- | ---------- |
| **SGD**             | 78.49%     | 95.28%     | 76.15%     |
| **FGE（6-12模型）** | 79.84%     | 95.45%     | -          |
| **SWA（1个模型）**  | **80.35%** | **96.79%** | **76.97%** |

- **关键发现**：
  - **SWA 比传统 SGD 提高 1-2% 泛化性能**，并且对**不同架构和数据集均有效**。
  - **SWA 比 FGE 计算成本低，但在泛化性能上相近或更优**。
  - **SWA 收敛到的极小值更宽，模型对数据扰动的鲁棒性更强**。

### **最终结论**

- **SWA 能够找到更平坦的损失极小值，提高泛化能力，而计算开销几乎与 SGD 相同**。
- **SWA 是对 SGD 训练的简单补充，可以无缝集成到现有深度学习训练流程中**。
- **与 FGE 相比，SWA 计算成本更低，但具有相似的泛化优势**。

------

## **7. 讨论（Discussion & Prove）**

- **SWA 是否适用于所有优化问题？**
  - SWA **适用于大多数 DNN 训练，但在某些非凸优化问题上仍需验证**。
  - SWA 的**权重平均策略能否推广到 Transformer 等架构**，仍是开放性问题。
- **如何进一步改进 SWA？**
  - **DSWA（Dense SWA）** 提出了**密集采样**，在 $t_s$ 到 $t_e$ 之间进行**权重平均**，进一步避免过拟合，提高泛化能力。
  - **可以结合其他正则化方法**（如 Mixup、Dropout）进一步提升泛化能力。

------

## **8. 总结**

- **SWA 通过简单的权重平均机制，显著提高了深度学习模型的泛化能力。**
- **相比 SGD，SWA 收敛到更平坦的极小值，提高测试集表现，且计算成本极低。**
- **相比 FGE，SWA 仅需一个模型即可达到相似的泛化提升，适用于大规模训练任务。**

------

**📌 代码开源**：[GitHub - SWA](https://github.com/timgaripov/swa) 🚀

以上内容完整解析了 SWA 方法，并以 Markdown 代码格式输出，确保可复制与二次编辑。







## **方法解析：SWA（Stochastic Weight Averaging）—— 通过权重平均找到更宽的极小值并提升泛化能力**

本文提出的 **SWA（Stochastic Weight Averaging）** 方法，旨在通过 **权重平均（Weight Averaging）** 使模型收敛到更平坦的极小值（Flat Minima），从而提升模型的泛化能力。以下内容详细分析 SWA 的 **目的、与现有方法的区别、创新点、具体训练步骤、关键细节设置、注意事项、方法不足及未来改进方向**。

------

## **1. 方法提出的目的**

### **背景**

- 现代深度神经网络（DNN）的优化通常采用 **随机梯度下降（SGD）及其变体**。
- **SGD 可能会收敛到较尖锐的局部极小值（Sharp Minima）**，导致模型对测试集的泛化能力较差。
- **研究发现，模型收敛到较宽的极小值（Flat Minima）时，泛化能力更强**。

### **SWA 方法的目标**

- **通过权重平均**，找到**更平坦的极小值**，使模型具有更好的泛化性能。
- **避免 SGD 可能收敛到尖锐极小值的问题**，减少模型对数据分布变化的敏感性。
- **无需额外计算开销**，在几乎不增加训练成本的情况下提升模型性能。

------

## **2. SWA 与之前方法的区别与创新点**

### **2.1 SWA vs. 传统 SGD**

| 方法    | 目标                         | 收敛点               | 泛化能力         | 计算开销 |
| ------- | ---------------------------- | -------------------- | ---------------- | -------- |
| **SGD** | 直接最小化损失               | 可能收敛到尖锐极小值 | 泛化能力可能较弱 | 低       |
| **SWA** | 通过权重平均找到更宽的极小值 | 选择平坦极小值       | 泛化能力更强     | 低       |

**主要区别：**

- **SWA 不改变 SGD 训练过程**，而是**在后期进行权重平均**，从而找到更平坦的极小值。
- **SWA 权重更新不会改变模型结构**，可以直接与现有训练流程结合。

### **2.2 SWA vs. Fast Geometric Ensembling (FGE)**

| 方法    | 计算策略                     | 计算开销 | 泛化能力 |
| ------- | ---------------------------- | -------- | -------- |
| **FGE** | 训练多个模型，并计算权重平均 | 高       | 高       |
| **SWA** | 训练单个模型，并计算权重平均 | 低       | 高       |

**创新点：**

- SWA **仅对单个模型进行权重平均**，避免了 FGE 需要训练多个模型的高计算成本。
- **SWA 通过“周期性学习率”策略，让模型探索更广泛的权重空间**，比 FGE 更高效。

------

## **3. SWA 方法步骤**

### **步骤 1：初始化**

- 采用 **标准 SGD 训练模型**，并设置：
  - **周期性学习率（Cyclical Learning Rate, CLR）** 或 **固定学习率（Constant LR）**。
  - 权重存储列表 $\Theta_{\text{SWA}}$。

### **步骤 2：收集权重**

- 在训练过程中，每隔 **$K$轮**，将当前模型权重 $\theta_t$ 存入 SWA 权重列表：

  $$\Theta_{\text{SWA}} = \{\theta_{t_1}, \theta_{t_2}, ..., \theta_{t_n}\}$$

### **步骤 3：计算最终权重**

- 计算所有采样权重的均值：

  

  $$\theta_{\text{SWA}} = \frac{1}{N} \sum_{i=1}^{N} \theta_i$$

  

### **步骤 4：应用 Batch Normalization 重新计算均值和方差**

- 由于 SWA 采用的是多个不同时间点的权重，其 BatchNorm 统计可能不匹配，需要重新计算。

------

## **4. 细节设置**

### **(1) 采样策略**

- **固定学习率（Constant LR）**：确保采样的权重均匀分布在最优区域。
- **周期性学习率（Cyclical LR）**：
  - 让模型在不同权重区域探索，使采样权重覆盖更广泛的极小值区域。

### **(2) 权重存储间隔**

- $K$ 设定为 5-10 轮，使得权重的波动性足够大，能够覆盖更平坦的区域。

### **(3) 适用的深度网络**

- **CNN**（ResNet, VGG, DenseNet）
- **PyramidNet**
- **Shake-Shake Networks**

------

## **5. 关键实验结果**

### **实验数据集**

| 数据集    | 训练样本数 | 测试样本数 | 任务类型       |
| --------- | ---------- | ---------- | -------------- |
| CIFAR-10  | 50,000     | 10,000     | 图像分类       |
| CIFAR-100 | 50,000     | 10,000     | 图像分类       |
| ImageNet  | 1,281,167  | 50,000     | 大规模图像分类 |

### **实验结果**

| 方法                | CIFAR-100  | CIFAR-10   | ImageNet   |
| ------------------- | ---------- | ---------- | ---------- |
| **SGD**             | 78.49%     | 95.28%     | 76.15%     |
| **FGE（6-12模型）** | 79.84%     | 95.45%     | -          |
| **SWA（1个模型）**  | **80.35%** | **96.79%** | **76.97%** |

- **SWA 在所有数据集上都超越了标准 SGD**，并且**计算开销远低于 FGE**。
- **SWA 训练的模型在测试集上泛化能力更强，表现更稳定**。

------

## **6. 关键注意点（Tips）**

1. **权重存储间隔 $K$ 过大或过小都会影响最终效果**
   - $K$ 过小：权重之间的变化幅度太小，可能没有涵盖足够大的优化区域。
   - $K$ 过大：可能会错过关键的平坦极小值区域。
2. **周期性学习率比固定学习率表现更优**
   - **周期性学习率** 可以让模型探索更广的参数空间，从而使 SWA 能够找到更平坦的极小值。
3. **重新计算 BatchNorm 统计信息**
   - 由于 SWA 使用的是多个不同时间点的权重，BatchNorm 统计值可能不匹配，需要重新计算均值和方差。

------

## **7. 不足之处与未来改进方向**

### **(1) 不足之处**

1. **SWA 只适用于分类任务，未在回归任务上进行测试**。
2. **当前方法只使用均值，而未考虑权重的方差信息**，可能无法充分利用不同极小值之间的差异。
3. **不适用于训练中不使用 BatchNorm 的模型**，因为 SWA 依赖 BatchNorm 重新计算统计信息。

### **(2) 未来改进方向**

1. **结合方差信息进行更精细的权重聚合**：
   - 例如，引入 **动态权重加权策略**，而不是简单的均值计算。
2. **扩展到 Transformer 结构**：
   - 目前 SWA 主要应用于 CNN 结构，未来可以探索其在 Transformer（如 ViT）上的表现。
3. **与正则化方法结合**：
   - 结合 **Mixup, Dropout** 等方法，进一步提升泛化能力。

------

## **8. 结论**

- **SWA 通过简单的权重平均策略，大幅提升了深度神经网络的泛化能力**。
- **相比于 SGD，SWA 能够找到更平坦的极小值，且计算成本极低**。
- **相比于 FGE，SWA 计算成本更低，但能达到相似的泛化能力提升效果**。

------

**📌 代码开源**：[GitHub - SWA](https://github.com/timgaripov/swa) 🚀



# Improvement 

# Paper 1:ASAM: Adaptive Sharpness-Aware Minimization  for Scale-Invariant Learning of Deep Neural Networks



------

## **1. 论文分析**

### **(1) 研究背景与范围 (Scope or Setting)**

- **背景**：
  - 近年来，基于损失曲面**锐度（sharpness）**的学习算法成为衡量模型泛化能力的重要工具，如 **Sharpness-Aware Minimization (SAM)**。
  - SAM 方法在各种图像分类任务上实现了 SOTA 结果，其主要思想是通过最小化损失曲面的锐度来提升模型的泛化能力。
  - 但**固定半径的锐度度量**存在尺度依赖性（scale dependency）问题，即参数缩放可能影响锐度计算，从而影响其与泛化能力之间的相关性。
- **研究目标**：
  - 提出**自适应锐度（Adaptive Sharpness）**的概念，使得锐度的计算具有尺度不变性（scale-invariant）。
  - 设计基于 Adaptive Sharpness 的优化算法 **Adaptive Sharpness-Aware Minimization (ASAM)**，提高模型泛化性能，避免 SAM 受尺度依赖性问题影响。

### **(2) 关键思想 (Key Idea)**

- **引入自适应锐度（Adaptive Sharpness），并证明其具有尺度不变性**：
  - 传统锐度定义在一个固定半径的 $L_p$ 球体上，而 ASAM 通过**归一化操作**（normalization operator）调整优化区域，使得锐度计算不受参数缩放影响。
  - 证明 Adaptive Sharpness 仍然可以作为泛化误差的上界，从理论上支持其作为泛化能力的衡量指标。
- **设计新的优化方法 ASAM，使用 Adaptive Sharpness 进行优化**：
  - ASAM 通过**自适应调整最大化区域**（adaptive maximization region）来增强优化的鲁棒性，避免 SAM 可能的训练不稳定问题。
  - ASAM 保持了与 SAM 类似的两步更新策略，但**适配了自适应归一化的扰动计算方法**。

### **(3) 方法 (Method)**

1. **定义 Adaptive Sharpness**：

   $$\max_{\|\mathbf{T}^{-1}_w \mathbf{\epsilon} \|_p \leq \rho} L(w + \mathbf{\epsilon}) - L(w)$$

   - 其中，$\mathbf{T}_w^{-1}$ 是归一化操作，使得锐度计算不受参数缩放影响。

2. **提出 ASAM 的优化目标**：

   $$\min_w \max_{\|\mathbf{T}^{-1}_w \mathbf{\epsilon} \|_p \leq \rho} L(w + \mathbf{\epsilon}) + \frac{\lambda}{2} \|w\|^2_2$$

   - 该优化目标类似于 SAM，但采用自适应锐度进行最小化。

3. **两步优化流程**：

   1. 计算最大化方向上的扰动$\epsilon_t$：

      $$\epsilon_t = \rho \mathbf{T}_w \frac{\nabla L(w_t)}{\|\mathbf{T}_w \nabla L(w_t) \|_q}$$

   2. 更新模型参数：

      $$w_{t+1} = w_t - \alpha_t (\nabla L(w_t + \epsilon_t) + \lambda w_t)$$

### **(4) 贡献 (Contribution)**

1. **引入 Adaptive Sharpness，克服了 SAM 受尺度变化影响的局限性**：
   - 证明 Adaptive Sharpness 具有**尺度不变性**（scale-invariant property），相比传统锐度定义，其与泛化误差的相关性更强。
2. **提出 ASAM 算法，基于 Adaptive Sharpness 进行优化**：
   - **自适应调整扰动区域**，避免因参数缩放导致的优化不稳定性，提高优化的鲁棒性。
3. **在多个基准测试上验证 ASAM 的有效性**：
   - 在 **CIFAR-10、CIFAR-100、ImageNet** 和 **IWSLT’14（机器翻译任务）** 上进行实验，结果表明 ASAM **优于 SGD 和 SAM**，具有更好的泛化能力。

### **(5) 创新点与区别 (Difference and Innovation)**

- **与 SAM 的区别**：
  - **SAM 使用固定半径的球形区域**进行锐度计算，而 ASAM **使用自适应归一化区域**，避免参数缩放影响锐度计算。
  - ASAM 提出的归一化算子 **$\mathbf{T}_w$** 使得锐度计算不依赖于模型参数的缩放，优化更稳定。
- **与其他锐度优化方法的区别**：
  - 与 Hessian-based 方法（如 FGE, SWA）相比：
    - ASAM 计算成本较低，不需要 Hessian 矩阵的计算，适用于更大规模的模型训练。
  - 与 m-sharpness 方法相比：
    - ASAM 通过**归一化方法**而非单纯扩大扰动半径，提供更稳定的优化策略。

### **(6) 结果与结论 (Result and Conclusion)**

1. **在图像分类任务上的结果（CIFAR-10, CIFAR-100, ImageNet）**
   - ASAM **在所有测试模型上都超过了 SAM 和 SGD**，特别是在 PyramidNet-272、ResNeXt29 等深度模型上提升显著。
   - CIFAR-10 最高测试准确率：
     - **SGD：96.34%**，**SAM：96.98%**，**ASAM：97.28%**
   - CIFAR-100 最高测试准确率：
     - **SGD：81.56%**，**SAM：83.42%**，**ASAM：83.68%**
2. **在机器翻译任务 IWSLT’14 DE-EN 上的 BLEU 评分**
   - ASAM 也超越了 SAM，在 Transformer 结构上提升翻译质量：
     - **Adam：34.86**
     - **Adam+SAM：34.78**
     - **Adam+ASAM：35.02**
3. **鲁棒性测试**
   - 在 **标签噪声任务**（CIFAR-10 添加 20%~80% 噪声）中，ASAM 在大多数情况下优于 SAM 和 SGD，表明其对噪声的鲁棒性较强。

### **(7) 不足之处与未来方向**

- **计算复杂度仍然比 SGD 高**：

  - ASAM 仍然需要额外的梯度计算，与 SAM 一样 

    计算量约为 SGD 的 2 倍

    ，未来可考虑降低计算开销的方法，如：

    - 采用**LookSAM 方式减少计算频率**
    - 结合**低秩近似**减少计算量

- **归一化方法的选择仍然值得优化**：

  - 目前仅研究了 **element-wise 和 filter-wise 归一化**，未来可探讨 **layer-wise 或更复杂的归一化方法**，进一步优化泛化能力。

- **适应性超参数选择仍需优化**：

  - 目前需要人为调节 $\rho$，未来可以采用**自适应学习$\rho$的方法（如 LETS-SAM）**，减少超参数调优的难度。

------

## **总结**

ASAM 提出了一种**自适应锐度最小化方法**，通过 **Adaptive Sharpness 解决了 SAM 受参数缩放影响的问题**，并在多个任务中取得优于 SAM 的效果。尽管计算量仍然高于 SGD，但其**泛化能力和优化稳定性更优**，未来可进一步研究计算优化和归一化方法来提升效率。



## **方法解析：《ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks》**

------

## **1. 方法提出的目的**

### **背景**

- **SAM（Sharpness-Aware Minimization）** 是一种通过优化损失曲面锐度（sharpness）来提升深度神经网络泛化能力的优化方法。
- SAM 通过在**固定半径的扰动区域**内最大化损失，然后最小化最大扰动下的损失，促使模型找到**平坦的最小值（flat minima）**，从而提高泛化能力。
- **问题：SAM 依赖于固定半径的扰动区域，而这种固定尺度的锐度度量在不同的神经网络结构中可能具有不同的尺度，导致优化过程不稳定。**

### **目标**

- 设计一种 **Adaptive Sharpness-Aware Minimization (ASAM)** 方法，使得扰动区域能够自适应调整，而不受参数尺度的影响。
- 解决 SAM 存在的**尺度依赖性问题（scale dependency）**，使得优化过程更加稳定，并进一步提升泛化能力。

------

## **2. 主要区别与创新**

### **ASAM 与 SAM 的区别**

| 方法     | 扰动计算方式                          | 是否受参数尺度影响 | 是否有更稳定的优化过程 |
| -------- | ------------------------------------- | ------------------ | ---------------------- |
| **SAM**  | 采用固定半径的 $L_p$ 球形区域计算扰动 | **受参数尺度影响** | **优化可能不稳定**     |
| **ASAM** | 采用自适应归一化的扰动区域计算扰动    | **无尺度依赖性**   | **优化更加稳定**       |

### **创新点**

1. **提出 Adaptive Sharpness 度量方法**
   - **SAM 计算锐度时，使用固定尺度的扰动**，而 ASAM 通过自适应归一化，使得扰动计算与参数尺度无关。
   - 证明 **Adaptive Sharpness 仍然可以作为泛化误差的上界**，从理论上支持其作为优化目标。
2. **引入 Adaptive Maximization Region**
   - SAM 在计算扰动时，会受到参数尺度的影响，而 ASAM **通过归一化操作** 来调整最大化区域，使得优化更加鲁棒。
3. **提升优化稳定性**
   - ASAM 通过**自适应调整扰动大小**，避免了 SAM 可能出现的不稳定训练行为，提高了模型的泛化能力。

------

## **3. 具体方法步骤**

### **3.1 Adaptive Sharpness 公式**

ASAM 采用**自适应归一化的扰动区域**，其优化目标为：

$$w_{t+1} = w_t - \alpha_t (\nabla L(w_t + \epsilon_t) + \lambda w_t)$$

其中：

- $\mathbf{T}_w^{-1}$ 是归一化操作，使得锐度计算不受参数缩放影响。
- 这一优化目标类似于 SAM，但采用**自适应锐度计算**。

### **3.2 计算最大化方向上的扰动**

为了计算最大扰动 $\epsilon^*$，ASAM 采用以下公式：

$$\epsilon_t = \rho \mathbf{T}_w \frac{\nabla L(w_t)}{\|\mathbf{T}_w \nabla L(w_t) \|_q}$$

其中：

- $\mathbf{T}_w$ 是归一化算子，避免参数缩放影响锐度计算。

### **3.3 计算参数更新**

- 计算带有扰动的梯度：

$$w_{t+1} = w_t - \alpha_t (\nabla L(w_t + \epsilon_t) + \lambda w_t)$$

### **3.4 归一化算子 $\mathbf{T}_w$**

- ASAM 提出了不同的归一化策略：
  1. **Element-wise 归一化**：对每个参数进行独立归一化
  2. **Filter-wise 归一化**：针对 CNN 结构，每个滤波器单独归一化
  3. **Layer-wise 归一化**：针对 Transformer 结构，对每层参数归一化

不同的归一化方法适用于不同的任务和网络架构。

------

## **4. 关键细节设置**

### **4.1 评估数据集**

- 计算机视觉任务：
  - CIFAR-10 / CIFAR-100
  - ImageNet
- 自然语言处理任务：
  - GLUE
  - IWSLT’14 DE-EN（翻译任务）

### **4.2 评估指标**

- **Top-1/Top-5 Accuracy**（分类任务）
- **BLEU Score**（翻译任务）
- **泛化误差（Generalization Gap）**
- **对抗鲁棒性（Adversarial Robustness）**

### **4.3 计算设置**

- **优化器**：SGD / AdamW
- **学习率**：$5 \times 10^{-4}$
- **扰动半径初始值**：$\rho_0 = 0.05$
- **$\beta$（扰动半径学习率）**：0.1

------

## **5. 关键实验结果**

### **5.1 ASAM 对比 SAM / SGD**

| 方法     | CIFAR-10 Top-1 Acc (%) | CIFAR-100 Top-1 Acc (%) | ImageNet Top-1 Acc (%) |
| -------- | ---------------------- | ----------------------- | ---------------------- |
| SGD      | 96.34                  | 81.56                   | 76.9                   |
| SAM      | 96.98                  | 83.42                   | 77.5                   |
| **ASAM** | **97.28**              | **83.68**               | **78.9**               |

- ASAM **在所有测试模型上都超过了 SAM 和 SGD**，特别是在 PyramidNet-272、ResNeXt29 等深度模型上提升显著。

### **5.2 在机器翻译任务 IWSLT’14 DE-EN 上的 BLEU 评分**

| 方法          | BLEU      |
| ------------- | --------- |
| Adam          | 34.86     |
| Adam+SAM      | 34.78     |
| **Adam+ASAM** | **35.02** |

- ASAM 也超越了 SAM，在 Transformer 结构上提升翻译质量。

### **5.3 对抗鲁棒性测试**

- 在 **标签噪声任务**（CIFAR-10 添加 20%~80% 噪声）中，ASAM 在大多数情况下优于 SAM 和 SGD，表明其对噪声的鲁棒性较强。

------

## **6. 关键注意点（Tips）**

1. **归一化方式的选择**
   - **Element-wise 归一化** 适用于小型网络，如 ResNet-18。
   - **Filter-wise 归一化** 适用于 CNN，如 ResNeXt。
   - **Layer-wise 归一化** 适用于 Transformer，如 BERT、GPT。
2. **扰动半径 $\rho$ 需要调整**
   - 在 NLP 任务中，较大的 $\rho$ 可能会影响模型稳定性。
3. **计算开销**
   - ASAM 计算量与 SAM 类似，比 SGD 高出 **2 倍**。

------

## **7. 不足之处与未来方向**

### **7.1 不足之处**

1. **计算复杂度仍然较高**
   - ASAM 仍然需要额外的梯度计算，与 SAM 一样 **计算量约为 SGD 的 2 倍**，未来可考虑降低计算开销的方法。
2. **归一化方式的选择仍然值得优化**
   - 目前仅研究了 **element-wise 和 filter-wise 归一化**，未来可探讨 **layer-wise 或更复杂的归一化方法**，进一步优化泛化能力。
3. **自适应参数调整**
   - 目前需要人为调节 $\rho$，未来可以采用**自适应学习$\rho$的方法（如 LETS-SAM）**，减少超参数调优的难度。

------

## **8. 结论**

ASAM 提出了一种**自适应锐度最小化方法**，通过 **Adaptive Sharpness 解决了 SAM 受参数缩放影响的问题**，并在多个任务中取得优于 SAM 的效果。尽管计算量仍然高于 SGD，但其**泛化能力和优化稳定性更优**，未来可进一步研究计算优化和归一化方法来提升效率。



**ASAM（Adaptive Sharpness-Aware Minimization）与 L2 正则化的关系**主要体现在它们都在优化过程中**影响参数的更新方向和最终收敛的解**，但两者的作用机制和优化目标不同。以下是详细的对比分析：

------

## **1. ASAM 与 L2 正则化的共同点**

1. **都能提升泛化能力**
   - ASAM 通过最小化损失曲面的锐度（sharpness）找到更平坦的最小值，以此提高模型的泛化能力。
   - L2 正则化（通常以权重衰减 weight decay 的形式使用）通过对参数施加 $L_2$ 罚项，限制参数的增长，减少过拟合，提高泛化能力。
2. **都影响优化路径**
   - ASAM 通过改变梯度更新方向，使参数向平坦的极小值收敛。
   - L2 正则化通过在梯度更新中添加一个 $L_2$ 罚项，使参数收敛到更小的范数。
3. **都可以防止模型在训练集上的过拟合**
   - ASAM 通过避免 sharp minima，使模型更鲁棒。
   - L2 正则化限制权重大小，防止神经网络对训练数据的过度拟合。

------

## **2. 主要区别**

| **方法**      | **优化目标**                                                 | **主要作用**                       | **更新方式**                                                 |
| ------------- | ------------------------------------------------------------ | ---------------------------------- | ------------------------------------------------------------ |
| **L2 正则化** | 限制权重的范数，使参数收敛到较小的值                         | 限制模型复杂度，减少过拟合         | 额外添加 $L_2$ 罚项，权重更新规则：$w \leftarrow w - \eta (\nabla L + \lambda w)$ |
| **SAM**       | 通过最大化局部扰动后的损失，再最小化损失，使模型在平坦极小值处收敛 | 使损失曲面更平坦，提高泛化能力     | 先计算最坏情况下的梯度扰动，再更新参数                       |
| **ASAM**      | SAM 的改进版，使用自适应扰动，使锐度计算不受参数尺度影响     | 进一步增强优化稳定性，提高泛化能力 | 采用归一化的自适应锐度计算方法                               |

------

## **3. ASAM 与 L2 正则化的数学关系**

L2 正则化的优化目标：

$$\min_w L(w) + \frac{\lambda}{2} \|w\|^2_2$$

其中 $\lambda$ 是正则化系数。

ASAM 的优化目标：

$$\min_w \max_{\|\mathbf{T}^{-1}_w \mathbf{\epsilon} \|_p \leq \rho} L(w + \mathbf{\epsilon}) + \frac{\lambda}{2} \|w\|^2_2$$

其中：

- $\mathbf{T}_w^{-1}$ 是自适应归一化操作，
- $\rho$ 是扰动半径。

可以看出，ASAM 其实**在原始 SAM 的目标函数上额外添加了 L2 正则化项**，这意味着 ASAM 既**优化了模型的锐度（通过扰动梯度），又通过 L2 正则化约束了模型的参数范数**。

------

## **4. 为什么 ASAM 与 L2 正则化可以协同作用？**

1. **L2 正则化保证参数范数不会过大，减少 sharp minima 影响**
   - SAM/ASAM 主要调整优化路径，但不限制权重大小。
   - 如果不加 L2 正则化，ASAM 可能会收敛到**较大权重值但仍然是平坦的最小值**，但这种解可能会带来计算不稳定性。
   - L2 正则化可以避免参数过大，从而**在 SAM/ASAM 选择的平坦极小值中优先选择参数范数较小的解**。
2. **ASAM 使损失曲面更平坦，而 L2 正则化可以提高训练稳定性**
   - 纯 L2 正则化可能无法保证找到平坦的极小值，而 ASAM 的优化方式可以有效避免锐度大的解。
   - ASAM 和 L2 结合可以共同优化模型，使得优化过程更加稳定，并且提升泛化能力。
3. **两者在权重更新上的互补性**
   - **L2 正则化** 在梯度下降中增加 $-\lambda w$ 项，使权重在每次更新时减少一小部分。
   - **ASAM** 则在梯度更新前先在局部扰动空间中寻找最坏情况，然后更新权重，确保最终的解更加稳定。

------

## **5. 实际应用中的启示**

- **ASAM + L2 正则化是一种常见的组合**，因为它可以同时优化**损失曲面锐度（ASAM）\**和\**参数范数（L2 正则化）**，提升模型的稳定性和泛化能力。
- **如果不加 L2 正则化，ASAM 可能会导致模型权重过大，从而增加训练的不稳定性**。
- **在大规模神经网络（如 Transformer 或 ResNet）中，L2 正则化通常与 ASAM 一起使用，以获得更好的训练效果。**

------

## **6. 未来优化方向**

虽然 ASAM 和 L2 正则化可以一起提升泛化能力，但仍然有优化的空间：

1. **自适应 L2 正则化**
   - 目前 ASAM 使用固定的 L2 权重衰减参数 $\lambda$，但可以尝试动态调整 $\lambda$ 以适应不同的训练阶段。
   - 例如，初始训练时 $\lambda$ 较大，以抑制权重的过度增长，随后逐渐降低。
2. **结合 LETS（Learnable Perturbation Radius）优化 ASAM**
   - ASAM 仍然使用固定的扰动半径 $\rho$，可以结合 LETS 进行自适应学习，使扰动大小也能动态调整。
3. **结合 SWA（Stochastic Weight Averaging）**
   - 既然 ASAM 已经选择了平坦极小值，可以在 ASAM 之后使用 SWA 进一步平滑模型参数，提高泛化能力。

------

## **7. 结论**

- **ASAM 与 L2 正则化具有互补性**，ASAM 通过寻找平坦极小值优化损失曲面，而 L2 正则化控制权重范数，防止参数过大导致的不稳定性。
- **ASAM 本质上是 SAM 的改进版，并且其优化目标已经包含了 L2 正则化项**，表明两者可以自然结合以提升模型的泛化能力。
- **未来可以探索 L2 正则化的动态调整策略，以及结合其他优化方法（如 LETS, SWA）进一步提升 ASAM 的效果。**

**最终结论：ASAM 与 L2 正则化可以协同作用，提高神经网络的训练稳定性和泛化能力。**



# Paper SWAD 2: SWAD: Domain Generalization by Seeking Flat Minima



## **论文综述：SWAD - 通过寻找平坦极小值实现领域泛化**

### **1. 研究背景与问题设定（Scope & Setting）**

在深度学习模型的训练过程中，一个关键挑战是 **域泛化（Domain Generalization, DG）**，即如何在训练域（source domains）上的学习能够有效泛化到未知的测试域（target domains）。由于现实世界的数据分布往往存在 **域偏移（domain shift）**，即训练数据和测试数据的分布可能存在较大差异，使得模型在测试域上的表现可能急剧下降。

**现有 DG 方法的问题：**

- **经验风险最小化（ERM）问题**：研究表明，在 **DomainBed** 评测协议下，传统的 ERM 方法在复杂的非凸损失函数上训练，容易收敛到**尖锐极小值（sharp minima）**，从而导致泛化能力受限。
- **如何寻求更好的泛化解**：近年来的研究表明，**平坦极小值（flat minima）** 更有利于泛化，即损失函数的局部区域变化较小的最优解能够更稳健地适应分布偏移。

------

### **2. 研究目标（Key Idea）**

论文提出了一种新方法 **Stochastic Weight Averaging Densely (SWAD)**，旨在 **通过寻找更平坦的极小值来提高域泛化能力**。具体来说：

- **数学理论支持**：论文构建了一个 **稳健风险最小化（Robust Risk Minimization, RRM）** 框架，理论上证明了平坦极小值能够缩小域泛化误差。
- **算法优化**：论文提出了一种 **密集且防过拟合的随机权重采样策略**，改进了传统的 SWA 方法，以更精确地找到平坦极小值。
- **实验验证**：在多个 DG 基准数据集上进行实验，证明 SWAD 在 OOD（Out-of-Domain）泛化任务上优于现有 SOTA 方法。

------

### **3. 主要方法（Method）**

论文的方法基于 **Stochastic Weight Averaging (SWA)**，并进行优化，以适应域泛化问题。

#### **3.1 传统 SWA 的局限**

**SWA（随机权重平均）** 是一种已被证明可以找到平坦极小值的方法，主要思路是：

- 采用 **周期性或高恒定学习率** 训练模型；
- **间隔性地采样** 训练过程中的模型权重，并进行均值化，以获得一个更具鲁棒性的模型。

**问题**：

- **权重采样密度不足**：SWA 通常每 **K 轮采样一次**（通常小于 10 次），在高维参数空间中，可能难以准确逼近平坦极小值。
- **容易过拟合**：由于 DG 任务中的数据集较小，ERM 方法可能会快速收敛到局部最优解，从而导致模型过拟合。

------

#### **3.2 SWAD 方案**

**（1）密集采样策略（Dense Sampling）**

- **SWA**：每 K 轮采样一次，导致采样点稀疏，难以准确捕捉平坦极小值。
- **SWAD**：**每次迭代都采样权重**，增加采样点密度，使得最终平均后的模型更趋于平坦极小值。

**（2）防过拟合策略（Overfit-Aware Sampling）**

- 观察到小数据集上，模型的训练损失可能会在少量训练轮次后趋于最优，然后过拟合。
- SWAD 通过 **监测验证集损失（validation loss）**，动态调整采样区间：
  - 设定 **起始点 $t_s$**：当验证集损失首次达到局部最优时。
  - 设定 **终止点 $t_e$**：当验证集损失连续上升一定次数时。

**（3）最终的 SWAD 算法**

```
1. 初始化训练参数 θ

2. 进行标准训练，并监测验证集损失

3. 确定权重采样区间 [ts, te]

4. 在此区间内，密集采样模型权重

5. 计算平均权重，得到最终模型
```

------

### **4. 贡献与创新点（Contribution, Difference & Innovation）**

- **理论贡献**：
  - 证明了**平坦极小值的域泛化误差上界较低**。
  - 通过 **稳健风险最小化（RRM）框架**，提供了数学上的解释。
- **算法贡献**：
  - **提出 SWAD 进行域泛化**，优化了 SWA 方法：
    - **密集采样** 以更精确地找到平坦极小值。
    - **防过拟合策略** 使得模型更稳健地适应不同域的分布偏移。
  - **兼容性强**：SWAD 可以无缝集成到其他域泛化方法中。
- **实验贡献**：
  - **在五个 DG 基准数据集上超过 SOTA 方法**：
    - **PACS**: +2.6%
    - **VLCS**: +0.3%
    - **OfficeHome**: +1.9%
    - **TerraIncognita**: +1.4%
    - **DomainNet**: +2.9%
  - **可与其他方法结合**，进一步提升性能，如 SWAD + CORAL 获得最佳结果。

------

### **5. 结果分析（Result & Conclusion）**

#### **5.1  **与其他方法对比**

| 方法         | PACS     | VLCS     | OfficeHome | TerraInc | DomainNet | 平均提升        |
| ------------ | -------- | -------- | ---------- | -------- | --------- | --------------- |
| ERM          | 85.5     | 77.5     | 66.5       | 46.1     | 40.9      | 63.3            |
| 最优 SOTA    | 86.6     | 78.8     | 68.7       | 48.6     | 43.6      | 65.3            |
| **SWAD**     | **88.1** | **79.1** | **70.6**   | **50.0** | **46.5**  | **66.9 (+3.6)** |
| SWAD + CORAL | **88.3** | **78.9** | **71.3**   | **51.0** | **46.8**  | **67.3**        |

#### **5.2 关键结论**

#### **5.2 关键结论**

- **证明了平坦极小值在 DG 任务中的重要性**。
- **SWAD 显著优于 ERM 及现有 SOTA 方法**。
- **可与其他 DG 方法结合，进一步提升性能**。
- **可用于更广泛的任务，如 ImageNet 迁移学习，提高模型鲁棒性**。

------

### **6. 讨论（Discussion）**

**SWAD 的优势**

- **无须修改模型架构**，可直接应用于现有 DG 方法。
- **减少对模型选择的敏感性**，提高泛化稳定性。
- **不仅提升 OOD 泛化，还提升了 ID 泛化（in-domain generalization）**。

**局限性**

- **理论分析的置信界限较宽**，仍需进一步改进。
- **没有利用领域标签进行显式优化**，未来可以结合域适应（domain adaptation）技术。

------

### **7. 结论（Conclusion）**

- **引入平坦极小值的概念到域泛化任务**，证明其有效性。
- **提出 SWAD 方法，优化 SWA 以更精确地找到平坦极小值**。
- **在多个基准数据集上超越 SOTA 方法，并可与其他方法结合进一步提升性能**。

**代码开源**：[GitHub - SWAD](https://github.com/khanrc/swad) 🚀

本研究为 **域泛化问题提供了一个新的方向，即通过寻找平坦极小值来提升模型的泛化能力**，有望在更多应用场景中进一步推广。



## **方法解析：SWAD - 通过寻找平坦极小值实现领域泛化**

本文提出的 **SWAD（Stochastic Weight Averaging Densely）** 是一种改进版的 **随机权重平均（SWA）**，用于解决 **领域泛化（Domain Generalization, DG）** 问题。以下内容详细分析该方法的 **目的、创新点、步骤、细节设置及关键注意点（Tips）**。

------

## **1. 方法提出的目的**

现有 **领域泛化（DG）** 方法主要基于：

1. **经验风险最小化（ERM）**：直接在源域训练，模型容易收敛到**尖锐极小值（sharp minima）**，导致泛化能力受限。
2. **不变表示学习（Invariant Representation Learning）**：如 **CORAL、MMD** 等方法，关注源域间的分布对齐，但未充分考虑模型优化过程中的损失曲面特性。

**平坦极小值（flat minima）与泛化的关系：**

- 研究表明，**平坦的损失区域比尖锐极小值更具泛化能力**，能够更稳健地适应测试域。
- SWA（Stochastic Weight Averaging）是一个基于**权重平均**的优化方法，可以找到更平坦的最优解。

**问题：**

- 传统 SWA 方法的**采样密度低**，导致寻找的最优解可能不够平坦。
- 传统 SWA **未考虑过拟合问题**，在小数据集训练时可能过早收敛到局部最优。

**SWAD 目标：**

- **改进 SWA，提高采样密度**，找到更平坦的极小值，提高泛化能力。
- **引入防过拟合机制**，在训练过程中动态调整权重平均策略。

------

## **2. SWAD 与传统方法的区别与创新**点**

| 方法     | 目标                                             | 主要问题                       | 解决方案                    |
| -------- | ------------------------------------------------ | ------------------------------ | --------------------------- |
| ERM      | 直接最小化训练损失                               | 可能收敛到尖锐极小值，泛化性差 | 无特定泛化机制              |
| SWA      | 通过权重平均找到平坦极小值                       | 采样密度低，过拟合风险         | 仅进行权重平均              |
| **SWAD** | 通过更密集的采样找到更平坦的极小值，提高泛化能力 | 采样不足 & 过拟合风险          | **密集采样 + 防过拟合策略** |

**创新点**

1. **引入密集采样（Dense Sampling）**：
   - SWA 仅每 K 轮采样一次，SWAD **在整个训练过程中均匀采样权重**，提高搜索精度。
2. **防过拟合策略（Overfit-Aware Sampling）**：
   - 通过 **验证集损失监测**，在**合适的区间**进行权重平均，避免过早进入过拟合状态。

------

## **3. 方法步骤**

### **步骤 1：训练初始化**

- 设定 **训练参数 $\theta$**，学习率 $\eta$，训练轮数 $T$。
- 设定**采样间隔 $K$**，确保在整个训练过程中进行均匀采样。

### **步骤 2：标准训练**

- 采用 SGD 或 Adam 进行正常训练，并在每轮计算 **验证集损失**：

  $L_{val}(\theta_t) = \frac{1}{N} \sum_{i=1}^{N} L(f(x_i, \theta_t), y_i)$

### **步骤 3：密集采样**

- 传统 SWA **每 K 轮采样一次**：

  $\theta_{SWA} = \frac{1}{M} \sum_{i=1}^{M} \theta_i$

- **SWAD 采样更密集**：

  - 设定权重平均的**起始点 $t_s$**（当验证集损失首次下降到最优值）。

  - 设定**终止点 $t_e$**（当验证集损失连续上升超过一定阈值）。

  - 在**$[t_s, t_e]$ 之间进行密集权重平均**：

    $$
    \theta_{SWAD} = \frac{1}{|t_e - t_s|} \sum_{t=t_s}^{t_e} \theta_t
    $$


### **步骤 4：最终模型**

- 使用计算得到的 $\theta_{SWAD}$ 作为最终模型。

------

## **4. 细节设置**

### **(1) 采样密度**

- **SWA 采样间隔大（$K = 10$）**，在高维优化空间中，可能找到的极小值仍然较尖锐。
- **SWAD 采样间隔小（$K = 1$）**，可以更准确地找到平坦区域。

### **(2) 过拟合控制**

- 训练过程中，监测 **验证集损失**：
  - 如果损失连续下降，继续采样。
  - 如果损失连续上升，停止采样，防止模型进入过拟合状态。

### **(3) 适配其他方法**

- **可与其他 DG 方法结合**（如 **CORAL, MMD, Mixup**）。
- SWAD **在原始训练策略上添加权重平均，不改变原始模型架构**。

------

## **5. 关键实验结果**

### **实验数据集**

| 数据集     | 训练样本数 | 测试样本数 | 任务类型     |
| ---------- | ---------- | ---------- | ------------ |
| PACS       | 9991       | 2048       | 图像分类     |
| VLCS       | 10729      | 2819       | 图像分类     |
| OfficeHome | 15500      | 3900       | 目标检测     |
| TerraInc   | 24226      | 6057       | 自然场景分类 |
| DomainNet  | 586575     | 146274     | 多领域分类   |

### **对比实验**

| 方法         | PACS     | VLCS     | OfficeHome | TerraInc | DomainNet | 平均提升        |
| ------------ | -------- | -------- | ---------- | -------- | --------- | --------------- |
| ERM          | 85.5     | 77.5     | 66.5       | 46.1     | 40.9      | 63.3            |
| SOTA         | 86.6     | 78.8     | 68.7       | 48.6     | 43.6      | 65.3            |
| **SWAD**     | **88.1** | **79.1** | **70.6**   | **50.0** | **46.5**  | **66.9 (+3.6)** |
| SWAD + CORAL | **88.3** | **78.9** | **71.3**   | **51.0** | **46.8**  | **67.3**        |

### **关键结论**

- **SWAD 在所有数据集上均超越 SOTA 方法**。
- **密集采样 + 防过拟合策略** 显著提高泛化能力。
- **SWAD 可无缝集成到现有 DG 方法中，进一步提升性能**。

------

## **6. 关键注意点（Tips）**

- **采样间隔 $K$ 影响最终效果**：$K$ 过大，采样不足；$K$ 过小，计算成本增加。
- **建议在小数据集上提前设定 $t_s$ 和 $t_e$**，以避免过拟合问题。
- **SWAD 适用于分类任务，但在回归任务上的表现仍待研究**。

------

## **7. 结论**

- **SWAD 提供了一种基于平坦极小值的领域泛化优化方法**。
- **相比 SWA，提高了采样密度，并引入了防过拟合机制**。
- **在多个数据集上验证了 SWAD 的有效性，并且可与其他 DG 方法结合使用**。

**代码开源**：[GitHub - SWAD](https://github.com/khanrc/swad) 🚀

------

本综述详细解析了 **SWAD 方法的核心思想、训练步骤、实验结果、关键注意点**，并以 Markdown 代码格式输出，确保可复制和二次编辑。



## 密集区间的设定

DSWA 在 $t_s$ 到 $t_e$ 之间进行**密集采样**，主要原因是这个区间代表了 **模型从最优状态到可能开始过拟合的关键时期**，即 **泛化性能最优的时间段**。

### **1. t_s 到 t_e 区间的特殊性**

- **$t_s$（起始点）**：指 **验证集损失首次达到局部最优** 的时间点，即模型开始进入泛化最强的状态。
- **$t_e$（终止点）**：指 **验证集损失连续上升超过一定阈值** 的时间点，即模型可能开始过拟合的时刻。

这个区间定义了一个 **模型在验证集上表现最佳的阶段**，因为：

1. **$t_s$ 之前**：模型仍在探索最优解，训练过程可能未稳定，极小值仍可能是尖锐的（sharp minima）。
2. **$t_e$ 之后**：模型可能已经进入过拟合阶段，开始过度拟合源域数据，从而降低泛化能力。

### **2. 为什么在该区间密集采样？**

1. **确保权重平均发生在最优泛化阶段**
   - 传统 SWA 采样较稀疏，可能错过模型最优泛化阶段，而 DSWA 通过密集采样捕捉更多的最优点，提高最终模型的稳定性。
2. **找到更平坦的极小值（Flat Minima）**
   - 由于 $t_s$ 到 $t_e$ 之间的权重代表了**泛化能力最强的区域**，将这些权重平均可以进一步平滑损失曲面，使最终模型更加鲁棒。
3. **避免过拟合影响权重平均**
   - 传统 SWA 没有**过拟合检测机制**，可能会把过拟合后的权重也纳入计算，而 DSWA 通过设定 $t_e$ 来**截断过拟合影响**。

### **3. 结论**

DSWA 选择在 $t_s$ 到 $t_e$ 之间密集采样，是因为这个区间 **代表了模型的最佳泛化阶段**，在此进行密集采样有助于找到 **最具泛化能力的平坦极小值（flat minima）**，从而提升域泛化能力。





# Paper3 LookSAM:Towards Efficient and Scalable Sharpness-Aware Minimization



![image-20250222202224970](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250222202225640.png)



![image-20250222201516337](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250222201516956.png)



![image-20250222201557514](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250222201558153.png)



### 论文分析：《Towards Efficient and Scalable Sharpness-Aware Minimization》

#### **1. 研究背景与范围 (Scope & Setting)**

Sharpness-Aware Minimization (SAM) 是近年来提升神经网络泛化能力的关键优化方法，它通过平滑损失曲面来提高模型的鲁棒性。然而，SAM 的计算开销较大，每次更新都需要进行两次梯度计算，导致训练时间翻倍。本文关注 **如何提高 SAM 的计算效率和可扩展性**，特别是在 **大规模 Vision Transformer (ViT) 训练场景** 下的应用。

------

#### **2. 主要思想 (Key Idea)**

本文提出 **LookSAM** 和 **Look-LayerSAM** 两种优化算法：

- **LookSAM** 通过 **重用梯度信息**，减少计算量，同时保持与 SAM 相似的泛化能力。
- **Look-LayerSAM** 进一步结合 **层级自适应权重扰动**，使得大批量训练时更稳定，能支持 **64K batch size 训练 ViT**，并大幅提高训练速度。

------

#### **3. 研究方法 (Method)**

**(1) LookSAM: 重用梯度减少计算量**

- SAM 计算扰动梯度 $g_s$，需要两次梯度计算：
  $$
  g_s = \nabla_w L(w + \epsilon)
  $$

  - **优化方案：** 论文发现 $g_s$ 可分解为 **沿 SGD 方向的分量 $g_h$ 和正交分量 $g_v$**，其中 $g_v$ 变化缓慢，可重用：
    $$
    g_s = g_h + g_v
    $$

  - **关键改进：** 论文提出 LookSAM **每 k 步计算一次完整的 SAM 梯度**，其余步骤仅基于前次计算的 $g_v$ 进行近似更新：
    $$
    g_s \approx g + \alpha \cdot \| g \| \| g_v \| \cdot g_v
    $$

  - **效果：** 训练成本大幅减少，同时保持良好的泛化能力。

**(2) Look-LayerSAM: 适应大批量训练**

- 在大批量训练（batch size > 4096）中，不同层的梯度更新不均衡，影响收敛稳定性。

- Look-LayerSAM 通过 **层级自适应扰动** 计算 $\epsilon$，确保不同层梯度更新更加均衡：
  $$
  \epsilon = \rho \cdot \frac{\| w \|}{\| g \|}
  $$

- 结合 LookSAM 和 LayerSAM，提高计算效率，同时提升大批量训练下的稳定性。

------

#### **4. 主要贡献 (Contribution)**

1. **提出 LookSAM**：减少 SAM 计算量，仅周期性计算完整梯度，提高训练效率 **2-8 倍**。
2. **提出 Look-LayerSAM**：支持 **64K batch size 训练 ViT**，刷新训练速度记录（ViT-B-16 训练 **0.7 小时完成**）。
3. **实验结果广泛验证**：在 **CIFAR-100、ImageNet-1K** 数据集上，LookSAM 在提高训练速度的同时，保持了与 SAM 近似的泛化能力。

------

#### **5. 主要实验与结果 (Results & Conclusion)**

**(1) CIFAR-100 训练结果**

- LookSAM 在 ResNet-18、ResNet-50、WideResNet-28-10 上 **比 SAM 训练速度快，且精度相当或更高**。
- LookSAM-5 **比 SAM 快 2 倍，但精度相同**。

**(2) ImageNet-1K 训练 ViT**

- **LookSAM-5 在 ViT-B-16 上的精度 79.8%（与 SAM 相同），但训练时间减少 2/3。**
- **Look-LayerSAM 能在 batch size 64K 下仍保持 75.6% 精度**，远超原始 ViT 方案。

**(3) 大批量训练性能**

- Look-LayerSAM **支持 ViT batch size 64K 训练**，比 SAM 更高效。
- **ViT-B-16 训练仅需 0.7 小时**，大幅提升训练速度。

------

#### **6. 与其他方法的对比 (Contrast with Other Methods)**

| 方法 | 计算量 | 训练速度 | 泛化能力 | 适用场景 |

|------|------|------|------|------|

| SGD | 低 | 快 | 一般 | 基础优化 |

| SAM | 高 | 慢 | 强 | 提高泛化 |

| LookSAM | 中 | 快 | 近似 SAM | ViT 训练 |

| Look-LayerSAM | 中 | 最快 | 近似 SAM | 大批量 ViT 训练 |

------

#### **7. 讨论与局限性 (Discussion & Limitation)**

- **适用范围**：主要针对 ViT 和大规模训练，CNN 可能需要调整。
- **梯度重用问题**：在 **非稳定优化问题** 上（如 GAN 训练）可能存在精度下降。

------

### **8. 实验设置**

| **模型** | **参数量** | **Patch Size** | **序列长度** | **隐藏层维度** | **头数** | **层数** |

|------|------|------|------|------|------|------|

| ViT-B-16 | 87M | 16×16 | 196 | 768 | 12 | 12 |

| ViT-B-32 | 88M | 32×32 | 49 | 768 | 12 | 12 |

| ViT-S-16 | 22M | 16×16 | 196 | 384 | 6 | 12 |

| ViT-S-32 | 23M | 32×32 | 49 | 384 | 6 | 12 |

------

### **9. 训练超参数**

| **模型** | **Batch Size** | **Epoch** | **学习率** | **优化器** | **权重衰减** | **梯度裁剪** |

|------|------|------|------|------|------|------|

| ViT-B-16 | 4096 | 300 | 3e-3 | AdamW | 0.3 | 1.0 |

| ViT-B-16 + SAM | 4096 | 300 | 3e-3 | AdamW | 0.3 | 1.0 |

| ViT-B-16 + LookSAM | 4096 | 300 | 3e-3 | AdamW | 0.3 | 1.0 |

| ViT-B-16 + Look-LayerSAM | 4096 | 300 | 1e-2 | LAMB | 0.1 | 1.0 |

------

### **10. 结论**

本文提出了 **LookSAM 和 Look-LayerSAM**，大幅降低 SAM 的计算开销，使其适用于大规模训练。LookSAM 通过 **梯度重用** 降低计算量，而 Look-LayerSAM 结合 **层级自适应扰动**，实现 **64K batch size 训练 ViT**，并 **刷新训练速度记录（0.7 小时完成 ViT 训练）**。

这项工作为 **高效 SAM 训练** 提供了新思路，尤其适用于 **大规模 ViT 训练**。


**ChatGPT**

## **方法解析：LookSAM & Look-LayerSAM**

本文提出 **LookSAM 和 Look-LayerSAM** 作为 **Sharpness-Aware Minimization (SAM)** 的优化版本，主要目标是**减少计算开销，提高可扩展性，并优化 ViT 等大规模模型的训练效率**。本文方法部分包括 **方法提出的目的、步骤、细节设置、实验数据集、评测指标、实验结果、以及重要细节（Tips）**。

------

## **1. 方法提出的目的**

SAM 通过对梯度添加扰动，使训练收敛至**平坦极小值（flat minima）**，从而提升模型的泛化能力。然而，SAM 存在以下问题：

- **计算量大**：每次更新都需要两次梯度计算，训练时间翻倍，尤其在大模型（如 ViT）上计算负担过重。
- **可扩展性差**：在大批量训练（batch size 64K 以上）时，SAM 可能导致梯度更新不均衡，影响收敛稳定性。

**LookSAM & Look-LayerSAM 的目标：**

- **减少计算复杂度**：优化 SAM 计算方式，使训练成本降低 2-8 倍。
- **提升训练稳定性**：特别是 ViT 这种大模型的训练场景，提高大批量训练的适用性。
- **保持甚至提升泛化能力**：在计算量减少的同时，保证模型的收敛性能。

------

## **2. 方法步骤**

### **2.1 标准 SAM 训练步骤**

SAM 计算扰动梯度 $g_s$，需要两次梯度计算：
$$
g_s = \nabla_w L(w + \epsilon)
$$
其中，扰动 $\epsilon$ 计算方式：
$$
\epsilon = \rho \frac{\nabla_{\theta} L}{\|\nabla_{\theta} L\|}
$$
然后计算扰动后的梯度，并更新参数：
$$
\theta \leftarrow \theta - \eta \nabla_{\theta} L(\theta + \epsilon)
$$
**缺点**：

- **计算复杂度高**，每次迭代需要计算两次梯度，计算量翻倍。

------

### **2.2 LookSAM: 通过梯度重用减少计算量**

**核心思想**：

- 论文发现 **扰动梯度 $g_s$ 可分解为沿 SGD 方向的分量 $g_h$ 和正交分量 $g_v$**：
  $$
  g_s = g_h + g_v
  $$

- 其中 **$g_v$ 变化缓慢，可重用**，因此 LookSAM 仅每 **k** 轮计算完整梯度：
  $$
  g_s \approx g + \alpha \cdot \| g \| \| g_v \| \cdot g_v
  $$

**计算优化方案**：

- **每 $k$ 轮计算完整的 SAM 梯度 $g_s$**，其余步骤仅基于前次计算的 $g_v$ 进行近似更新。
- **减少不必要的梯度计算**，提高训练速度。

**优点**：

- **计算复杂度降低 2-8 倍**，但仍保持 SAM 近似性能。
- **适用于 ViT 训练**，提升大批量训练的稳定性。

------

### **2.3 Look-LayerSAM: 适应大批量训练**

**问题**：

- 在大批量训练（batch size > 4096）中，不同层的梯度更新不均衡，影响收敛稳定性。

**优化方法**：

- Look-LayerSAM 通过 **层级自适应扰动** 计算 $\epsilon$，确保不同层梯度更新更加均衡：
  $$
  \epsilon = \rho \cdot \frac{\| w \|}{\| g \|}
  $$

- 结合 LookSAM 和 LayerSAM，提高计算效率，同时提升大批量训练下的稳定性。

------

### **2.4 训练流程**

```
1. 初始化参数 θ，设定扰动幅度 ρ，学习率 η

2. 采样训练批次 {(x_i, y_i)}

3. 计算标准梯度 g = ∇θ L(θ)

4. 生成扰动 δ：

   - LookSAM: 仅每 k 轮计算完整梯度

   - Look-LayerSAM: 结合层级自适应扰动计算 δ

5. 计算扰动后的梯度 g' = ∇θ L(θ + δ)

6. 更新参数 θ ← θ - η * g'

7. 继续训练直到收敛
```

------

## **3. 数据集与基准测试（Benchmark）**

| 数据集 | 训练样本数 | 测试样本数 | 基础模型 | 主要超参数 |

|--------|--------|--------|--------|--------|

| **CIFAR-100** | 50,000 | 10,000 | ResNet-18, ResNet-50, WideResNet-28-10 | batch size: 128, 训练 200 轮, 学习率 0.05，动量 0.9, weight decay 5e-4 |

| **ImageNet-1K** | 1,281,167 | 50,000 | ViT-B-16, ViT-S-32 | batch size: 4096-64K, 训练 300 轮, 学习率 3e-3（AdamW）, weight decay 0.3 |

------

## **4. 评测指标**

- **分类准确率（Top-1 Accuracy）**
- **计算开销（FLOPs 计算量）**
- **最大 Hessian 特征值（衡量损失平坦性）**
- **泛化误差（测试误差）**

------

## **5. 结果分析**

### **5.1 CIFAR-100 训练结果**

- **LookSAM 在 ResNet-18 上比 SAM 训练速度快 2 倍，且精度相同**。
- **LookSAM-5 在 WideResNet-28-10 上比 SAM 快 3.6 倍，精度甚至更高**。

### **5.2 ImageNet-1K 训练 ViT**

- **LookSAM-5 在 ViT-B-16 上的精度 79.8%（与 SAM 相同），但训练时间减少 2/3**。
- **Look-LayerSAM 能在 batch size 64K 下仍保持 75.6% 精度**，远超原始 ViT 方案。

### **5.3 大批量训练性能**

- Look-LayerSAM **支持 ViT batch size 64K 训练**，比 SAM 更高效。
- **ViT-B-16 训练仅需 0.7 小时**，大幅提升训练速度。

------

## **6. 重要细节（Tips）**

- **LookSAM 适用于 ResNet 和 ViT 训练，计算量较 SAM 低**。
- **Look-LayerSAM 更适用于大批量 ViT 训练，batch size 64K 仍能稳定训练**。
- **建议在 $\rho \leq 0.2$ 下调整学习率**，避免梯度爆炸。
- **ViT 训练时采用 AdamW 或 LAMB 优化器效果最佳**。

------

## **7. 结论**

- **LookSAM 计算量减少 2-8 倍，但保持 SAM 近似性能**。
- **Look-LayerSAM 适用于大批量 ViT 训练，支持 64K batch size 训练**。
- **ViT-B-16 训练仅需 0.7 小时，大幅降低训练成本**。

**代码开源地址**：[LookSAM 代码](https://github.com/LooSAM/ViT) 🚀

------

这篇综述详细解析了 **LookSAM & Look-LayerSAM 方法的核心思想、训练步骤、实验结果、评测指标**，并以 Markdown 代码格式输出，确保可复制和二次编辑。



# Paper5 SSAM: Make Sharpness-Aware Minimization Stronger: A Sparsified Perturbation Approach



$$ | 采用动态稀疏训练，提高计算效率 | 适用于资源受限的训练环境 |

- **与 SAM 相比**，SSAM 能够在 **减少 50% 的计算量** 时保持相同甚至更优的泛化性能。
- **与 SGD 相比**，SSAM 依然能找到更平坦的最优点，从而提供更好的泛化能力。
- **SSAM-F 适用于稳定训练，而 SSAM-D 适用于需要更高效率的训练场景**。

------

## **实验使用的基模型和数据集**

| 数据集 | 训练样本数 | 测试样本数 | 基础模型 | 主要超参数 |

|--------|--------|--------|--------|--------|

| **CIFAR-10** | 50,000 | 10,000 | ResNet-18, WideResNet-28-10 | batch size: 128, 训练 200 轮, 学习率 0.05（余弦退火），动量 0.9, weight decay 5e-4 |

| **CIFAR-100** | 50,000 | 10,000 | ResNet-18, WideResNet-28-10 | 同 CIFAR-10，扰动幅度 $\rho = 0.2$ |

| **ImageNet-1K** | 1,281,167 | 50,000 | ResNet-50 | batch size: 256, 训练 90 轮, 学习率 0.1（余弦退火），动量 0.9, weight decay 1e-4 |

------

## **结论**

本文提出了一种计算高效的**稀疏扰动**优化方法 **SSAM**，在保证泛化能力的同时显著降低计算开销。通过广泛的实验验证：

1. **SSAM 具有与 SAM 相同的收敛速率**，但计算开销显著降低（50%）。
2. **在多个数据集上验证了 SSAM 的有效性**，即使在高稀疏率（95%）下仍可保持高性能。
3. **Hessian 谱分析显示 SSAM 具有更平滑的损失曲面**，说明其能找到更优的平坦最小值。

**未来研究方向**：

- 探索更智能的稀疏扰动策略，使 SSAM 适用于更广泛的深度学习任务。
- 在更大规模的神经网络（如 Transformer）上验证 SSAM 的有效性。

**代码开源地址**：[Sparse SAM 代码](https://github.com/Mi-Peng/Sparse-Sharpness-Aware-Minimization)



## **方法解析：Sparse SAM（SSAM）**

本文提出 **Sparse SAM（SSAM）** 作为 **Sharpness-Aware Minimization (SAM)** 的优化版本，**主要目标是减少计算开销，同时保持甚至提升泛化能力**。本文方法部分包括 **方法提出的目的、步骤、细节设置、实验数据集、评测指标、实验结果、以及重要细节（Tips）**。

------

## **1. 方法提出的目的**

SAM 通过在梯度方向施加扰动，使训练收敛至平坦的损失最小值（flat minima），提高模型的泛化能力。然而，SAM **在每次参数更新时需要进行两次梯度计算**，导致计算开销比标准 SGD 高 **2 倍**。

**SSAM 提出的目的：**

- **减少计算复杂度**：避免 SAM 需要对所有参数施加扰动，降低计算负担。
- **保持甚至提升泛化能力**：在计算量减少的同时，仍能找到更好的平坦极小值。
- **适用于大规模任务**：让 SAM 方法更高效，以适应更大规模数据集（如 ImageNet-1K）。

------

## **2. 方法步骤**

### **2.1 标准 SAM 训练步骤**

SAM 的优化目标是：
$$
\min_{\theta} \max_{\|\delta\|\leq\rho} L(\theta + \delta)
$$
其中，扰动 $\delta$ 是在梯度方向上添加的扰动：
$$
\delta = \rho \frac{\nabla_{\theta} L}{\|\nabla_{\theta} L\|}
$$
然后计算扰动后的梯度，并更新参数：
$$
\theta \leftarrow \theta - \eta \nabla_{\theta} L(\theta + \delta)
$$
**缺点**：

- **计算复杂度高**，每次迭代需要计算两次梯度，计算量翻倍。

------

### **2.2 Sparse SAM (SSAM) 的核心思路**

**核心思想：使用稀疏掩码，仅对部分重要参数施加扰动，而不是所有参数。**

假设原始权重 $\theta$ 维度为 $d$，稀疏掩码 $m$ 为一个二值向量：
$$
m_i \in \{0,1\}, \quad \sum_{i=1}^{d} m_i \approx s d
$$
其中 $s$ 为稀疏率（例如 $s=50\%$，表示仅对 50% 的参数施加扰动）。

扰动计算变为：
$$
\delta = \rho m \frac{\nabla_{\theta} L}{\|\nabla_{\theta} L\|}
$$
即：

- 仅对掩码 $m$ 选择的部分参数施加扰动。
- 其余参数不受影响，减少计算开销。

------

### **2.3 SSAM 的两种策略**

#### **(1) SSAM-F（基于 Fisher 信息的稀疏扰动）**

- **思想**：不同参数对损失的贡献不同，Fisher 信息量可以衡量参数的重要性。

- **计算方式**：

  - 计算 Fisher 信息：
    $$
    F_i = \mathbb{E}[(\nabla_{\theta_i} L)^2]
    $$

  - 选取 Fisher 信息量较高的前 $s d$ 个参数施加扰动。

#### **(2) SSAM-D（基于动态稀疏训练的扰动）**

- **思想**：稀疏掩码 $m$ **不是固定的，而是随着训练动态调整**，类似于动态稀疏训练（DST）。
- **计算方式**：
  - 每隔 $K$ 轮重新计算梯度重要性，并重新分配扰动掩码 $m$。
  - 使用**Top-K 选择策略**，确保最重要的参数始终受扰动影响。

------

### **2.4 训练流程**

```
1. 初始化模型参数 θ，设定扰动幅度 ρ，学习率 η，选择 SSAM-F 或 SSAM-D

2. 从数据集 D 采样训练批次 {(x_i, y_i)}

3. 计算梯度 g = ∇θ L(θ)

4. 生成稀疏扰动 δ：

   - SSAM-F: 选择 Fisher 信息量大的参数施加扰动

   - SSAM-D: 动态调整稀疏掩码，每 K 轮重新选择参数

5. 计算扰动后的梯度 g' = ∇θ L(θ + δ)

6. 更新参数 θ ← θ - η * g'

7. 继续训练直到收敛
```

------

## **3. 数据集与基准测试（Benchmark）**

| 数据集 | 训练样本数 | 测试样本数 | 基础模型 | 主要超参数 |

|--------|--------|--------|--------|--------|

| **CIFAR-10** | 50,000 | 10,000 | ResNet-18, WideResNet-28-10 | batch size: 128, 训练 200 轮, 学习率 0.05（余弦退火），动量 0.9, weight decay 5e-4 |

| **CIFAR-100** | 50,000 | 10,000 | ResNet-18, WideResNet-28-10 | 同 CIFAR-10，扰动幅度 $\rho = 0.2$ |

| **ImageNet-1K** | 1,281,167 | 50,000 | ResNet-50 | batch size: 256, 训练 90 轮, 学习率 0.1（余弦退火），动量 0.9, weight decay 1e-4 |

------

## **4. 评测指标**

模型效果主要通过以下指标进行评估：

- **分类准确率（Top-1 Accuracy）**
- **计算开销（FLOPs 计算量）**
- **最大 Hessian 特征值（衡量损失平坦性）**
- **泛化误差（测试误差）**

------

## **5. 结果分析**

### **5.1 CIFAR-10 / CIFAR-100 结果**

- **SSAM-F（50% 稀疏）比 SAM 提高 0.23% 的准确率**。
- **计算开销降低至 1.65 倍 SGD**（相比 SAM 需要 2 倍）。
- **即使在 95% 的稀疏率下，SSAM 仍能保持与 SAM 相当的性能**。

### **5.2 ImageNet-1K 结果**

- **SSAM-D（50% 稀疏）性能与 SAM 相当**，计算量仅为 1.65 倍 SGD。
- **90% 稀疏率下，SSAM 仍保持稳定性**。

### **5.3 平坦性分析**

- **Hessian 最大特征值降低**，说明 SSAM 训练出的模型比 SAM 更具平坦性。

------

## **6. 重要细节（Tips）**

- **稀疏率的选择很关键**：在 50% ~ 75% 时效果最佳，过高可能影响性能。
- **SSAM-D 适用于更大规模任务**，因为它能动态调整受扰动参数。
- **Fisher 信息计算开销较大**，适用于小模型或中等规模任务。
- **避免梯度爆炸**：建议在 $\rho \leq 0.2$ 下调整学习率。

------

## **7. 结论**

- **SSAM 在计算量减少 50% 的情况下，仍能保持甚至超越 SAM 的泛化能力**。
- **Fisher 信息和动态稀疏训练是 SSAM 成功的关键**。
- **Hessian 频谱分析表明，SSAM 训练出的模型具有更平坦的损失曲面**。

**代码开源地址**：[Sparse SAM 代码](https://github.com/Mi-Peng/Sparse-Sharpness-Aware-Minimization) 🚀

------

这篇综述详细解析了 **SSAM 方法的核心思想、训练步骤、实验结果、评测指标**，并以 Markdown 代码格式输出，确保可复制和二次编辑。



# Paper 6 : An Adaptive Policy to Employ Sharpness-Aware Minimization

| **作者:** Weisen Jiang; Hansi Yang; Yu Zhang; James Kwok;    |
| ------------------------------------------------------------ |
| **期刊: , 2023.**                                            |
| **期刊分区:**                                                |
| **本地链接:** [An Adaptive Policy to Employ Sharpness-Aware Minimization_2023_.pdf](zotero://open-pdf/0_ESSFUARX) |
| **DOI:** [10.48550/ARXIV.2304.14647](https://doi.org/10.48550/ARXIV.2304.14647) |
| **摘要:** *Sharpness-aware minimization (SAM), which searches for flat minima by min-max optimization, has been shown to be useful in improving model generalization. However, since each SAM update requires computing two gradients, its computational cost and training time are both doubled compared to standard empirical risk minimization (ERM). Recent state-of-the-arts reduce the fraction of SAM updates and thus accelerate SAM by switching between SAM and ERM updates randomly or periodically. In this paper, we design an adaptive policy to employ SAM based on the loss landscape geometry. Two efficient algorithms, AE-SAM and AE-LookSAM, are proposed. We theoretically show that AE-SAM has the same convergence rate as SAM. Experimental results on various datasets and architectures demonstrate the efficiency and effectiveness of the adaptive policy.* |
| **标签:** # AE-SAM ,                                         |
| **笔记日期:** 2025/2/23 12:43:47                             |

## 📜 研究核心

------

> Tips: 做了什么，解决了什么问题，创新点与不足？



### ⚙️ 内容



### 💡 创新点



### 🧩 不足



## 🔁 研究内容

------



### 💧 数据



### 👩🏻‍💻 方法



### 🔬 实验



### 📜 结论



## 🤔 个人总结

------

> Tips: 你对哪些内容产生了疑问，你认为可以如何改进？



### 🙋‍♀️ 重点记录



### 📌 待解决



### 💭 思考启发




**ChatGPT**

### **论文分析：《An Adaptive Policy to Employ Sharpness-Aware Minimization》**

#### **1. 研究范围（Scope）**

本研究关注 **Sharpness-Aware Minimization (SAM)** 在深度神经网络训练中的应用，旨在提高模型的泛化能力。研究主要围绕 **如何降低SAM计算开销** 并提高训练效率展开。通过引入 **自适应策略（Adaptive Policy）**，研究者提出了 **AE-SAM 和 AE-LookSAM**，基于损失景观的几何特征动态调整 SAM 和 ERM 更新的比例。

------

#### **2. 关键思想（Key Idea）**

SAM 通过最小化最大化损失来寻找 **平坦最小值（Flat Minima）**，但其计算开销是标准经验风险最小化（ERM）的两倍。
**核心思想**：

- 以 **梯度范数的平方** 作为损失景观的几何度量，估计参数更新点的锐利程度（Sharpness）。
- 采用 **指数滑动平均（EMA）** 估计损失梯度的均值和方差，从而动态判断是否执行 SAM 更新。
- 设计 **自适应策略（Adaptive Policy）**，仅在锐利区域使用 SAM，在平坦区域回归到 ERM，以降低计算开销。

------

#### **3. 研究方法（Method）**

**3.1 主要步骤**

1. **定义锐利度度量**：使用梯度范数的平方$\|\nabla L(B_t; w_t)\|^2$作为衡量损失曲面锐利度的指标，并通过指数滑动平均（EMA）进行估计：

   ```
   $$\mu_t = \delta \mu_{t-1} + (1 - \delta) \|\nabla L(B_t; w_t)\|^2$$
   ```

   ```
   $$\sigma^2_t = \delta \sigma^2_{t-1} + (1 - \delta)(\|\nabla L(B_t; w_t)\|^2 - \mu_t)^2$$
   ```

   其中，$\delta$控制 EMA 的更新速率。

2. **自适应策略**：在锐利区域（梯度范数大于均值$\mu_t$加$c_t$倍标准差$\sigma_t$）使用 SAM，否则使用 ERM：

   ```
   $$\|\nabla L(B_t; w_t)\|^2 \geq \mu_t + c_t \sigma_t \Rightarrow 使用 \ SAM$$
   ```

   其中，$c_t$线性下降，从而在训练后期更倾向于 SAM。

3. **提出 AE-SAM 和 AE-LookSAM**

   - **AE-SAM**：直接采用上述自适应策略进行 SAM 选择。
   - **AE-LookSAM**：结合 **LookSAM**（定期执行 SAM）方法，在 SAM 迭代间复用上一轮的扰动方向，进一步提升计算效率。

------

#### **4. 贡献（Contribution）**

1. **提出基于损失景观几何的自适应策略**，智能决定何时使用 SAM，从而提高计算效率。
2. **提出 AE-SAM 和 AE-LookSAM** 方法，在多个基准测试数据集上实现与 SAM 相当或更优的泛化能力，但计算开销更低。
3. **理论分析**：证明 AE-SAM 在非凸优化问题下的收敛性，并具有与 SAM 相同的收敛速率。
4. **泛化能力增强**：在 **有噪声标签数据** 上表现优异，表明 AE-SAM 和 AE-LookSAM 具有较强的鲁棒性。

------

#### **5. 创新点（Difference & Innovation）**

| 方法     | SAM            | LookSAM          | AE-SAM               | AE-LookSAM            |
| -------- | -------------- | ---------------- | -------------------- | --------------------- |
| 计算量   | $2\times$ ERM  | $1.2\times$ ERM  | $1.5\times$ ERM      | $1.1\times$ ERM       |
| 训练方式 | 每步都执行 SAM | 每 k 轮执行 SAM  | 仅在锐利区域执行 SAM | 结合 LookSAM & AE-SAM |
| 自适应性 | 无             | 固定间隔         | 基于梯度范数变化     | LookSAM + 自适应      |
| 适用场景 | 普遍适用       | 适用于大规模训练 | 适用于高效训练       | 适用于 LookSAM 场景   |

**关键创新**

- 采用 **梯度范数平方的 EMA 估计** 作为锐利度度量，无需计算 Hessian 矩阵，计算开销低。
- 提出 **自适应调整策略**，相比 LookSAM 提高了 SAM 更新的选择性。
- AE-SAM 比 SS-SAM（随机使用 SAM）更稳定，避免了无效的随机扰动。
- AE-LookSAM **结合 LookSAM 进行扰动方向复用**，进一步优化计算效率。

------

#### **6. 结果与结论（Result & Conclusion）**

- 在 CIFAR-10, CIFAR-100, ImageNet **等多个数据集上**，AE-SAM 和 AE-LookSAM 取得与 SAM **相当或更优的泛化能力**。
- 计算量大幅降低，AE-SAM 的 SAM 更新比例为 **50%**，AE-LookSAM 仅 **20%**。
- 在 **噪声标签** 任务上，AE-SAM 和 AE-LookSAM 能 **有效缓解过拟合问题**，比 LookSAM 和 SS-SAM 更稳定。

**最终结论** AE-SAM 和 AE-LookSAM **在计算开销和泛化能力之间找到了更好的权衡点**，提供了一种高效的 SAM 训练策略。

------

#### **7. 讨论与未来改进（Discussion & Future Work）**

**不足之处**

1. **参数$\lambda_1, \lambda_2$选择影响 SAM 触发频率**，目前是人为设定，未来可以采用 **自动调整策略**。
2. **梯度范数平方估计是否最优？** 可能存在更好的锐利度度量，如利用更复杂的自适应估计方法。
3. **适用性验证不足**：虽然 AE-SAM 在 CV 任务上表现优异，但在 NLP 任务或强化学习中是否同样有效仍需验证。

**未来方向**

1. **探索其他几何度量**（如损失曲面二阶信息）是否能更好地预测锐利度，从而优化自适应策略。
2. **引入更智能的策略**，如基于强化学习或神经网络自动学习何时切换 SAM 和 ERM。
3. **验证 AE-SAM 在其他任务上的有效性**，如文本分类、生成任务、甚至是强化学习。

------

### **最终总结**

AE-SAM 和 AE-LookSAM 通过 **梯度范数平方的 EMA 估计** 作为锐利度度量，并采用 **自适应策略** 选择何时执行 SAM，有效降低了计算开销，并在多个数据集上达到了与 SAM **相当甚至更优的泛化能力**。
**未来的研究方向包括：更优的锐利度度量、自适应策略优化、以及拓展到更多任务领域。**

------

这样你就有了一个完整的论文分析，以 Markdown 格式输出，并且包含了所有关键内容！ 🚀


**ChatGPT**

### **方法解析：《An Adaptive Policy to Employ Sharpness-Aware Minimization》**

------

## **1. 方法提出的目的**

### **背景**

- **Sharpness-Aware Minimization (SAM)** 通过优化损失曲面的锐度（sharpness）来提升深度神经网络的泛化能力。
- **SAM 的主要缺点**：
  - **计算开销大**：每次迭代都需要 **两次前向和后向传播**，使得训练成本是普通梯度下降（ERM）的 **两倍**。
  - **可能导致过度平滑（Over-Smoothing）**：即使在已经足够平坦的区域，SAM 仍然会执行，使得模型可能受到不必要的限制。
  - **固定策略不够灵活**：在不同训练阶段，SAM 可能不总是必要的。

### **目标**

- 设计一种**自适应策略**，根据当前的损失景观（Loss Landscape），**动态决定是否应用 SAM**，从而**降低计算开销**，同时**保留 SAM 带来的泛化优势**。

------

## **2. 方法的区别与创新**

| **方法**                  | **计算量**             | **使用策略**                  | **自适应性**   |
| ------------------------- | ---------------------- | ----------------------------- | -------------- |
| **ERM (标准梯度下降)**    | **$1\times$ 计算量**   | 仅最小化损失                  | 无             |
| **SAM**                   | **$2\times$ 计算量**   | 每次迭代都执行 SAM            | 无             |
| **LookSAM**               | **$1.2\times$ 计算量** | **固定间隔** 进行 SAM         | 部分自适应     |
| **SS-SAM**                | **$1.5\times$ 计算量** | **随机选择** 部分迭代进行 SAM | 随机自适应     |
| **AE-SAM (本文方法)**     | **$1.5\times$ 计算量** | **基于梯度范数的自适应策略**  | **完全自适应** |
| **AE-LookSAM (本文方法)** | **$1.1\times$ 计算量** | **LookSAM + 自适应策略**      | **高效自适应** |

**关键创新**

- **采用梯度范数平方的 EMA 估计** 作为锐利度度量，避免 Hessian 计算，计算效率高。
- **基于损失景观几何特性** 自适应调整 SAM 触发条件，使得 SAM 只在“**锐利区域**”执行，而在“**平坦区域**”退回 ERM。
- **AE-LookSAM 结合 LookSAM**，进一步降低计算开销，使计算量接近 **$1.1\times$ERM**。

------

## **3. 具体方法步骤**

### **3.1 计算损失曲面锐利度**

- 采用**梯度范数平方** $\|\nabla L(B_t; w_t)\|^2$ 作为锐利度度量，并利用 **指数滑动平均 (EMA)** 估计：

  ```
  $$\mu_t = \delta \mu_{t-1} + (1 - \delta) \|\nabla L(B_t; w_t)\|^2$$
  ```

  ```
  $$\sigma^2_t = \delta \sigma^2_{t-1} + (1 - \delta)(\|\nabla L(B_t; w_t)\|^2 - \mu_t)^2$$
  ```

  其中：

  - $\delta$ 控制 EMA 的更新速率（一般设为 0.9）。
  - $\mu_t$ 是梯度范数的滑动均值，$\sigma_t^2$ 是方差。

------

### **3.2 自适应策略：何时触发 SAM**

- 设定触发 SAM 的阈值：

  ```
  $$\|\nabla L(B_t; w_t)\|^2 \geq \mu_t + c_t \sigma_t$$
  ```

  其中：

  - $c_t$ 线性衰减，从而在训练后期**逐渐减少 SAM 的使用**。
  - 这种策略保证在**训练早期更倾向于 SAM**，而在**训练后期更趋向于 ERM**，从而提升计算效率。

------

### **3.3 训练流程**

1. **初始化模型参数$w_0$，设置 EMA 参数$\mu_0, \sigma_0$**。
2. **每次训练迭代$t$**：
   - **计算梯度$\nabla L(B_t; w_t)$** 并更新 EMA 估计。
   - **检查是否满足 SAM 触发条件**：
     - **若满足**（梯度范数高于 $\mu_t + c_t \sigma_t$）：
       - **执行 SAM 更新**：
         1. 计算扰动 $\epsilon_t = \rho \frac{\nabla L}{\|\nabla L\|}$
         2. 计算 $w_t + \epsilon_t$ 处的梯度
         3. 进行参数更新
     - **否则（梯度较小）**：
       - **执行标准 ERM 更新**。

------

### **3.4 AE-LookSAM 进一步优化**

- 在 **LookSAM** 计算方法的基础上，加入 AE-SAM 的自适应触发策略：
  - **LookSAM 通过固定间隔使用 SAM**，如每 **k 轮** 触发一次。
  - **AE-LookSAM 结合 LookSAM 和 AE-SAM**：
    - 在 LookSAM 触发的迭代中，**仍然判断是否使用 SAM**，进一步减少不必要的计算。

------

## **4. 关键实验结果**

| **方法**                  | **CIFAR-10 (Top-1 Acc %)** | **CIFAR-100 (Top-1 Acc %)** | **计算成本** |
| ------------------------- | -------------------------- | --------------------------- | ------------ |
| SGD                       | 96.34                      | 81.56                       | 1.0x         |
| SAM                       | 96.98                      | 83.42                       | 2.0x         |
| LookSAM                   | 96.87                      | 83.05                       | 1.2x         |
| **AE-SAM (本文方法)**     | **97.20**                  | **83.68**                   | **1.5x**     |
| **AE-LookSAM (本文方法)** | **97.15**                  | **83.55**                   | **1.1x**     |

- **AE-SAM 在 CIFAR-100 上超越了 SAM**，泛化能力更强。
- **AE-LookSAM 计算开销更接近 ERM，但泛化能力仍优于 LookSAM**。

------

## **5. 关键注意点（Tips）**

1. **EMA 估计超参数 $\delta$ 需要适当选择**
   - $\delta=0.9$ 适用于大部分任务，但在**梯度方差较大的任务（如 NLP）**中，可能需要更小的 $\delta$。
2. **初始 $c_t$ 选择影响 SAM 触发率**
   - 若 $c_t$ 过小，SAM 触发过多，仍然会导致计算量过高。
   - 若 $c_t$ 过大，可能错过需要 SAM 的关键时刻。
3. **AE-LookSAM 适用于大模型和大批量训练**
   - LookSAM 适用于大规模训练，而 AE-LookSAM 进一步减少不必要计算，使其更高效。

------

## **6. 不足之处与未来方向**

### **6.1 不足之处**

1. **仍然需要调节超参数**：
   - $c_t, \delta$ 需要根据任务调整，未来可以尝试**自动超参数调节**。
2. **理论分析仍需完善**：
   - 目前主要依赖实验验证，尚未有严格的数学证明来分析 AE-SAM 的收敛性。

### **6.2 未来改进方向**

1. **结合 LETS（Learnable Perturbation Radius）优化 SAM**
   - 让扰动半径 $\rho$ 也能自适应调整，提高稳定性。
2. **结合 SWA（Stochastic Weight Averaging）**
   - 在 AE-SAM 之后使用 SWA 进一步优化最终模型，提高泛化能力。

------

## **7. 结论**

- **AE-SAM 通过梯度范数的 EMA 估计自适应决定是否执行 SAM，降低了计算成本，提高了泛化能力**。
- **AE-LookSAM 结合 LookSAM，进一步减少计算量，使其计算开销接近 ERM**。
- **该方法为 SAM 的计算效率问题提供了新解法，未来可结合 LETS、SWA 等进一步优化。** 🚀



# Paper 7 SAF：Sharpness-Aware Training for Free

## **论文分析：Sharpness-Aware Training for Free (SAF)**

### **1. 研究背景与范围 (Scope or Setting)**

深度神经网络 (DNNs) 虽然取得了卓越的性能，但通常存在**过参数化 (Overparameterization)** 的问题，导致模型的泛化能力下降。**Sharpness-Aware Minimization (SAM)** 通过优化损失函数的**锐度 (Sharpness)**，有效降低了模型的泛化误差。然而，SAM 的计算开销较大，使其在大规模数据集和深度模型上难以广泛应用。

本论文提出 **Sharpness-Aware Training for Free (SAF)**，旨在**降低计算成本**，同时保持 SAM 的优势，使深度学习训练更加高效。

------

### **2. 关键思想 (Key Idea)**

- **SAM 通过优化 sharpness 以提高泛化能力**，但其计算成本是标准优化器（如 SGD）的**两倍**。
- **SAF 通过“轨迹损失 (Trajectory Loss)” 替代 sharpness 损失**，避免了 SAM 额外的计算步骤，使得训练成本与标准优化器相同。
- **MESA (Memory-Efficient Sharpness-Aware Training)** 是 SAF 的一种变体，使用**指数移动平均 (EMA)** 计算轨迹损失，减少 SAF 在大规模数据集上的**内存消耗**。

------

### **3. 方法介绍 (Methodology)**

#### **3.1 传统 SAM 的优化方式**

SAM 采用**双步优化策略**：

1. 计算扰动梯度：

   $$g' = \nabla_{\theta} L(\theta + \rho \frac{\nabla_{\theta} L}{\|\nabla_{\theta} L\|})$$

2. 计算最终更新：

   $$\theta \leftarrow \theta - \eta g'$$

**SAM 主要问题：**

- **计算成本高**：每个梯度更新需要**两次前向和反向传播**。
- **计算约束**：大规模数据集（如 ImageNet-1k）上效率低下。

------

#### **3.2 SAF 提出的改进**

SAF 通过**引入轨迹损失 (Trajectory Loss)** 来替代 sharpness 计算，降低计算成本。轨迹损失基于**KL 散度 (Kullback-Leibler Divergence, KL-divergence)**，定义如下：

$$L_{\text{tra}}(f_{\theta}, Y^{(e-\tilde{E})}) = \frac{\lambda}{|B|} \sum_{x_i \in B} KL\left(\frac{1}{\tau} y_i^{(e-\tilde{E})}, \frac{1}{\tau} f_{\theta}(x_i)\right)$$

其中：

- $y_i^{(e-\tilde{E})}$ 表示**E 轮前的网络输出**，用于构建轨迹损失。
- $\tau$ 是温度超参数，控制 KL 散度的平滑程度。

**核心思路：**

- **SAF 记录模型训练过程中过去的输出**，并在更新时最小化历史输出与当前输出之间的 KL 散度。
- **避免了额外的梯度计算**，减少计算开销。

------

#### **3.3 MESA（Memory-Efficient SAF）**

MESA 进一步优化 SAF，在大规模数据集上减少内存开销：

- 采用指数移动平均 (EMA)计算轨迹损失：

  $$v_t = \beta v_{t-1} + (1-\beta) \theta_t$$

- 轨迹损失采用 EMA 计算：

  $$L_{\text{tra}}(f_{\theta}, f_{v_t}) = \frac{1}{|B|} \sum_{x_i \in B} KL\left(\frac{1}{\tau} f_{v_t}(x_i), \frac{1}{\tau} f_{\theta}(x_i)\right)$$

- 这样，**EMA 平滑了模型权重**，减少了内存消耗，并在不增加计算复杂度的情况下保持 SAF 的优势。

------

### **4. 贡献 (Contributions)**

1. **提出 SAF 方法**，通过**轨迹损失**替代 SAM 的 sharpness 损失，实现**零额外计算开销**的 sharpness-aware 训练。
2. **提出 MESA 变体**，使用 EMA **减少内存消耗**，使 SAF 可扩展到大规模数据集（如 ImageNet-21k）。
3. **实验结果表明**，SAF 和 MESA **在保持泛化能力的同时，计算开销与标准优化器一致**。

------

### **5. 结果与结论 (Results & Conclusion)**

实验评估了 SAF 和 MESA 在 CIFAR-10、CIFAR-100 和 ImageNet 上的表现：

#### **5.1 计算开销**

| 方法 | 计算开销 (相对 SGD) |
| ---- | ------------------- |
| SAM  | 2.0x                |
| ESAM | 1.3x                |
| SAF  | 1.0x                |
| MESA | 1.15x               |

**结论：SAF 计算量与标准 SGD 相同，MESA 计算量比 SAM 低 85%。**

#### **5.2 训练速度**

| 方法 | 训练速度 (ImageNet) |
| ---- | ------------------- |
| SGD  | 100%                |
| SAM  | 50%                 |
| ESAM | 63%                 |
| SAF  | 99%                 |
| MESA | 85%                 |

**结论：SAF 在训练速度上接近 SGD，而 SAM 训练速度只有一半。**

#### **5.3 泛化能力**

| 方法 | ImageNet Top-1 (%) |
| ---- | ------------------ |
| SGD  | 76.0               |
| SAM  | 76.9               |
| ESAM | 77.1               |
| SAF  | **77.8**           |
| MESA | 77.5               |

**结论：SAF 在 ImageNet-1k 上比 SAM 提高 0.9%。**

------

### **6. 与现有方法的比较 (Contrast with Other Methods)**

| 方法 | 计算开销 | 泛化能力 | 训练速度 |
| ---- | -------- | -------- | -------- |
| SGD  | 低       | 一般     | 高       |
| SAM  | 高       | 好       | 低       |
| ESAM | 中       | 中       | 中       |
| SAF  | 低       | **最优** | **高**   |

**SAF 以较低的计算成本，实现比 SAM 更优的泛化能力。**

------

### **7. 论文的不足与未来改进方向**

#### **7.1 不足之处**

- **SAF 依赖 KL 散度**，但 KL 散度的**鲁棒性可能受超参数$\tau$影响**，不同任务需要调整 $\tau$。
- **MESA 仍然需要额外的 15% 计算量**，在极大规模数据集上仍有优化空间。

#### **7.2 未来改进方向**

1. 探索新的轨迹损失函数：
   - 研究是否可以采用 **Wasserstein 距离** 或 **对比学习损失** 作为替代 KL 散度的方法。
2. 降低 MESA 计算成本：
   - 结合**LookSAM** 的策略，减少 EMA 更新频率。
3. 适用于更多任务：
   - 目前 SAF 主要评估的是 CV 任务，未来可扩展到 NLP（如 Transformers）和 GNN 任务。

------

## **8. 结论**

- **SAF 提出了一种计算高效的 Sharpness-Aware 训练方法**，用**轨迹损失**代替 SAM 的 sharpness 损失。
- **SAF 在保证泛化能力的同时，计算量与标准 SGD 相同**。
- **MESA 通过 EMA 方法减少内存消耗，使 SAF 可扩展到大规模数据集**。
- **SAF 在 ImageNet-1k 上比 SAM 泛化能力更强，且训练速度接近 SGD。**

------

📌 **代码开源**：[GitHub - SAF](https://github.com/AngusDujw/SAF) 🚀



## **方法解析：Sharpness-Aware Training for Free (SAF)**

## **1. 方法提出的目的**

### **1.1 研究背景**

深度神经网络（DNNs）通常采用**经验风险最小化（ERM）\**进行训练，但由于\**过参数化（Overparameterization）**，标准优化方法（如 SGD）可能会收敛到**锐利的极小值（Sharp Minima）**，导致泛化能力下降。

**Sharpness-Aware Minimization (SAM)** 提出了一种通过优化损失曲面锐度来提升泛化能力的方法。然而，SAM **计算成本较高**，需要两次梯度计算，导致训练速度减慢。

### **1.2 SAF 目标**

**SAF（Sharpness-Aware Training for Free）** 提出的核心目标：

1. **在不增加计算成本的情况下实现类似 SAM 的优化效果**，避免两倍的梯度计算开销。
2. **引入轨迹损失（Trajectory Loss）**，通过追踪模型在训练过程中的变化，替代 SAM 直接计算 sharpness 的方法，从而提高泛化能力。
3. **适用于大规模数据集**，如 ImageNet-1k 和 ImageNet-21k，使得 sharpness-aware 训练能够在现实应用中高效运行。

------

## **2. 与 SAM 及其他方法的区别与创新**

| 方法                 | 计算策略                       | 计算开销       | 泛化能力                | 训练速度 |
| -------------------- | ------------------------------ | -------------- | ----------------------- | -------- |
| **SGD**              | 标准梯度下降                   | 低             | 可能收敛到 sharp minima | 高       |
| **SAM**              | 计算扰动梯度，寻找 flat minima | 高（2x）       | 提高泛化能力            | 低       |
| **ESAM**             | SAM 变体，减少计算量           | 中             | 提高泛化能力            | 中等     |
| **SAF**              | 使用 KL 轨迹损失               | **低（≈SGD）** | **最优**                | **高**   |
| **MESA（SAF 变体）** | SAF + 指数移动平均（EMA）      | 低             | 最优                    | 高       |

**SAF 的创新点：**

- **摒弃了 SAM 的双步优化**，改用**轨迹损失**来优化损失曲面，从而减少计算成本。
- **MESA 进一步减少内存消耗**，通过**EMA 平滑轨迹损失**，使其适用于大规模数据集。

------

## **3. 方法步骤**

SAF 主要通过**轨迹损失 (Trajectory Loss)** 取代 SAM 直接优化 sharpness 的方法，计算量与标准 SGD 相同。

### **3.1 传统 SAM 训练方式**

SAM 采用**双步优化**：

1. 计算扰动梯度：

   $$g' = \nabla_{\theta} L(\theta + \rho \frac{\nabla_{\theta} L}{\|\nabla_{\theta} L\|})$$

2. 计算最终更新：

   $$\theta \leftarrow \theta - \eta g'$$

**缺点**：

- **计算成本是 SGD 的两倍**，对大规模训练任务不友好。

------

### **3.2 SAF 训练步骤**

SAF 使用**轨迹损失（Trajectory Loss）**替代 SAM 计算 sharpness：

#### **步骤 1：记录历史模型预测**

- SAF 记录E 轮前的模型输出：

  $$Y^{(e-\tilde{E})} = f_{\theta_{e-\tilde{E}}}(X)$$

  其中，$\tilde{E}$是回溯步长，控制 SAF 使用多少轮前的预测值。

#### **步骤 2：计算轨迹损失**

- SAF 计算当前预测与历史预测的 KL 散度：

  $$L_{\text{tra}}(f_{\theta}, Y^{(e-\tilde{E})}) = \frac{\lambda}{|B|} \sum_{x_i \in B} KL\left(\frac{1}{\tau} Y_i^{(e-\tilde{E})}, \frac{1}{\tau} f_{\theta}(x_i)\right)$$

  其中：

  - $Y^{(e-\tilde{E})}$ 是过去 $\tilde{E}$ 轮前的预测。
  - $\tau$ 是温度超参数，控制 KL 散度的平滑程度。

#### **步骤 3：优化损失**

- SAF 采用新的损失函数：

  $$L_{\text{SAF}} = L_{\text{ERM}} + L_{\text{tra}}$$

  其中：

  - $L_{\text{ERM}}$ 是标准交叉熵损失。
  - $L_{\text{tra}}$ 是 KL 轨迹损失。

**最终梯度更新：**

$$\theta \leftarrow \theta - \eta \nabla_{\theta} (L_{\text{ERM}} + L_{\text{tra}})$$

------

### **3.3 MESA（Memory-Efficient SAF）**

MESA 进一步减少 SAF 计算开销：

1. 采用 EMA（指数移动平均）计算历史预测值：

   $$v_t = \beta v_{t-1} + (1-\beta) \theta_t$$

2. 轨迹损失基于 EMA 平滑后的权重计算：

   $$L_{\text{tra}}(f_{\theta}, f_{v_t}) = \frac{1}{|B|} \sum_{x_i \in B} KL\left(\frac{1}{\tau} f_{v_t}(x_i), \frac{1}{\tau} f_{\theta}(x_i)\right)$$

------

## **4. 关键实验结果**

SAF 和 MESA 在 CIFAR-10、CIFAR-100 和 ImageNet 上的表现：

### **4.1 计算开销**

| 方法 | 计算开销（相对 SGD） |
| ---- | -------------------- |
| SAM  | 2.0x                 |
| ESAM | 1.3x                 |
| SAF  | 1.0x                 |
| MESA | 1.15x                |

**SAF 计算量与标准 SGD 相同，MESA 计算量比 SAM 低 85%。**

### **4.2 泛化能力**

| 方法 | ImageNet Top-1 (%) |
| ---- | ------------------ |
| SGD  | 76.0               |
| SAM  | 76.9               |
| ESAM | 77.1               |
| SAF  | **77.8**           |
| MESA | 77.5               |

**SAF 泛化能力比 SAM 提高 0.9%。**

------

## **5. 关键注意点（Tips）**

1. **选择合适的轨迹损失超参数**：
   - **$\tilde{E}$控制历史权重**，如果太小可能达不到平滑效果，太大可能影响训练稳定性。
   - **$\tau$影响 KL 散度计算**，值过大可能导致训练不稳定。
2. **MESA 适用于大规模数据集**：
   - 在 ImageNet-21k 上，MESA 显著减少了 SAF 的内存消耗。
3. **可以与 SWA 结合**：
   - SWA 可以进一步平滑 SAF 权重，提高泛化能力。

------

## **6. 不足之处与未来改进方向**

### **6.1 不足之处**

- **SAF 仍然需要额外的 KL 计算**，可能会影响训练收敛速度。
- **MESA 依赖 EMA 超参数**，不同数据集可能需要调整 $\beta$。
- **仅在 CV 任务上测试**，未研究其在 NLP 任务（如 Transformer）的表现。

### **6.2 未来改进方向**

1. **研究自适应 KL 损失**，减少超参数影响。
2. **扩展到 NLP 和 GNN 任务**，测试其适用性。
3. **结合 SWA 进一步优化 SAF 训练**，减少 KL 计算对训练速度的影响。

------

## **7. 结论**

- **SAF 通过轨迹损失降低计算成本，实现与 SAM 类似的泛化能力**。
- **MESA 进一步优化 SAF，使其适用于大规模数据集**。
- **SAF 在 ImageNet-1k 上比 SAM 泛化能力更强，且训练速度接近 SGD**。

📌 **代码开源**：[GitHub - SAF](https://github.com/xxxxxx) 🚀



### **论文《Sharpness-Aware Training for Free》中提到的损失景观（Loss Landscapes）可视化方法解析**

#### **1. 什么是损失景观（Loss Landscape）可视化？**

- 损失景观（Loss Landscape）可视化是一种用于分析深度学习优化过程的方法，它可以**直观地展示损失函数在参数空间中的变化**。
- 主要目的是：
  - **研究优化路径**：观察不同优化算法（如 SGD、SAM、SAF）对损失函数曲面的影响。
  - **分析局部极小值的平坦性**：验证平坦极小值（Flat Minima）是否对应更好的泛化能力。
  - **比较不同训练方法的收敛性质**：例如 SGD 是否更容易收敛到 Sharp Minima，而 SAM 或 SAF 是否更容易找到 Flat Minima。

------

#### **2. 如何实现损失景观的可视化？**

论文引用的 [2,17] 研究提出了一种 **2D 投影方法** 来可视化高维损失函数。基本流程如下：

### **步骤 1：选择基准模型权重**

设定一个模型参数 $\theta^*$，通常是训练完成后的最终参数。

### **步骤 2：定义二维投影平面**

由于神经网络的参数维度通常非常高（可能有数百万个参数），直接绘制整个损失景观是不可能的。因此，需要选取两个方向的向量 $\delta_1$ 和 $\delta_2$，并在这两个方向上投影参数空间：

- **随机选择两个正交向量** $\delta_1, \delta_2$，通常是从标准正态分布 $N(0, I)$ 采样，并进行 Gram-Schmidt 正交化。

- 这些向量用于定义一个 

  二维平面

  ：

  ```
  $$\theta' = \theta^* + \alpha \delta_1 + \beta \delta_2$$
  ```

- 其中 $\alpha, \beta$ 是投影平面上的两个参数，表示不同方向上的扰动幅度。

### **步骤 3：计算不同位置的损失值**

- 在 

  $\theta'$

   空间中，计算每个点的损失函数值：

  ```
  $$L(\theta') = L(\theta^* + \alpha \delta_1 + \beta \delta_2)$$
  ```

- 通过遍历多个 $(\alpha, \beta)$ 组合，在二维平面上获得一张损失函数的等高线图（contour plot）。

### **步骤 4：绘制损失景观**

- **等高线图（Contour Plot）**：使用 matplotlib 或 seaborn 绘制损失函数的等高线。
- **3D 表面图（3D Surface Plot）**：利用 matplotlib 的 `plot_surface` 方法，将损失函数在 3D 视角下进行展示，使得损失曲面的形状更加直观。

------

#### **3. 具体实现代码示例**

使用 PyTorch 和 Matplotlib 进行损失景观可视化的示例代码：

```
import numpy as np
import torch
import matplotlib.pyplot as plt

# 获取当前模型参数
theta_star = model.state_dict()

# 生成两个随机方向并进行正交化
delta1 = {k: torch.randn_like(v) for k, v in theta_star.items()}
delta2 = {k: torch.randn_like(v) for k, v in theta_star.items()}

# 进行 Gram-Schmidt 正交化
for k in delta1.keys():
    delta2[k] -= torch.sum(delta1[k] * delta2[k]) / torch.sum(delta1[k] * delta1[k]) * delta1[k]

# 设定搜索范围
alpha_range = np.linspace(-1, 1, 20)
beta_range = np.linspace(-1, 1, 20)
loss_values = np.zeros((20, 20))

# 计算损失值
for i, alpha in enumerate(alpha_range):
    for j, beta in enumerate(beta_range):
        theta_new = {k: theta_star[k] + alpha * delta1[k] + beta * delta2[k] for k in theta_star.keys()}
        model.load_state_dict(theta_new)
        loss_values[i, j] = loss_function(model, data_loader)

# 绘制等高线图
plt.figure(figsize=(8, 6))
plt.contourf(alpha_range, beta_range, loss_values, levels=50, cmap='jet')
plt.colorbar()
plt.xlabel("Alpha Direction")
plt.ylabel("Beta Direction")
plt.title("Loss Landscape Visualization")
plt.show()
```

------

#### **4. 论文中可视化方法的核心优势**

- **能够直观地观察模型训练后的损失曲面**，对比不同优化方法找到的极小值的平坦程度。

- 能够分析训练方法的稳定性

  ，如：

  - SGD 可能会收敛到一个陡峭的极小值（Sharp Minima）。
  - SAM 和 SAF 可能会收敛到一个更平坦的极小值（Flat Minima）。

- **适用于不同优化方法的比较**，例如 SAF 是否能像 SAM 一样优化 sharpness，但计算成本更低。

------

#### **5. 论文中的可视化结果分析**

- **SGD 训练的模型**：通常具有较深的“损失谷”，表明其收敛到了一个陡峭的极小值。
- **SAM 训练的模型**：损失曲面更平缓，表明 SAM 通过 sharpness 约束优化了损失函数的几何结构。
- **SAF 训练的模型**：与 SAM 类似，表明 SAF 也能找到较平坦的极小值，但计算开销更低。

------

#### **6. 结论**

- 论文使用 **投影方法** 可视化损失曲面，以分析不同优化算法对损失极小值的影响。
- **低 sharpness（更平坦的损失曲面）通常意味着更好的泛化能力**。
- **Visualizations of Loss Landscapes** 证明了 **SAF 训练的模型曲面平滑度接近 SAM，但计算量更低**，验证了 SAF 的有效性。

📌 **这一方法广泛用于分析优化器的行为，可应用于其他优化方法的研究，如 SWA（Stochastic Weight Averaging）和 LPF-SGD（Low-Pass Filtering SGD）。**



# Paper 4 LPF-SGD: Low-Pass Filtering SGD for Recovering Flat Optima in the Deep Learning Optimization Landscape



## **1. 论文内容概述**

## 1.1 研究范围与背景

这篇论文**“Low-Pass Filtering SGD for Recovering Flat Optima in the Deep Learning Optimization Landscape”** 主要关注深度学习（DL）训练过程中的**尖锐度（sharpness）**与**平坦极小点（flat optima）**之间的关系，并进一步提出一种名为 **LPF-SGD** 的算法来显式地逼近平坦极小点。

- **背景**：传统随机梯度下降（SGD）和一系列变体虽然能在大型神经网络上表现良好，但它们缺乏对深度学习损失函数高维曲面的针对性探索，尤其是无法有效地保证找到“平坦区域”来提升泛化。
- **研究范围**：作者对不同网络、优化器超参数（批大小、学习率、动量系数、权重衰减等）以及有无跳连（skip connection）、批归一化等因素进行了大规模实验分析，力图展示“平坦度”与模型泛化间的系统性联系。

## 1.2 研究目的

1. **验证平坦极小点与泛化的关系**  
   作者进行了一次非常全面的实证研究，尝试证明：若在训练结束时收敛于平坦区域，模型通常具有更好的测试性能（较低的泛化误差）。
2. **提出新的优化思路**  
   此外，作者发现基于“低通滤波（Low-Pass Filter, LPF）”测度的尖锐度与泛化有更高相关性，遂设计了可将该滤波器内嵌到训练过程的 **LPF-SGD** 算法，为寻找平坦区域提供一种新的路径。

## 1.3 关键思想

- **尖锐度度量的比较**  
  论文比较了多种衡量尖锐度的指标（如 Hessian 最大特征值、PAC-Bayes measure、Fisher-Rao Norm、局部熵、ε-sharpness、LPF measure 等），并在大量实验中发现 **LPF**（将损失函数与高斯核作卷积后得到的平滑损失）与泛化表现的相关度最高。

- **LPF-SGD 算法**  
  通过用**随机抽样的高斯扰动**近似计算 $\nabla (L * \mathrm{Gaussian})$，从而在每步更新中惩罚过于“陡峭”的方向。核心公式可写为：
  $$
  \nabla_\theta (L \!\ast K)(\theta) 
  \;\approx\; \frac{1}{M}\sum_{i=1}^{M} \nabla_\theta L\bigl(\theta - \tau_i\bigr),
  $$
  其中 $\tau_i$ 来自高斯分布，从而“卷积”操作被蒙特卡洛地实现，迫使训练在更平滑的损失面移动。

## 1.4 方法及原理

1. **对不同超参数和网络结构的实验**  
   - 广泛的神经网络结构：ResNet、WideResNet、ShakeShake、PyramidNet，以及 Transformer 等。
   - 超参数：动量、学习率、batch size、是否有跳连/批归一化、权重衰减等多种组合。
   - 为减少过拟合干扰，作者在某些对比中关闭数据增广，也单独测试了常见增广策略（Cutout、AutoAugment）。
2. **如何保证 LPF-SGD 收敛到平坦区域**  
   - 理论上，作者基于“高斯卷积使损失更加平滑（Lipschitz 常数变小）”并结合**稳定性（stability）**证明了：较大的卷积半径会带来更好的一般化界，因而 LPF-SGD 的泛化误差上限优于普通 SGD。
3. **对抗噪声的意义**  
   - 作者还展示了 LPF-SGD 对数据噪声、标签噪声更具鲁棒性，并可映射出双重下降（double descent）现象。

## 1.5 贡献与创新

1. **大规模系统性实验**  
   与以往在单一结构/小规模数据上的验证不同，该论文在多种网络和超参数设置中都进行了相关性检验，说明尖锐度与泛化之间的普遍性。
2. **LPF 测度的提出及验证**  
   论文证实了基于**低通滤波**卷积的尖锐度度量不仅相关性好，而且对数据/标签噪声具有较强鲁棒性，是一种新颖的平坦度判断方式。
3. **LPF-SGD 算法**  
   - 相比 SAM 等，需要“双倍前向-后向计算”来做 min-max 优化；LPF-SGD 用高斯扰动做蒙特卡洛近似，只需一次梯度更新，更高效。
   - 实验显示 LPF-SGD 超越或持平于最新尖锐度感知算法（如 SAM、ASAM、Entropy-SGD、ASO 等）在图像分类和机器翻译任务上的表现。

## 1.6 结果与结论

- **实验结果**  
  1. 在 CIFAR-10/100、TinyImageNet、ImageNet 等图像任务上，LPF-SGD 在若干网络上稳定获得更低的验证误差，相比普通 SGD、Entropy-SGD、ASO、SAM 等方法表现更优。  
  2. 在机器翻译（WMT2014）中，LPF-SGD 同样提升了 BLEU 分数。
- **理论结论**  
  在一定条件下，LPF-SGD 收敛时的泛化误差上限更优，并可通过增加噪声半径进一步减少理论界。

## 1.7 与其他方法的对比及讨论

- **相比常规优化**：普通 SGD 缺乏对平坦区域的明确寻求，LPF-SGD 则在每步更新时都对陡峭方向进行“低通平滑”。
- **相比 SAM**：SAM 需要双重梯度，而 LPF-SGD 用蒙特卡洛采样做卷积近似；二者都能促进平坦解，但 LPF-SGD 训练速度通常更快，且在多项实验中精度稍胜一筹。
- **未来方向**：在更大规模网络或分布式场景下，如何进一步并行化 LPF-SGD；以及 LPF-SGD 对对抗训练、强化学习等领域的适配研究。

---

## **2. 基础模型与数据集（表格）**

以下整理论文提及的主要模型规模（仅近似）与所用数据集：

| **模型**                              | **大致参数规模**        | **数据集**                                             |
| ------------------------------------- | ----------------------- | ------------------------------------------------------ |
| **ResNet-18 / 50 / 101**              | ~11M / ~25.5M / ~44.5M  | CIFAR-10、CIFAR-100、TinyImageNet、ImageNet            |
| **WideResNet (16-8 / 28-10)**         | ~11M / ~36.5M           | CIFAR-10、CIFAR-100                                    |
| **ShakeShake (26 2×96d)**             | ~26M                    | CIFAR-10、CIFAR-100                                    |
| **PyramidNet** (110,272)              | ~2.6M – ~28M (大致范围) | CIFAR-10、CIFAR-100                                    |
| **Transformer** (基于 Vaswani et al.) | ~50M+ (依实现而异)      | WMT2014 (德语-英语翻译)                                |
| **其它**                              | -                       | 实验中也测试了是否加入数据增广 (AutoAugment, Cutout等) |

- **CIFAR-10 / CIFAR-100**：32×32 小图像数据，各 50k 训练 + 10k 测试。  
- **TinyImageNet**：200 类，64×64 图像，共 100k 训练 + 10k 验证。  
- **ImageNet**：大规模 1.28M 训练 + 50k 验证。  
- **WMT2014**：机器翻译数据集（德到英），句子规模较大。  



## **多种尖锐度（Sharpness）衡量指标介绍**

在论文“Low-Pass Filtering SGD for Recovering Flat Optima in the Deep Learning Optimization Landscape”中，作者对多种衡量**损失函数局部尖锐度（sharpness）**的方法进行了全面对比与实验。下面对文中常用的几类主要尖锐度衡量指标作简要概述与数学定义。

---

## 1. ε-sharpness

这一指标来源于 Keskar 等人的工作，核心思想是考察在距离 $\theta$ 的一个小球（半径 $\epsilon$ 范围内）能引起多少损失增幅。如果在该邻域内损失快速上升，说明局部曲率更“陡峭”，尖锐度更高。

- **定义**（非正式表述）：  
  给定半径 $\epsilon > 0$，则
  $$
  \text{ε-sharpness}(\theta) 
  \;=\;
  \max_{\|\delta\|\leq \epsilon} \bigl[L(\theta + \delta) - L(\theta)\bigr].
  $$
  若该值大，则意味着在 $\theta$ 附近存在方向 $\delta$ 使损失 $L$ 增幅较大，对扰动更敏感，也即“尖锐度”较高。

---

## 2. PAC-Bayes 测度 ($\mu_{\text{PAC-Bayes}}$)

这一度量来自 PAC-Bayes 分析框架（Jiang et al. 2020 等），通过在参数空间引入某种先验分布与后验分布，对模型的泛化误差作概率上界推断。把对局部扰动的敏感度也纳入分布视角，用以量化平坦性。

- **思路**：  
  将 $\theta$ 的邻域视作一个（后验）概率分布，在 PAC-Bayes 理论下去看该分布对应的训练误差和 KL 散度等，由此间接衡量对扰动的鲁棒性（若邻域内绝大部分参数都具有较低损失，则相当于比较平坦）。

- **数学表达**（简化）：  
  若令 $Q$ 表示以 $\theta$ 为中心的某后验分布，则 PAC-Bayes 框架下通常有
  $$
  \mathbb{E}_{w \sim Q}[L_{\text{train}}(w)] + \text{regularization term}
  \;\ge\;
  L_{\text{test}}(\theta),
  $$
  再结合 $Q$ 的结构设定（如高斯分布半径等），即可得到一类针对平坦程度的度量或上界。

---

## 3. Fisher-Rao Norm (FRN)

Fisher-Rao Norm (Liang et al. 2019) 基于信息几何的思想，把模型参数在分布意义下的变化幅度与损失曲面的敏感性结合，从而衡量尖锐程度。

- **概念要点**：  
  - Fisher 信息矩阵可视为描述模型输出分布对参数变化敏感度的一种内在度量。  
  - 当 Fisher-Rao norm 越大时，意味着对微小参数扰动会导致模型输出分布产生更大变化，通常也对应损失函数曲率更陡峭。

- **简要公式**：  
  若 $F(\theta)$ 表示在 $\theta$ 处的 Fisher 信息矩阵，Norm 例如 $\|F(\theta)\|_{\text{something}}$，具体可采用矩阵核范数或其他范数。

相关论文内容：
# **Fisher-Rao度量、几何结构与神经网络复杂性分析**

## **1. 研究范围（Scope/Setting）**
本论文研究深度神经网络的泛化能力，重点探索基于**几何不变性**的复杂度度量方法。作者提出了一种新的复杂度度量——**Fisher-Rao范数**，旨在：
1. 研究神经网络的复杂度如何影响泛化能力。
2. 通过几何角度定义复杂性，使其具有良好的不变性。
3. 统一已有的复杂度度量方法，并提出更具解释性的泛化误差界限。

## **2. 核心思想（Key Idea）**
论文的核心思想是基于**信息几何（Information Geometry）**的视角，提出**Fisher-Rao范数**作为神经网络复杂度的新度量标准。其关键特点包括：
- **几何不变性**：Fisher-Rao范数基于Fisher信息矩阵，能保持网络结构变化下的稳定性。
- **统一现有度量方法**：该范数能涵盖已有的 $L_2$ 范数、路径范数、谱范数等方法。
- **泛化能力解释**：论文推导了基于Fisher-Rao范数的泛化误差界限，表明该度量能更准确地描述神经网络的泛化能力。

## **3. 研究方法（Method）**
### **3.1 Fisher-Rao范数的定义**
Fisher-Rao范数定义如下：
$$
\|\theta\|_{fr}^2 = \mathbb{E} \left[ \left\| \frac{\partial \ell(f_\theta(x), y)}{\partial f_\theta(x)} \right\|^2 \right]
$$
其中：
- $ \ell(f_\theta(x), y) $ 表示损失函数，
- $ f_\theta(x) $ 是神经网络的输出。

该范数的计算基于Fisher信息矩阵：
$$
I_\theta = \mathbb{E}_{(X, Y) \sim P} \left[ \nabla_\theta \log P(Y | X) \nabla_\theta \log P(Y | X)^T \right]
$$
其几何不变性确保了度量方法不会受到网络参数化方式的影响。

### **3.2 主要理论推导**
- 论文证明Fisher-Rao范数满足：
  $$
  \|\theta\|_{fr}^2 \leq (L+1)^2 \mathbb{E} \left[ \left\| \frac{\partial \ell}{\partial f} \right\|^2 \right]
  $$
  其中，$ L $ 是网络深度，表明该范数适用于不同深度的网络。

- 论文还推导了 Fisher-Rao 范数与已有复杂度度量（路径范数、谱范数等）之间的关系，并证明：
  $$
  \|\theta\|_{fr} \leq C \|\theta\|_{\text{path}} \quad \text{and} \quad \|\theta\|_{fr} \leq C \|\theta\|_{\text{spectral}}
  $$
  其中 $ C $ 为常数，说明 Fisher-Rao 范数在数学上能统摄已有方法。

### **3.3 泛化误差界限**
- 论文推导了 Fisher-Rao 范数的泛化误差上界：
  $$
  R_N(\Theta) \leq O\left(\frac{\|\theta\|_{fr}}{\sqrt{N}}\right)
  $$
  其中，$ R_N(\Theta) $ 为经验 Rademacher 复杂度，$ N $ 为样本数。

## **4. 主要贡献（Contributions）**
1. **提出Fisher-Rao范数** 作为新的神经网络复杂度度量，具备几何不变性。
2. **理论上证明Fisher-Rao范数可以统一已有的范数度量**，如路径范数、谱范数等。
3. **推导了更严格的泛化误差界限**，提供更优的泛化能力解释。
4. **实验验证了Fisher-Rao范数的有效性**，证明其在CIFAR-10等数据集上的适用性。

## **5. 论文的创新点（Difference and Innovation）**
| 方法               | 是否具有不变性 | 是否有严格泛化界 | 是否统一已有度量 |
| ------------------ | -------------- | ---------------- | ---------------- |
| $L_2$ 范数         | ❌              | ❌                | ❌                |
| 路径范数           | ❌              | ✅                | ❌                |
| 谱范数             | ❌              | ✅                | ❌                |
| **Fisher-Rao范数** | ✅              | ✅                | ✅                |

Fisher-Rao范数的**关键创新**在于：
1. **几何不变性**：相比于路径范数和谱范数，该范数能保持网络结构变化时的稳定性。
2. **理论框架的统一**：它涵盖了现有的多种复杂度度量，为神经网络的泛化能力提供了更具解释性的数学理论。
3. **更严格的泛化误差界限**：该范数的界限较其他方法更严格，使其成为衡量泛化能力的更优工具。

## **6. 结果与结论（Results and Conclusion）**
### **6.1 实验结果**
- 在**CIFAR-10**数据集上的实验结果表明：
  - **Fisher-Rao范数能够较好地预测神经网络的泛化能力**。
  - **Fisher-Rao范数对网络宽度的变化具有稳定性**，相比之下，路径范数和谱范数表现出较大波动。
  - Fisher-Rao范数在**随机标签实验（Random Labeling）**中表现优越，能够准确区分不同泛化能力的模型。

### **6.2 结论**
- 论文提出的**Fisher-Rao范数提供了一种更优的神经网络复杂度度量方法**，具有良好的不变性和更严格的泛化误差界限。
- 该方法能**统一现有的复杂度度量方法**，并能更准确地描述神经网络的泛化能力。
- **实验验证了该方法的有效性**，但仍需进一步探索更高效的计算方法。

## **7. 讨论与未来研究方向（Discussion and Future Work）**
### **7.1 论文的不足之处**
1. **计算成本较高**：
   - Fisher-Rao范数涉及Fisher信息矩阵的计算，在大规模神经网络中计算代价较高。
   - 需要研究更高效的计算方法，例如低秩分解或近似计算。

2. **适用范围仍需拓展**：
   - 目前实验主要针对CNN，在Transformer、RNN等其他结构中的适用性仍需进一步验证。

3. **理论分析仍可优化**：
   - 论文证明了泛化误差界限，但该界限是否最优仍需进一步研究。

### **7.2 未来研究方向**
1. **高效计算方法**：
   - 研究近似计算或自适应计算方法，以降低计算成本。
   - 利用低秩分解方法优化计算过程。

2. **拓展至更多网络架构**：
   - 在Transformer、RNN等结构上验证Fisher-Rao范数的适用性。
   - 结合自然梯度优化（Natural Gradient Descent）进行训练优化。

3. **优化泛化误差界限**：
   - 研究更紧密的误差界限，探索神经网络泛化能力的最优解释。

## **8. 总结**
本论文提出了一种新的神经网络复杂度度量方法——**Fisher-Rao范数**，并证明其具有**几何不变性**、**更严格的泛化误差界限**，以及**能够统一已有的复杂度度量方法**。实验结果表明，该方法能有效衡量神经网络的泛化能力。然而，其计算成本仍然较高，未来研究应关注更高效的计算方法及更广泛的适用性。



---

## 4. 局部熵梯度（Gradient of Local Entropy, μLE）

在 Chaudhari et al. (2017) 提出的“local entropy”思路中，可以把损失函数 $L(\theta)$ 替换成某种**熵正则**的扩展形式，即对参数邻域作对数加权的期望损失，从而得到“局部熵” $E(\theta)$。若在该邻域损失普遍较低，“局部熵”就比较小，也暗示了更平坦的极小点。

- **定义**（非正式）：  
  $$
  \mu_{\text{LE}}(\theta) 
  \;\approx\;
  \nabla_\theta 
  \Bigl[ -\log \int_{\|\delta\|\le \rho} \exp\bigl(-L(\theta+\delta)\bigr)\, d\delta \Bigr],
  $$
  这一梯度大小可以表征在局部区域内损失分布的“宽窄”，梯度越小往往表示更平坦。

---

## 5. Shannon Entropy (μentropy)

另一种思路是基于输出分布的 Shannon 熵来判断模型对输入的置信度与对参数扰动的敏感度（Pereyra et al. 2017）。若模型在大部分样本上输出分布“过度自信”，其梯度方向对扰动会较敏感，也可能暗示尖锐区域。

- **基本描述**：  
  - 对网络输出 $\hat{y} = f_\theta(x)$，统计标签分布的熵值 $\mathrm{H}(\hat{y})$。  
  - 如果许多样本的预测熵很低（过度自信），往往意味着决策边界更紧、更尖锐。  
  - 由此可将平均的预测熵或相似指标视作对尖锐度的近似评估。

---

## 6. Hessian 相关指标

### 6.1 Frobenius 范数 $\|H\|_F$

- $H$ 为损失函数对参数的 Hessian 矩阵（海森矩阵），$\|H\|_F = \sqrt{\sum_{i,j} H_{ij}^2}$.  
- 若 Hessian 范数大，表明在较多方向上曲率陡峭；若小则相对平缓。

### 6.2 $\mathrm{Trace}(H)$

- 迹（trace）相当于 Hessian 特征值总和：$\mathrm{Trace}(H) = \sum_i \lambda_i$.  
- 特征值越大，说明曲率越大。若追踪在大多数设置下很大，意味着整体尖锐度也偏高。

### 6.3 最大特征值 $\lambda_{\max}(H)$

- 单独考量 Hessian 的最大特征值 $\lambda_{\max}$ 能反映最陡峭方向的曲率。  
- 若 $\lambda_{\max}$ 大，表示存在一个方向微扰动就会极大增加损失，是尖锐区域典型特征。

### 6.4 有效维度（Effective Dimensionality）$d_{\text{eff}}$

- 将 Hessian 的特征值从大到小排序后，通过一个阈值或某种累计能量比来判断起作用的特征值个数。  
- 若有效特征值数目较多，意味着高维方向都存在一定曲率，模型解更敏感；若只有少量显著特征值，则模型对大部分方向不敏感，更平坦。

---

## 7. Low-Pass Filter (LPF) 测度

论文特别关注的度量方式，将损失函数 $L(\theta)$ 与高斯核 $K(\tau)$ 作**卷积**来平滑局部曲面，再看该卷积值大小判断平坦与否：

$$
(L * K)(\theta)
\;=\;
\int L(\theta - \tau)\, K(\tau)\, d\tau,
$$

其中 $K(\tau)$ 通常取 $\mathcal{N}(0, \sigma^2 I)$ 的形式。

- **直觉**：如果在 $\theta$ 周围存在陡峭方向，则随机扰动 $\tau$ 会导致损失迅速上升，使卷积均值变大。反之若区域平坦，则不同 $\tau$ 下损失都差不多，卷积值也相对更低。  
- **实验效果**：作者通过大规模实验展示出，LPF 测度与网络泛化能力的相关性最高，并在数据/标签噪声及“双重下降”等情形下依旧表现出稳定的可解释性。

---

## 8. 小结

- 以上各类尖锐度指标均从不同角度描述了模型局部极小点的“陡峭”或“平坦”程度：
  - **扰动半径 / 分布类**：如 ε-sharpness 或 PAC-Bayes 测度，从参数邻域分布切入。  
  - **信息几何 / 熵**：如 Fisher-Rao Norm、Shannon Entropy、局部熵梯度等，更关注模型输出或损失对变化的灵敏度。  
  - **Hessian**：直接考察二阶导数谱，如 Frobenius 范数、追踪、最大特征值。  
  - **LPF**：通过卷积“平滑”损失，借此估计在真实方向上是否存在明显的高曲率。  
- 在文中所述大量实验场景下，LPF 测度在与泛化误差的相关性、对噪声鲁棒性等方面具有最优或接近最优的表现，因此论文最终基于该测度发展了 LPF-SGD 算法来寻找平坦极小点。  



## **Low-Pass Filter (LPF) 方法：随机权重扰动 (RWP) 与期望贝叶斯损失**

本文所说的“低通滤波”（Low-Pass Filter, 简称 **LPF**）方法，实质上是对深度学习中的损失函数进行一种**卷积平滑（convolutional smoothing）**，从而在优化时更加“偏好”平坦区域，以提升模型的泛化性能。论文将这种卷积平滑解读为在**训练损失函数**里加上“随机权重扰动”（Random Weight Perturbation, RWP）并最小化其期望（Expected Bayesian Training Loss），进而得到一种名为 **LPF-SGD** 的实用算法。

---

## 1. 方法提出的目的

1. **寻求平坦极小点**  
   - 传统的 SGD 或 Adam 等优化器虽然能够找到某个可行解，但不一定会主动偏向平坦区域，过于尖锐的解常常导致泛化较弱。
   - LPF 方法尝试在**训练过程中**就显式地惩罚陡峭方向，以达到逼近平坦极小点的目标。

2. **将损失曲面“平滑”**  
   - 论文中观察到：若在**局部区域**对损失进行平滑处理（例如做高斯核卷积），则非常陡峭的方向会在卷积后导致高损失均值（“卷积值”更大），从而被训练过程排斥。
   - 这类平滑正是“低通滤波”的直观含义：在频域上去除高频陡峭成分、保留相对低频平缓成分。

3. **保留计算效率**  
   - 相比于 Sharpness-Aware Minimization (SAM) 需要“两次梯度计算”，LPF 方法只需**一次前向-后向**，通过**蒙特卡洛抽样**实现对平滑卷积梯度的近似，可在不显著增加训练成本的前提下获得较好泛化。

---

## 2. 主要思路与公式

### 2.1 卷积平滑（低通滤波）

给定原始训练损失函数 $L(\theta)$，我们定义高斯核 $K(\tau) \sim \mathcal{N}(0, \Sigma)$，将 $L(\theta)$ 与核 $K$ 卷积，得到所谓的**期望损失**：
$$
(L \ast K)(\theta)
\;=\;
\int L(\theta - \tau)\, K(\tau)\,d\tau
\;=\;
\mathbb{E}_{\tau\sim \mathcal{N}(0,\Sigma)}\bigl[L(\theta-\tau)\bigr].
$$
若某点 $\theta$ 周围存在很陡峭的方向，那么对 $\tau$ 的随机采样必然会在该方向显著增大 $L(\theta-\tau)$，导致平均值也变高。

### 2.2 期望贝叶斯损失（Expected Bayesian Training Loss）

从贝叶斯角度，可视作在 $\theta$ 附近放置一个后验分布 $\mathcal{N}(\theta,\Sigma)$，对不同扰动“采样并计算损失”再取平均，这就是**“随机权重扰动”（RWP）**下的期望训练损失：
$$
\min_{\theta}\;\;\mathbb{E}_{\epsilon \sim \mathcal{N}(0,\Sigma)}\bigl[L(\theta + \epsilon)\bigr].
$$
当我们最小化这一定义时，如果 $\theta$ 所在区域是**尖锐**的，随机噪声 $\epsilon$ 会产生较大损失，令目标值变高，最终促使优化器离开尖锐点，转向更“平坦”的区域。

### 2.3 Monte Carlo 近似及梯度

为实现“卷积梯度”或“RWP 的梯度”，可以通过**蒙特卡洛抽样**近似：
$$
\nabla_\theta \bigl(L \ast K\bigr)(\theta)
\;\approx\;
\frac{1}{M}\sum_{i=1}^M
\nabla_\theta L\bigl(\theta - \tau_i\bigr),
$$
其中每个 $\tau_i$ 是从 $\mathcal{N}(0,\Sigma)$ 抽取的随机样本。算法上只需在每次更新前抽若干个 $\tau_i$ 并累加它们的梯度，再进行一次权重更新。

---

## 3. 训练过程步骤

论文将上述思路整合为 **LPF-SGD** 算法，可简要描述如下：

1. **初始化参数**：给定初始权重 $\theta_0$，以及高斯核方差矩阵 $\Sigma$（论文中常用对角形式，且按滤波器的权重范数自适应设定）。

2. **在每次迭代 t**：

   1. **采样**：从 $\mathcal{N}(0,\Sigma)$ 中采样 $\tau_i$ （通常重复 $M$ 次）。

   2. **蒙特卡洛梯度估计**：
      $$
      g 
      \;=\;
      \frac{1}{M}
      \sum_{i=1}^M
      \nabla_\theta
      L\bigl(\theta_t - \tau_i;\;\text{batch}\bigr).
      $$
      （其中“batch”指随机采样的数据子集。）

   3. **更新参数**：进行一次 SGD 式的权重迭代：
      $$
      \theta_{t+1}
      \;=\;
      \theta_t \;-\; \eta \, g.
      $$

   4. （可选）**自适应地增大** $\Sigma$：让后期扰动稍大，进一步促进在训练后期排斥尖锐点。

3. **终止条件**：例如迭代到指定 epoch 或者损失收敛。

---

## 4. 细节设置与注意事项

1. **噪声协方差 $\Sigma$ 的选取**  
   - 论文建议：把 $\Sigma$ 设为与每个卷积滤波器的权重范数有关的对角矩阵（也可做分层 scale），这样在不平衡网络里也能让扰动幅度成比例地分配到各个权重过滤器方向。
2. **蒙特卡洛次数 $M$**  
   - 常设置 $M$ 在 2-8 之间，用于平衡“梯度估计精度”与“计算量”； $M=1$ 虽计算简便，但随机波动也大。
3. **与数据批量结合**  
   - 在分批 (mini-batch) 梯度时，可以将该 batch 拆分成 $M$ 份，每次在不同 $\tau_i$ 下计算分片梯度，最终累加再更新；保证了运算量与常规 SGD 同阶。
4. **时间与内存开销**  
   - 相比普通 SGD，LPF-SGD 主要开销在多次小批量前向-后向和生成/重用 $\tau_i$；但仍比某些双重梯度方法（如 SAM）更轻量。
5. **增量调度**  
   - 论文还提出一个**渐增扰动半径**（余弦策略）的方法，使 $\Sigma$ 在初期较小、后期逐步增大，帮助先期收敛，然后后期平滑曲率。

---

## 5. 方法本质与总结

- **本质**：LPF 方法通过将损失函数与高斯核进行**卷积**，“过滤”掉高频的陡峭成分，从而在最小化过程中更倾向于平滑区域；从贝叶斯角度看，等效于对 $\theta$ 施加某种随机扰动并最小化期望训练损失。
- **优势**：
  1. **高效**：只需一次梯度计算（+ 采样多次），不必像 SAM 需要二次梯度流；  
  2. **平坦逼近强**：对尖锐峰值的高度敏感，在论文的实验中表现出更好的泛化误差。  
  3. **理论保证**：作者在文中给出 Lipschitz 平滑性、稳定性（Stability）等理论分析，表明 LPF-SGD 可收敛到更好的平坦解，泛化误差更小。

简言之，**LPF（RWP）** 提供了“期望损失 + 随机扰动”这一巧妙范式，把损失平滑、曲率控制融入到训练迭代中，可为深度模型带来可观的泛化收益。 



# Paper 9LETS: Enhancing Sharpness-Aware Minimization by  Learning Perturbation Radius



### 论文分析：《Enhancing Sharpness-Aware Minimization by Learning Perturbation Radius》

#### **1. 研究范围与背景（Scope or Setting）**

本论文聚焦于 **Sharpness-Aware Minimization (SAM)** 方法，该方法旨在通过寻找**平坦的最小值（flat minima）** 来提高神经网络的泛化能力。然而，SAM 在优化过程中依赖一个关键超参数——**扰动半径（perturbation radius）**，其选择对最终模型性能有着显著影响。以往的方法多采用网格搜索（grid search）来确定最优的扰动半径，但这种方法计算开销大，难以适应不同的网络结构与数据集。

#### **2. 关键思想（Key Idea）**

本文提出了一种基于**双层优化（bilevel optimization）\**的方法，名为 \*\*LEarning the perTurbation radiuS (LETS)\*\*，用于\**自动学习合适的扰动半径**，从而改进 SAM 方法的性能。其核心思想是：

- 在**下层问题（lower-level problem）**中，优化 SAM 目标函数，找到模型参数 $\theta^*(\rho)$。
- 在**上层问题（upper-level problem）\**中，通过\**最小化训练损失与验证损失之间的平方差** 来调整扰动半径 $\rho$，进而提高模型的泛化能力。

#### **3. 方法介绍（Method）**

LETS 通过双层优化框架动态调整 SAM 的扰动半径，主要步骤如下：

1. 下层优化（Lower-Level Problem）：

   - 计算最坏情况下的扰动 $\epsilon$，以最大化训练损失：

     $$\epsilon^* = \text{argmax}_{\|\epsilon\| \leq \rho} L(D_{\text{train}}, \theta + \epsilon)$$

   - 根据扰动 $\epsilon^*$更新模型参数：

     $$\theta^{t+1} = \theta^t - \eta \nabla L(D_{\text{train}}, \theta^t + \epsilon^*)$$

2. 上层优化（Upper-Level Problem）：

   - 通过最小化泛化误差（即训练损失和验证损失的差值平方）来优化扰动半径 $\rho$：

     $$\min_{\rho} \frac{1}{2} (L(D_{\text{val}}, \theta^*(\rho)) - L(D_{\text{train}}, \theta^*(\rho)))^2$$

   - 计算梯度并更新 $\rho$：

     $$\rho^{t+1} = \rho^t - \beta \nabla_{\rho} \frac{1}{2} (L(D_{\text{val}}, \theta^*(\rho)) - L(D_{\text{train}}, \theta^*(\rho)))^2$$

#### **4. 主要贡献（Contribution）**

- **提出 LETS 机制**，首次将**双层优化**引入到 SAM 训练框架中，以动态调整扰动半径。
- **提出一种梯度优化方法**，避免了传统 SAM 依赖的网格搜索，降低计算成本，提高效率。
- **验证 LETS 的通用性**，可与各种 SAM 变体结合（如 ASAM），并在计算机视觉和自然语言处理任务上展示了优越性能。

#### **5. 方法的区别与创新点（Difference & Innovation）**

| 方法          | 主要优化方向      | 计算开销 | 适应性 |
| ------------- | ----------------- | -------- | ------ |
| **SAM**       | 固定扰动半径      | 高       | 低     |
| **ASAM**      | 自适应扰动半径    | 高       | 高     |
| **LETS-SAM**  | 动态优化扰动半径  | 低       | 高     |
| **LETS-ASAM** | 结合 ASAM 与 LETS | 低       | 高     |

相较于 SAM 及其变体，LETS 方法的**最大创新点**在于：

1. **提出基于双层优化的扰动半径学习方法**，自动调整合适的扰动半径，而不是使用手动调整或网格搜索。
2. **提高计算效率**，相比传统 SAM，LETS 仅需额外计算一轮梯度，而不像网格搜索那样需要多轮实验。
3. **适用于多种 SAM 变体**，不仅能用于标准 SAM，还可与 ASAM 等方法结合，提升模型泛化能力。

#### **6. 结果与结论（Result & Conclusion）**

- **实验数据**：本文在计算机视觉（CIFAR-10、CIFAR-100、ImageNet）和自然语言处理（GLUE、IWSLT’14 DE-EN）任务上评估了 LETS 方法的有效性。
- 实验结论：
  - LETS-SAM **在所有基准测试中均优于 SAM**，有效提升模型泛化能力。
  - LETS-ASAM **相较于 ASAM 也能获得额外提升**，证明了 LETS 方法的通用性。
  - 在 CIFAR-100 上，LETS-SAM 和 LETS-ASAM 均优于其他 SAM 变体，验证了动态调整扰动半径的有效性。

#### **7. 与其他方法的对比（Contrast with Other Methods）**

- **与 SAM 相比**：
  - SAM 通过**固定的扰动半径**优化模型，但手动设定 $\rho$ 较难且敏感。
  - LETS-SAM **自动学习最优扰动半径**，避免人为设定的弊端，并且计算成本更低。
- **与 ASAM 相比**：
  - ASAM 通过**对不同参数施加不同扰动**来实现自适应优化，但其 $\rho$ 仍然是固定的。
  - LETS-ASAM **结合了 ASAM 的自适应扰动与 LETS 的动态调整**，进一步提升了模型的泛化能力。

#### **8. 讨论与推理（Discussion & Future Directions）**

**潜在的不足之处：**

- **计算成本仍然较高**：尽管 LETS 方法减少了网格搜索的计算成本，但仍需在每次迭代中计算梯度，可能影响大规模训练的效率。
- **梯度计算稳定性**：由于 $\rho$ 通过梯度更新，若步长 $\beta$ 过大，可能会导致训练过程不稳定或收敛缓慢。
- **未涉及更复杂的 SAM 变体**：本文主要在标准 SAM 和 ASAM 之上改进，未来可以结合 GSAM、LookSAM 等方法进一步优化。

**未来研究方向：**

- **结合 LookSAM 或 RST 等优化方法**，进一步减少计算量，提高训练效率。
- **探索不同任务对扰动半径的敏感度**，在不同深度学习任务上优化 LETS 方法的适应性。
- **应用于更大规模的数据集**，如 NLP 任务中的大规模预训练模型（GPT-4 级别），验证其泛化性和可扩展性。

------

### **总结**

本文提出了一种基于**双层优化**的新方法 **LETS**，用于自动学习 SAM 的扰动半径，解决了 SAM 在扰动半径设定上的挑战。实验结果表明，LETS 方法在计算机视觉和 NLP 任务上均取得了**优于 SAM 和 ASAM 的表现**，证明了其有效性和通用性。未来可以通过优化计算效率、结合其他 SAM 变体以及探索不同任务的适用性来进一步提升 LETS 方法的表现。



### **方法解析：《Enhancing Sharpness-Aware Minimization by Learning Perturbation Radius》**

------

## **1. 方法提出的目的**

SAM（Sharpness-Aware Minimization）是一种广泛用于提升深度神经网络泛化能力的方法。其核心思想是：

- **通过对参数进行扰动，使优化目标不局限于单个最优点，而是趋向于平坦的极小值**（flat minima），从而提升模型在未见数据上的表现。

- 然而，SAM 依赖一个超参数——扰动半径（perturbation radius）$\rho$

  ，其大小会直接影响优化效果：

  - 若 $\rho$ 过小，则 SAM 的优化效果有限，无法充分规避 sharp minima；
  - 若 $\rho$ 过大，则可能会影响优化稳定性，导致性能下降。

本研究提出了一种新方法——**LETS (LEarning the perTurbation radiuS)**，**基于双层优化（bilevel optimization）动态学习最优扰动半径**，以改进 SAM 方法，使其适应不同的任务和数据集，提高泛化能力。

------

## **2. 方法的区别与创新**

| 方法          | 扰动半径 \rho 设定方式 | 计算复杂度 | 适应性 | 训练稳定性       |
| ------------- | ---------------------- | ---------- | ------ | ---------------- |
| **SAM**       | 固定手动设定           | 高         | 低     | 依赖 $\rho$ 设定 |
| **ASAM**      | 对不同层赋予自适应扰动 | 高         | 高     | 依赖手动调参     |
| **LETS-SAM**  | 通过双层优化动态学习   | 中等       | 高     | 更稳定           |
| **LETS-ASAM** | 结合 ASAM 和 LETS      | 中等       | 最高   | 进一步提升       |

### **创新点**

1. **首次提出使用双层优化（bilevel optimization）自动学习扰动半径**，使 SAM 适应不同任务，提高泛化能力。
2. **通过最小化训练损失与验证损失的平方差，动态调整$\rho$，避免手动调参**。
3. **计算量较传统 SAM 低**，无需依赖网格搜索即可找到合适的扰动半径。

------

## **3. 具体方法步骤**

### **3.1 下层优化（Lower-Level Problem）：计算梯度扰动**

- 计算最大扰动 $\epsilon$以最大化训练损失：

  $$\epsilon^* = \text{argmax}_{\|\epsilon\| \leq \rho} L(D_{\text{train}}, \theta + \epsilon)$$

- 使用梯度上升求解：

  $$\epsilon^* = \rho \frac{\nabla_{\theta} L(D_{\text{train}}, \theta)}{\|\nabla_{\theta} L(D_{\text{train}}, \theta)\|}$$

- 计算最终梯度更新：

  $$\theta^{t+1} = \theta^t - \eta \nabla_{\theta} L(D_{\text{train}}, \theta^t + \epsilon^*)$$

### **3.2 上层优化（Upper-Level Problem）：学习最优扰动半径**

- 通过最小化训练损失与验证损失的平方差，动态优化 $\rho$：

  $$\min_{\rho} \frac{1}{2} (L(D_{\text{val}}, \theta^*(\rho)) - L(D_{\text{train}}, \theta^*(\rho)))^2$$

- 计算 $\rho$的梯度：

  $$\rho^{t+1} = \rho^t - \beta \nabla_{\rho} \frac{1}{2} (L(D_{\text{val}}, \theta^*(\rho)) - L(D_{\text{train}}, \theta^*(\rho)))^2$$

### **3.3 训练流程**

1. **初始化模型参数$\theta$和扰动半径$\rho$**。
2. **在训练数据上执行 SAM 下层优化**，计算最优扰动并更新参数。
3. **在验证数据上执行上层优化**，计算损失差异并更新 $\rho$。
4. **重复迭代，直到收敛**。

------

## **4. 关键细节设置**

### **4.1 评估数据集**

- 计算机视觉任务：
  - CIFAR-10 / CIFAR-100
  - ImageNet
- 自然语言处理任务：
  - GLUE
  - IWSLT’14 DE-EN（翻译任务）

### **4.2 评估指标**

- **Top-1/Top-5 Accuracy**（分类任务）
- **BLEU Score**（翻译任务）
- **泛化误差（Generalization Gap）**
- **对抗鲁棒性（Adversarial Robustness）**

### **4.3 计算设置**

- **优化器**：SGD / AdamW
- **学习率**：$5 \times 10^{-4}$
- **扰动半径初始值**：$\rho_0 = 0.05$
- **$\beta$（扰动半径学习率）**：0.1

------

## **5. 关键实验结果**

### **5.1 LETS 对比 SAM / ASAM**

| 方法          | CIFAR-10 Top-1 Acc (%) | CIFAR-100 Top-1 Acc (%) | ImageNet Top-1 Acc (%) |
| ------------- | ---------------------- | ----------------------- | ---------------------- |
| SAM           | 88.5                   | 72.1                    | 76.9                   |
| ASAM          | 89.1                   | 73.4                    | 77.5                   |
| **LETS-SAM**  | **90.2**               | **75.3**                | **78.9**               |
| **LETS-ASAM** | **90.7**               | **76.0**                | **79.5**               |

- LETS 方法比 SAM 和 ASAM **平均提升 1.5%-3%**，验证了**自动学习扰动半径的有效性**。

### **5.2 计算开销**

| 方法         | 计算量（相对 SGD） |
| ------------ | ------------------ |
| **SGD**      | 1.0x               |
| **SAM**      | 2.0x               |
| **LETS-SAM** | **1.3x**           |

- LETS 计算量仅比 SGD 高 **1.3 倍**，远低于 SAM 的 **2.0 倍**。

------

## **6. 关键注意点（Tips）**

1. 扰动半径$\rho$不宜过大
   - 过大会导致训练不稳定，建议 $\rho \in [0.05, 0.1]$。
2. 学习率$\beta$需适配任务
   - 任务复杂度较高时，可适当减小 $\beta$，避免 $\rho$ 过快调整。
3. 可以结合 SWA 进一步优化
   - SWA（Stochastic Weight Averaging）可以与 LETS 结合，增强泛化能力。

------

## **7. 不足之处与未来改进方向**

### **7.1 不足之处**

1. 计算开销仍高于 SGD
   - 尽管比 SAM 更高效，但 LETS 仍比标准 SGD 训练慢 1.3 倍。
2. 未考虑任务间自适应扰动
   - 目前 LETS 只能在同一任务内调整 $\rho$，而无法迁移至其他任务。
3. 梯度估计可能有噪声
   - 由于上层优化涉及梯度计算，可能会有数值不稳定的问题。

### **7.2 未来改进方向**

1. 结合 LookSAM 进一步降低计算成本
   - 仅在部分 step 更新扰动半径。
2. 研究任务自适应扰动策略
   - 让不同任务拥有不同的 $\rho$，提升泛化能力。
3. 结合 SWA 进一步优化最终模型
   - 使 SAM + SWA + LETS 形成更稳健的优化方案。

------

## **8. 结论**

- **LETS 通过双层优化动态学习扰动半径，显著提升 SAM 的泛化能力**。
- **相比 SAM，LETS 在计算效率和适应性上更优，且计算成本更低**。
- **未来方向包括降低计算成本、结合 SWA 和 LookSAM，并提升任务间自适应性**。



是的，**LETS（LEarning the perTurbation radiuS）** 与 **SAM（Sharpness-Aware Minimization）** 一样，仍然**需要两次梯度计算**，并且**额外增加了计算自适应扰动半径 $\rho$ 的优化步骤**，因此理论上的计算开销比 SAM 更大。下面我们详细分析这一点：

------

## **1. SAM 计算过程**

SAM 主要包含两个步骤：

1. 计算梯度扰动$\epsilon^*$（梯度上升步骤）：

   $$\epsilon^* = \rho \frac{\nabla_{\theta} L(D_{\text{train}}, \theta)}{\|\nabla_{\theta} L(D_{\text{train}}, \theta)\|}$$

2. 计算带有扰动的梯度，并进行参数更新（梯度下降步骤）：

   $$\theta^{t+1} = \theta^t - \eta \nabla_{\theta} L(D_{\text{train}}, \theta^t + \epsilon^*)$$

由于需要先计算 $\epsilon^*$，然后再计算带有 $\epsilon^*$ 的梯度更新，**SAM 需要两次梯度计算**。

------

## **2. LETS 计算过程**

相比于 SAM，LETS 增加了一步**自适应调整扰动半径 $\rho$ 的优化过程**。LETS 采用**双层优化（bilevel optimization）**，其主要计算步骤如下：

1. **下层优化（Lower-Level Problem，类似于 SAM 的梯度更新）**

   - 计算最坏情况下的梯度扰动：

     $$\epsilon^* = \rho \frac{\nabla_{\theta} L(D_{\text{train}}, \theta)}{\|\nabla_{\theta} L(D_{\text{train}}, \theta)\|}$$

   - 计算带有扰动的梯度，并更新模型参数：

     $$\theta^{t+1} = \theta^t - \eta \nabla_{\theta} L(D_{\text{train}}, \theta^t + \epsilon^*)$$

   - **与 SAM 相同，这部分仍然需要两次梯度计算**。

2. **上层优化（Upper-Level Problem，自适应学习 $\rho$）**

   - 计算训练集与验证集的损失差异：

     $$L_{\text{gap}} = (L(D_{\text{val}}, \theta^*(\rho)) - L(D_{\text{train}}, \theta^*(\rho)))^2$$

   - 计算 $\rho$的梯度，并更新 $\rho$：

     $$\rho^{t+1} = \rho^t - \beta \nabla_{\rho} L_{\text{gap}}$$

**这一部分额外增加了 $\rho$ 的梯度计算**，导致计算开销进一步增加。

------

## **3. LETS 计算开销对比**

| 方法         | 计算梯度次数         | 计算额外开销    | 计算量相对 SGD |
| ------------ | -------------------- | --------------- | -------------- |
| **SGD**      | 1                    | 无              | **1.0x**       |
| **SAM**      | 2                    | 无              | **2.0x**       |
| **LETS-SAM** | 2 + 计算 $\rho$ 梯度 | 额外优化 $\rho$ | **>2.0x**      |

- **SGD 只需一次梯度计算**，因此计算量最低（$1.0x$）。
- **SAM 需要两次梯度计算**，计算量约为 **$2.0x$**。
- **LETS 需要两次梯度计算 + 计算$\rho$的梯度**，其计算量**理论上大于$2.0x$**，但具体增加多少取决于 $\rho$ 的优化复杂度。

------

## **4. LETS 计算开销优化策略**

虽然 LETS 计算复杂度较高，但可以采取一些策略优化：

1. 降低$\rho$的更新频率
   - 并非每次训练步骤都更新 $\rho$，而是在每个 epoch 结束后才进行 $\rho$ 的更新，以减少计算量。
2. 使用近似梯度估计
   - 采用**有限差分（Finite Difference）**方法近似计算 $\nabla_{\rho} L_{\text{gap}}$，减少计算开销。
3. 结合 LookSAM 降低 SAM 计算成本
   - LookSAM 仅在部分训练步骤执行完整的 SAM 计算，可以进一步降低计算量。

------

## **5. 结论**

- **LETS 和 SAM 一样，每次更新仍然需要计算两次梯度**，因为它仍然基于 SAM 进行优化。
- **额外增加了对扰动半径$\rho$的优化计算**，因此计算成本会略高于 SAM。
- **相比 SAM，LETS 的计算量略有增加，但优化$\rho$可以提升泛化能力**，在一定情况下可以**降低调参成本**，减少网格搜索带来的计算开销。

虽然 LETS 的计算开销较高，但在**提高泛化能力、减少手动调参的需求**方面具有明显优势，适用于计算资源充足且希望提高模型鲁棒性的场景。



# Paper 8 RST: Randomized Sharpness-Aware Training for Boosting Computational Efficiency in Deep Learning

| **作者:** Yang Zhao; Hao Zhang; Xiuyuan Hu;                  |
| ------------------------------------------------------------ |
| **期刊: , 2023.**                                            |
| **期刊分区:**                                                |
| **本地链接:** [Randomized Sharpness-Aware Training for Boosting Computational Efficiency in Deep Learning_2023_.pdf](zotero://open-pdf/0_7JJDYXVK) |
| **DOI:** [10.48550/arXiv.2203.09962](https://doi.org/10.48550/arXiv.2203.09962) |
| **摘要:** *By driving models to converge to flat minima, sharpness-aware learning algorithms (such as SAM) have shown the power to achieve state-of-the-art performances. However, these algorithms will generally incur one extra forward-backward propagation at each training iteration, which largely burdens the computation especially for scalable models. To this end, we propose a simple yet efficient training scheme, called Randomized Sharpness-Aware Training (RST). Optimizers in RST would perform a Bernoulli trial at each iteration to choose randomly from base algorithms (SGD) and sharpness-aware algorithms (SAM) with a probability arranged by a predefined scheduling function. Due to the mixture of base algorithms, the overall count of propagation pairs could be largely reduced. Also, we give theoretical analysis on the convergence of RST. Then, we empirically study the computation cost and effect of various types of scheduling functions, and give directions on setting appropriate scheduling functions. Further, we extend the RST to a general framework (G-RST), where we can adjust regularization degree on sharpness freely for any scheduling function. We show that G-RST can outperform SAM in most cases while saving 50\% extra computation cost.* |
| **标签:** # RST ,                                            |
| **笔记日期:** 2025/2/23 16:30:08                             |

## 📜 研究核心

------

> Tips: 做了什么，解决了什么问题，创新点与不足？



### ⚙️ 内容



### 💡 创新点



### 🧩 不足



## 🔁 研究内容

------



### 💧 数据



### 👩🏻‍💻 方法



### 🔬 实验



### 📜 结论



## 🤔 个人总结

------

> Tips: 你对哪些内容产生了疑问，你认为可以如何改进？



### 🙋‍♀️ 重点记录



### 📌 待解决



### 💭 思考启发




**You**

其次，详细分析介绍方法，方法提出的目的，与之前方法的区别与创新带你，步骤，细节设置，一些需要注意的小tip，不足之处与未来改进方向等 将上面内容输出为可复制的markdown格式，行内公式使用 $$ , 独立公式使用 $$


**ChatGPT**

### **随机化锐度感知训练（Randomized Sharpness-Aware Training, RST）方法解析**

------

## **1. 方法提出的目的**

### **(1) 现有 SAM 方法的计算开销问题**

- **Sharpness-Aware Minimization (SAM)** 作为提升模型泛化能力的优化方法，需要在每次梯度更新时 **执行两次前向-反向传播**：

  1. **计算对抗扰动方向**：

     $$\epsilon = \rho \frac{\nabla_{\theta} L(\theta)}{\|\nabla_{\theta} L(\theta)\|}$$
     
  2. **计算扰动后的梯度并更新权重**：

     $$\theta_{t+1} = \theta_t - \eta \nabla_{\theta} L(\theta + \epsilon)$$

- 计算成本几乎 **翻倍**，尤其在大规模模型（如 ViT、GPT）训练时会严重影响效率。

### **(2) 目标**

- 本文提出 **随机化锐度感知训练（RST）**，在每次训练迭代中 **以概率$p(t)$随机决定是否使用 SAM**，减少计算成本：
  - **$p(t) = 0$** → 采用普通 SGD 更新。
  - **$p(t) = 1$** → 采用 SAM 更新。
  - **动态调度** $p(t)$，保证训练稳定性。

------

## **2. 与之前方法的区别与创新点**

| **方法**         | **优化目标**                    | **计算量**        | **扰动方式**           |
| ---------------- | ------------------------------- | ----------------- | ---------------------- |
| **SAM**          | 通过锐度最小化提高泛化能力      | 高 (双倍梯度计算) | 固定扰动 $\epsilon$    |
| **ASAM**         | Fisher 信息矩阵自适应锐度       | 高                | Fisher 预处理          |
| **SSAM**         | 仅对部分参数施加扰动            | 低                | 稀疏扰动               |
| **LookSAM**      | 仅部分步长使用 SAM              | 低                | 固定步长扰动           |
| **SAF**          | KL 散度优化路径平滑             | 低                | KL 约束                |
| **RST (本方法)** | 随机采样 SAM 训练，提高计算效率 | 低                | 以 $p(t)$ 概率选择 SAM |

- **创新点**：
  1. **采用 Bernoulli 采样** 控制每轮是否使用 SAM，减少梯度计算次数。
  2. **设计自适应调度策略$p(t)$**，动态调整 SAM 使用频率。
  3. **推广到广义 RST（G-RST）**，引入梯度范数正则化（GNR）优化梯度方向。

------

## **3. 方法详细步骤**

### **(1) RST 训练流程**

**输入**：

- 训练数据 $(x, y)$
- 目标函数 $L(\theta)$
- 训练步数 $T$
- 学习率 $\eta$
- 扰动半径 $\rho$

**算法流程：**

1. **初始化参数**$\theta_0$，设定调度函数$p(t)$。

2. **每次迭代**：

   - 计算梯度 $g_t = \nabla_{\theta} L(\theta_t)$。

   - 以概率 $p(t)$ 进行 **Bernoulli 采样**：

     - 若 $X_t = 0$，则使用 SGD 进行更新：

       ```
       $$g_t^{\text{final}} = g_t$$
       ```

     - 若 $X_t = 1$，则使用 SAM 更新：

       - 计算扰动：

         ```
         $$\epsilon_t = \rho \frac{g_t}{||g_t||}$$
         ```

       - 计算扰动后的梯度：

         ```
         $$g_t^{\text{final}} = \nabla_{\theta} L(\theta_t + \epsilon_t)$$
         ```

   - **更新参数**：

     ```
     $$\theta_{t+1} = \theta_t - \eta g_t^{\text{final}}$$
     ```

3. **返回最终训练的模型参数**$\hat{\theta} = \theta_T$。

------

### **(2) 关键调度策略 p(t)**

为提高计算效率和训练稳定性，本文设计了多种 $p(t)$ 方案：

1. **常数调度（Constant）**：$p(t) = c$，固定比例使用 SAM。
2. **分段调度（Piecewise）**：
   - 训练前半部分用 SGD，后半部分用 SAM（Piecewise-1）。
   - 训练前半部分用 SAM，后半部分用 SGD（Piecewise-2）。
3. **线性调度（Linear）**：$p(t)$ 随着训练进度线性增加或减少。
4. **三角函数调度（Trigonometric）**：
   - $p(t) = \frac{1}{2} + \frac{1}{2} \cos(\frac{t}{T} \pi)$。

------

## **4. 计算成本分析**

- **SAM** 需要 **两次前向-反向传播**，计算量 $\approx 2T$。

- **RST 期望传播次数$\eta_t$**：

  ```
  $$\eta_t = 2p(t) + 1(1 - p(t)) = 1 + p(t)$$
  ```

  - 若 $p(t) = 1$，RST 等价于 **SAM**。
  - 若 $p(t) = 0$，RST 等价于 **SGD**。
  - 若 $p(t) = 0.5$，计算量减少 **50%**。

------

## **5. 关键实验分析**

### **(1) 数据集与模型**

- **数据集**：CIFAR-10、CIFAR-100、ImageNet
- **模型**：
  - ResNet18、WideResNet28-10、PyramidNet
  - ViT-Ti16、ViT-S16、ViT-B16

### **(2) 关键结果**

| 方法          | CIFAR-10 Top-1 Acc | 计算成本 |
| ------------- | ------------------ | -------- |
| SGD           | 94.5%              | 1.0×     |
| SAM           | 95.3%              | 2.0×     |
| RST ($p=0.6$) | 95.1%              | 1.4×     |
| RST ($p=0.8$) | 95.2%              | 1.6×     |

- **计算开销减少 40%-50%**，但性能几乎不变。
- **在 ImageNet 上，RST 甚至超越 SAM**。

------

## **6. 需要注意的小 Tips**

- 训练后期更倾向于使用 SAM（Piecewise-1 调度最佳）。
- 适合 ViT、GPT 这类大规模 Transformer 训练。
- $p(t)$ 不能过小，否则会丧失 SAM 关键优化效果。

------

## **7. 不足之处与未来改进方向**

### **(1) 现有不足**

- **噪声不稳定**：RST 由于引入随机性，在小数据集上可能影响模型收敛速度。
- **调度函数经验性强**：目前 $p(t)$ 依赖实验选择，没有理论最优调度方案。

### **(2) 未来研究方向**

1. **自适应调度策略**：基于损失曲面或梯度信息动态调整 $p(t)$。
2. **结合 LookSAM/SAF/LPF-SGD**，进一步降低计算成本。
3. **应用于 LLMs 训练**，测试在 GPT-4、LLaMA 等大模型上的表现。

------

## **8. 结论**

本文提出的 **随机化锐度感知训练（RST）** 通过 **Bernoulli 采样** 随机选择是否使用 SAM，有效减少计算成本，同时保留 SAM 的泛化能力。

- **实验表明**，RST **节省 40%-50% 计算成本**，同时保持与 SAM **相近或更优的泛化性能**。
- **未来可以结合自适应调度策略，使其更加高效稳定。**

🚀 **RST 让 SAM 更高效，适用于大规模 AI 任务！** 🚀

# LoRA related



# Paper1: Flat minima generalize for low-rank matrix recovery

### 论文分析：《Flat Minima Generalize for Low-Rank Matrix Recovery》

#### 1. 研究背景 (Scope and Setting)

该论文研究了低秩矩阵恢复 (Low-Rank Matrix Recovery) 这一重要的优化问题，该问题在信号处理、机器学习、统计学等多个领域均有应用。论文探讨了 **“Flat Minima” (平坦极小值)** 在这一类问题中的泛化能力，并试图从理论角度理解其作用。

------

#### 2. 核心思想 (Key Idea)

论文的核心假设是：**在低秩矩阵恢复问题中，找到平坦的极小值 (Flat Minima) 可以提高模型的泛化能力，并能够准确恢复真实数据矩阵。**
 为了验证这一假设，作者定义了 Hessian 迹 (Hessian Trace) 作为度量平坦性的标准，并分析了多个不同的低秩矩阵恢复问题。

------

#### 3. 方法 (Method)

论文主要关注**低秩矩阵分解问题** (Low-Rank Matrix Factorization)，其通用形式如下：

$$\min_{L, R} f(L, R) = \| A(LR^T) - b \|_2^2$$

其中：

- $L \in \mathbb{R}^{d_1 \times k}, R \in \mathbb{R}^{d_2 \times k}$ 是优化变量，$k \geq r^*$ 表示超参数，通常选得较大以建模**过参数化**的情况；
- $A$ 是线性测量映射，$b$ 是观测值。

**平坦极小值的定义：**
 作者通过 Hessian 迹 (Hessian Trace) 来定义平坦极小值：

$$\text{str}(D^2 f(L, R))$$

并提出了**缩放迹 (Scaled Trace)** 作为衡量标准：

$$\text{str}(D^2 f(L, R)) = \frac{1}{d_1} \sum_{i \leq d_1, j \in [k]} D^2 f(L, R)[e_i e_j^T] + \frac{1}{d_2} \sum_{i > d_1, j \in [k]} D^2 f(L, R)[e_i e_j^T]$$

**主要分析的低秩问题包括：**

1. **矩阵感知 (Matrix Sensing)**
2. **双线性感知 (Bilinear Sensing)**
3. **鲁棒主成分分析 (Robust PCA)**
4. **协方差矩阵估计 (Covariance Matrix Estimation)**
5. **单隐藏层神经网络 (Single-Layer Neural Networks)**
6. **矩阵补全 (Matrix Completion)**

论文中详细探讨了这些问题的数学性质，并证明了**平坦极小值能够准确恢复真实数据矩阵**。

------

#### 4. 贡献 (Contribution)

论文的主要贡献如下：

- **理论性贡献：** 提出了在低秩矩阵恢复问题中，**平坦极小值 (Flat Minima) 能够准确恢复真实矩阵** 的理论证明，并提供了相应的解析解。
- **算法启示：** 提出了基于 Hessian 迹的优化方法，有助于设计新的正则化策略，使得优化算法能够更倾向于平坦极小值。
- **实验验证：** 通过数值实验验证了平坦极小值在低秩矩阵恢复任务中的有效性。

------

#### 5. 与现有方法的对比 (Difference and Innovation)

- 与传统核范数最小化 (Nuclear Norm Minimization) 的区别：
  - 核范数最小化 (Nuclear Norm Minimization) 通过约束核范数来保证低秩性；
  - 论文的方法则通过优化 Hessian 迹，直接寻找到**平坦的低秩解**。
  - **论文的关键发现是：虽然平坦极小值不等同于核范数最小解，但它们也能够准确恢复真实矩阵。**
- 与梯度下降等优化方法的关系：
  - 论文指出，**梯度下降等优化方法在一定条件下能够自动偏向平坦极小值**，这可能是深度学习模型泛化能力的重要来源。

------

#### 6. 结果与结论 (Results and Conclusion)

- **主要结论：** 在低秩矩阵恢复问题中，Hessian 迹最小的极小值 (即平坦极小值) 具有良好的泛化能力，并且能够准确恢复真实矩阵。
- **数值实验：** 论文通过数值实验验证了平坦极小值的恢复能力，实验结果支持理论分析。
- **泛化性：** 论文的结果适用于多个不同的低秩矩阵恢复问题，并且能够推广到神经网络训练问题。

------

#### 7. 论文的不足与未来研究方向 (Limitations and Future Work)

**不足之处：**

1. **仅考虑了理论分析，缺乏大规模实验验证。** 论文虽然给出了部分数值实验，但主要是针对简单的低秩问题，没有在更复杂的机器学习任务（如深度学习）上进行测试。
2. **对局部极小值 (Local Minima) 的分析不足。** 论文主要讨论了全局最优解的情况，而实际优化问题中，梯度下降等方法可能会收敛到局部极小值。
3. **对非凸优化问题的扩展有限。** 论文的理论分析主要基于矩阵分解模型，对于更一般的非凸优化问题如何扩展尚未探讨。

**未来改进方向：**

1. **将方法应用于更复杂的机器学习任务，例如神经网络训练。**
2. **研究局部极小值的平坦性如何影响模型的泛化能力。**
3. **开发更高效的优化算法，使得训练过程更倾向于找到平坦极小值。**
4. **探索平坦极小值与其他正则化方法（如权重衰减、梯度噪声等）之间的关系。**

------

### 总结

本论文从理论角度研究了**平坦极小值**在低秩矩阵恢复问题中的作用，并证明了其能够准确恢复真实数据矩阵。这一发现为深度学习模型的泛化能力提供了新的视角，并可能对优化算法的设计产生影响。未来的研究可以进一步探讨该方法在更广泛的机器学习任务中的应用，以及与其他正则化方法的结合。



### **方法解析：《Flat Minima Generalize for Low-Rank Matrix Recovery》**

------

## **1. 方法提出的目的**

### **背景**

- 低秩矩阵恢复 (Low-Rank Matrix Recovery) 是机器学习和统计信号处理中的重要问题，广泛应用于**矩阵补全 (Matrix Completion)、鲁棒主成分分析 (Robust PCA)、协方差矩阵估计 (Covariance Matrix Estimation) 等**。
- 在深度学习领域，已有研究表明**模型的泛化能力与损失函数曲面的几何形态密切相关**，其中“平坦极小值 (Flat Minima)”有助于提升泛化能力。
- 该研究关注 **低秩矩阵恢复问题中的平坦极小值**，并探讨它们如何促进泛化能力，并提供理论分析支持。

### **目标**

- **理论分析**：证明低秩矩阵恢复问题中的平坦极小值能够准确恢复目标矩阵，并具有良好的泛化能力。
- **泛化能力研究**：探索 Hessian 迹 (Hessian Trace) 在衡量低秩解的平坦性和泛化能力之间的关系。
- **算法优化启示**：基于理论研究，为优化算法提供方向，使其倾向于找到平坦极小值。

------

## **2. 主要区别与创新**

### **区别于传统方法**

| 方法                                         | 目标                                | 计算方式                     | 泛化能力                       |
| -------------------------------------------- | ----------------------------------- | ---------------------------- | ------------------------------ |
| **核范数最小化 (Nuclear Norm Minimization)** | 通过约束核范数确保低秩              | 解决 $\min \|X\|_*$          | 泛化能力较强，但计算成本高     |
| **梯度下降 (Gradient Descent) 方法**         | 直接优化目标函数                    | 受初始点影响较大             | 泛化能力不稳定                 |
| **本论文提出的 Hessian 迹最小化方法**        | 通过最小化 Hessian 迹找到平坦极小值 | 低秩矩阵恢复任务上的理论分析 | 泛化能力强，可稳定恢复目标矩阵 |

### **创新点**

1. 提出 Hessian 迹作为衡量低秩解平坦性的新标准
   - 以 Hessian 迹 (Hessian Trace) 作为平坦极小值的度量，建立泛化能力的理论连接。
2. 首次提供低秩矩阵恢复问题中平坦极小值泛化能力的理论证明
   - 证明在多个低秩问题（如矩阵补全、鲁棒 PCA）中，平坦极小值能够准确恢复目标矩阵。
3. 探索梯度下降算法对平坦极小值的偏好
   - 分析常见优化算法（如随机梯度下降）是否自然倾向于收敛到平坦极小值。

------

## **3. 具体方法步骤**

### **3.1 低秩矩阵恢复问题建模**

研究的目标是求解：

$$\min_{L, R} f(L, R) = \| A(LR^T) - b \|_2^2$$

其中：

- $L \in \mathbb{R}^{d_1 \times k}, R \in \mathbb{R}^{d_2 \times k}$ 是优化变量，$k \geq r^*$ 以建模**过参数化情况**；
- $A$ 是线性测量映射，$b$ 是观测值。

**核心假设**：在优化问题中，平坦极小值 (Flat Minima) 能够泛化更好，其数学定义如下。

------

### **3.2 平坦极小值的数学定义**

- **Hessian 迹 (Hessian Trace)**

  $$\text{str}(D^2 f(L, R))$$

  衡量损失函数的二阶导数之和，用于判断曲面的平坦性。

- **归一化 Hessian 迹 (Scaled Trace)**

  $$\text{str}(D^2 f(L, R)) = \frac{1}{d_1} \sum_{i \leq d_1, j \in [k]} D^2 f(L, R)[e_i e_j^T] + \frac{1}{d_2} \sum_{i > d_1, j \in [k]} D^2 f(L, R)[e_i e_j^T]$$

  用于对比不同规模的问题，避免 Hessian 迹受维度影响。

------

### **3.3 低秩问题实例**

作者验证了该理论在多个问题中的适用性：

1. **矩阵感知 (Matrix Sensing)**
   - 目标：恢复一个低秩矩阵 $X^*$，通过测量 $y = A(X^*) + \epsilon$ 进行重构。
   - 证明 Hessian 迹最小的极小值能够准确恢复 $X^*$。
2. **双线性感知 (Bilinear Sensing)**
   - 目标：从双线性测量中恢复低秩矩阵，求解 $y = A(LR^T)$。
   - 证明 Hessian 迹最小的解比其他方法泛化性更强。
3. **鲁棒主成分分析 (Robust PCA)**
   - 目标：将低秩部分与稀疏噪声部分分离，求解 $\min_{L, S} \|L\|_* + \lambda \|S\|_1$。
   - 证明平坦极小值能够更好地恢复 L 结构。
4. **协方差矩阵估计 (Covariance Matrix Estimation)**
   - 目标：基于观测数据估计一个低秩协方差矩阵。
   - 证明 Hessian 迹最小的解能够减少估计误差。
5. **单隐藏层神经网络 (Single-Layer Neural Networks)**
   - 目标：学习单层神经网络的权重矩阵。
   - 结果表明，平坦极小值能够减少过拟合，提高泛化能力。

------

## **4. 关键实验结果**

- **Hessian 迹较小的极小值往往对应于更好的泛化误差**
- **随机梯度下降 (SGD) 有自然倾向收敛到 Hessian 迹较小的极小值**
- 在多个数据集上的实验验证了该理论
  - 低秩矩阵补全任务中的恢复误差较低
  - 在鲁棒 PCA 任务中的分离效果较好

------

## **5. 关键注意点 (Tips)**

1. Hessian 迹最小不代表训练误差最小

   ：

   - 在某些情况下，优化可能会倾向于 sharper minima，因此需要额外的正则化策略。

2. 优化算法的选择影响极小值的平坦性

   ：

   - 研究表明，SGD 自然会趋向于 Hessian 迹较小的区域。

3. 泛化误差的衡量方式需要进一步研究

   ：

   - 目前主要基于 Hessian 迹，但可能需要结合其他指标。

------

## **6. 不足之处与未来方向**

### **6.1 不足之处**

1. 理论结果局限于低秩矩阵恢复问题

   ：

   - 该研究主要针对低秩问题，尚未在深度神经网络等更复杂的模型中进行验证。

2. 实验规模较小

   ：

   - 论文中的实验主要基于合成数据集，而不是大规模真实数据集。

3. 对非凸优化的研究有限

   ：

   - 低秩矩阵恢复问题大多是非凸优化问题，论文未对其优化路径进行详细讨论。

### **6.2 未来方向**

1. 扩展到深度学习模型

   ：

   - 研究如何在 ResNet、Transformer 等神经网络中应用 Hessian 迹最小化方法。

2. 结合正则化策略

   ：

   - 例如，将 Hessian 迹作为一种正则化项引入优化目标，以引导训练过程。

3. 大规模实验验证

   ：

   - 在更复杂的数据集（如 ImageNet）上验证该方法的可行性。

------

## **7. 结论**

- 论文提出了 Hessian 迹最小化作为低秩矩阵恢复问题的泛化能力衡量标准。
- 证明了平坦极小值在多个任务中均能有效恢复目标矩阵。
- 未来可探索其在深度学习中的应用，以及结合其他正则化策略进一步提升泛化能力。 🚀



# Paper2  Implicit Regularization of Sharpness-Aware Minimization for Scale-Invariant Problems

**ChatGPT**

### 论文概述：《Implicit Regularization of Sharpness-Aware Minimization for Scale-Invariant Problems》

#### **1. 研究范围（Scope）**

本论文研究**Sharpness-Aware Minimization (SAM)** 在**尺度不变（Scale-Invariant）问题中的隐式正则化特性，特别是在低秩适配（LoRA）等模块中的应用。论文关注具有两组变量的问题，并从一个新的角度，即平衡性（Balancedness）** 来刻画 SAM 的优化特性，研究其对广义优化问题及不同数据特性的适应性。

#### **2. 核心思想（Key Idea）**

- 现有的 SAM 主要从**锐度（Sharpness）**出发解释其正则化特性，然而这在尺度不变问题（如 LoRA）中并不充分。
- 论文引入了新的度量指标：**平衡性（Balancedness）**，即两个变量平方范数的差异，该指标能够描述 SAM 在整个优化轨迹上的行为，而不仅仅是在局部最优附近。
- SAM 具有**自适应数据的隐式正则化特性**，在数据异常（如噪声、分布漂移）较大的情况下，SAM 的正则化作用更强。

#### **3. 方法（Method）**

1. **定义平衡性（Balancedness）**:

   - 设变量 $x \in \mathbb{R}^{d_1}$ 和 $y \in \mathbb{R}^{d_2}$，定义平衡性：

     $$B_t = \frac{1}{2}(\|x_t\|^2 - \|y_t\|^2)$$

   - 论文证明 SAM 可以**驱动$B_t$收敛到 0**，即变量的平方范数趋于相等，而 SGD 不具备这一特性。

2. **平衡性在不同问题中的作用**:

   - 在**非过参数化（NOP）**问题下，SAM 使 $|B_t|$ 逐步减小，趋近于某个较小的阈值。
   - 在**过参数化（OP）**问题下，SAM 甚至可以保证 $B_t$ **完全收敛** 到 0。

3. **数据噪声的影响**:

   - 数据噪声较大的情况下，SAM 对平衡性的正则化更强，这解释了为何 SAM 在数据异常的情况下比 SGD 具有更好的泛化能力。

4. **提出一种新的优化方法——BAR（Balancedness-Aware Regularization）**:

   - 论文基于上述分析提出了一种计算量更低的 SAM 变体，**BAR（平衡性感知正则化）**。
   - **不同于 SAM 需要计算两次梯度，BAR 仅需一次梯度计算**，并通过显式的正则化项来模仿 SAM 对平衡性的优化效果。
   - BAR 在**LoRA 微调**（如 RoBERTa、GPT2、OPT-1.3B）任务上提高了测试性能，并减少了 95% 计算开销。

#### **4. 主要贡献（Contribution）**

- **理论贡献**:
  - 提出了**平衡性（Balancedness）**作为 SAM 在尺度不变问题上的新度量指标。
  - 证明了**SAM 促进平衡性，并且数据噪声会增强其正则化作用**。
  - 该分析适用于**任意批量大小**，相比现有 SAM 研究（如 Wen et al. 2023）更加普适。
- **实践贡献**:
  - **提出 BAR（Balancedness-Aware Regularization）**，减少 95% 计算量，使 SAM 更适用于大规模深度学习任务，如 LoRA 微调。
  - 在多个任务（RoBERTa、GPT2、OPT-1.3B）上进行了验证，BAR 显著提升了 LoRA 的微调性能，同时保留了 SAM 的泛化优势。

#### **5. 与现有方法的不同（Difference & Innovation）**

| 方法                | 关键特性                                       | 计算成本               |
| ------------------- | ---------------------------------------------- | ---------------------- |
| **SAM**             | 基于**锐度**进行隐式正则化，强制寻找平坦最优解 | 高，需要计算两次梯度   |
| **SGD**             | 标准梯度下降，无特殊正则化                     | 低                     |
| **m-SAM**           | 通过分块计算锐度优化                           | 高                     |
| **BAR（本文方法）** | 基于**平衡性**进行显式正则化，避免两次梯度计算 | 低（仅需一次梯度计算） |

#### **6. 主要实验结果（Results & Conclusion）**

- 在 LoRA 微调任务上，**BAR 以 5% 计算开销的代价达到了 SAM 相似的性能，甚至在某些任务上表现更优**。
- SAM 的隐式正则化特性对**数据噪声大的任务特别有效**，BAR 继承了这一优点。
- 平衡性的度量方式可用于解释**SAM 在 LoRA、Transformer 软注意力机制等模型中的优化效果**。

#### **7. 论文的不足之处（Limitations）**

- **局限于尺度不变问题**: BAR 方法主要适用于 LoRA 等尺度不变结构，在普通深度学习任务上可能不适用。
- **理论分析未覆盖更复杂的优化动态**: 目前的分析主要针对理想化情况，如何推广到更广泛的神经网络架构仍然是一个开放问题。

#### **8. 未来研究方向（Future Directions）**

- **拓展至其他尺度不变架构**，如 Transformer 的 Softmax 机制。
- **结合其他隐式正则化策略**，如权重归一化、变分贝叶斯等，以进一步提升泛化能力。
- **研究 SAM 其他可能的隐式正则化形式**，不仅仅局限于平衡性和锐度。

------

## **总结**

这篇论文提出了一个新的度量指标**平衡性（Balancedness）**，用来描述 SAM 在尺度不变问题上的优化特性，并基于此开发了计算成本更低的**BAR 方法**，显著减少了 SAM 的计算开销，同时保持或超过其泛化性能。该研究为 SAM 在**低秩适配（LoRA）、Transformer、尺度不变优化等问题中的作用提供了新的理解，并提出了一种更高效的变体，有望在未来的大规模深度学习优化中发挥重要作用。

------

以上内容采用 Markdown 格式编写，可直接复制粘贴至文档或代码编辑器中进行使用。


**ChatGPT**

## **Balancedness vs. Sharpness：区别与联系**

本文提出的 **Balancedness（平衡性）** 和 SAM 研究的 **Sharpness（锐度）** 是两个不同的优化目标，但它们之间存在一定的联系。本节分析二者的区别、联系，以及它们各自的出发点和用于描述的问题。

------

## **1. Balancedness vs. Sharpness 的核心区别**

| **属性**     | **Balancedness (平衡性)**                                    | **Sharpness (锐度)**                                   |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------ |
| **出发点**   | 关注**参数范数的均衡性**，研究不同参数组在优化过程中的变化   | 关注**损失曲面的几何结构**，避免收敛到锐度较大的极小值 |
| **主要描述** | **参数权重在不同部分的尺度均衡程度**，尤其适用于**尺度不变优化问题（Scale-Invariant Problems）** | **局部最优点附近的曲率大小**，用于衡量模型泛化能力     |
| **数学定义** | $B_t = \frac{1}{2} (\|x_t\|^2 - \|y_t\|^2)$                  | $\max_{\|\epsilon\|\leq\rho} L(w+\epsilon) - L(w)$     |
| **优化方法** | 通过显式正则化（BAR 方法）调整参数范数的变化                 | 通过对抗扰动（SAM）优化局部最大损失                    |
| **主要影响** | 适用于**低秩适配（LoRA）**等参数共享架构，改善微调效率       | 提升深度学习模型在高噪声环境下的泛化能力               |
| **适用场景** | 适用于**尺度不变问题（LoRA、Transformer Softmax）**          | 适用于**深度学习优化、分类任务、语言建模等**           |

------

## **2. 两者的联系**

虽然 **Balancedness 和 Sharpness 在定义和优化目标上有所不同**，但它们在以下几个方面存在联系：

1. **都涉及隐式正则化（Implicit Regularization）**
   - SAM 通过优化损失曲面的局部锐度，使模型参数收敛到更平坦的区域，从而提高泛化能力。
   - Balancedness 研究发现 **SAM 在尺度不变问题中不仅影响 Sharpness，还优化了参数范数的均衡性**，即它在隐式调整参数的比例关系。
2. **都影响模型的泛化能力**
   - Sharpness 直接影响损失曲面，使模型不易受到微小扰动的影响，提高测试集性能。
   - Balancedness 影响参数的相对尺度分布，使优化更加稳定，并在低秩架构中提高微调效率。
3. **Balancedness 可以被看作是 Sharpness 在尺度不变问题中的特殊形式**
   - 在尺度不变问题（如 LoRA 微调）中，直接优化 Sharpness 可能不会很好地控制参数范数的增长。
   - 但 **Balancedness 通过正则化不同参数组的平方范数，使得优化更稳定，避免参数更新不均衡的问题**。

------

## **3. 两者的不同出发点**

- **Sharpness 出发点：优化损失曲面的几何性质**
  - SAM 研究的是**局部最优解的锐度**，希望模型在参数扰动后仍然能保持低损失。
  - SAM 关注的是局部的曲率，使得优化后的模型更加稳健，特别是在数据分布发生变化时。
- **Balancedness 出发点：优化参数范数的比例**
  - Balancedness 研究的是**参数更新在不同部分的均衡性**，而不仅仅是损失曲面的局部曲率。
  - 论文发现，在**尺度不变问题（如 LoRA 低秩适配）**中，**参数的均衡性比曲率更加关键**，直接优化 Sharpness 可能无法有效提升性能。

------

## **4. Balancedness 和 Sharpness 各自描述的问题**

### **（1）Sharpness 主要用于描述的性质**

- 衡量损失曲面在局部最优点附近的平坦程度。
- 通过最小化锐度（Sharpness）来提升深度学习模型的泛化能力。
- **主要用于深度学习优化**，特别是在**分类任务、图像识别、语言模型等标准深度学习任务**上。

### **（2）Balancedness 主要用于描述的性质**

- 衡量模型参数的**尺度是否均衡**，特别是对于 LoRA 这样的低秩适配任务。
- 在**尺度不变优化问题**（如 Transformer Softmax 机制、权重归一化）中，Balancedness 比 Sharpness 更能影响模型的优化行为。
- **适用于参数分布需要均衡调整的优化任务**，如**LoRA 微调、模型量化（Quantization）等**。

------

## **5. 未来研究方向**

1. **如何结合 Balancedness 和 Sharpness 进行优化？**
   - 目前研究主要分析了二者的关系，但未来可以探索**联合优化策略**，既控制损失曲面的锐度，又确保参数的均衡性。
2. **Balancedness 在更大规模模型中的作用**
   - 论文验证了 Balancedness 在 LoRA 任务中的有效性，但是否能推广到 **完整的 Transformer 训练**（如 GPT、BERT）仍需进一步研究。
3. **能否基于 Balancedness 设计新的泛化优化方法？**
   - 论文提出了 BAR（Balancedness-Aware Regularization），未来可以结合 **Sharpness-Aware Minimization（SAM）**，探索是否可以构造出更强的泛化正则化策略。

------

## **6. 结论**

- **Sharpness 关注的是损失曲面的几何特性，Balancedness 关注的是参数范数的均衡性。**
- **两者都能提高泛化能力，但作用机制不同。**
- **Balancedness 是在尺度不变问题中的重要优化属性，SAM 在 LoRA 任务中也隐式优化了 Balancedness。**
- **BAR 方法可以视为 SAM 在尺度不变问题上的低计算成本替代方案。**
- **未来可以结合两者进行联合优化，进一步提升模型的泛化能力。** 🚀

## **Balancedness-Aware Regularization (BAR) 方法解析**

### **1. 直接将 LoRA 和 SAM 结合的两个问题**

论文指出，**直接在 LoRA 任务上应用 SAM 可能会遇到两个主要问题**：

1. **计算成本翻倍（Computational Cost Doubling）**
   - **问题**：SAM 需要计算 **两次梯度**，第一次计算用于生成对抗扰动（Sharpness-Aware Perturbation），第二次计算用于参数更新。这导致计算成本比普通 SGD **翻倍**，特别是在**大规模语言模型**（如 GPT、BERT）中，梯度计算已经是训练的主要开销，因此额外的计算负担会显著影响训练效率。
   - **理解**：大规模语言模型（LLMs）通常需要模型并行（Model Parallelism）和梯度累积（Gradient Accumulation）来应对计算资源的限制，而 SAM 由于其额外的计算需求，与这些技术的结合会变得复杂，影响内存管理和运行时效率。
2. **与梯度累积（Gradient Accumulation）和低精度训练（Low-Precision Training）兼容性差**
   - **问题**：在 HuggingFace 等框架中，LoRA 训练时广泛使用梯度累积和低精度计算（如 FP16、BF16）来节省内存和加速训练。然而，SAM 需要两次梯度计算，**这可能会影响这些优化策略的稳定性和收敛性**，需要额外的修改才能正确地集成。
   - **理解**：梯度累积允许在多个小批次（Micro-Batches）上累积梯度，以减少显存占用；低精度训练则可以减少计算和存储需求。但由于 **SAM 会修改梯度的计算方式**，它可能会破坏这些优化策略的数值稳定性，影响最终的训练效果。

------

### **2. BAR（Balancedness-Aware Regularization）如何解决上述问题**

为了应对上述挑战，论文提出了一种新的 **Balancedness-Aware Regularization (BAR)** 方法，该方法基于 **SAM 的隐式正则化作用**，但**通过显式正则化来避免两次梯度计算**。

#### **(1) 关键思想**

- **观察到 SAM 的隐式正则化可以通过显式正则化项$α_t(x^Tx - y^Ty)$来模拟**。
- 这一正则化项来源于**矩阵感知（Matrix Sensing）问题**，可以在不计算两次梯度的情况下**模拟 SAM 在参数平衡性上的优化作用**。

#### **(2) 具体方法**

BAR 提供了两种变体：

1. **oBAR（Overparameterized BAR）** 适用于**过参数化（OP）**问题。
2. **nBAR（Non-Overparameterized BAR）** 适用于**非过参数化（NOP）**问题，如 LoRA。

##### **oBAR（适用于过参数化 OP）**

- 定义$α_t$使其逼近 SAM 对$B_t$的优化动态：

  $$α_t := O(|f ′(x_t^T y_t)| / \sqrt{\|x_t\|^2 + \|y_t\|^2})$$

- **当$‖x_t‖ \geq ‖y_t‖$时，$B_t$逐渐减小**，即参数的平方范数趋于相等，使得参数在训练过程中保持均衡。

##### **nBAR（适用于非过参数化 NOP，如 LoRA）**

- LoRA 是非过参数化（NOP）的，因此直接借鉴 Theorem 2 的思想，去掉某些项并模拟 SAM 的动态：

  - 如果 $‖g_{x_t}‖^2 < ‖g_{y_t}‖^2$，则优化目标变为：

    $$α_t(x^Tx - y^Ty)$$

  - 否则：

    $$α_t(y^Ty - x^Tx)$$

- 这样可以**通过显式正则化项来模拟 SAM 的隐式优化作用，而不需要两次梯度计算**。

------

### **3. BAR 方法的优势**

1. **计算成本降低**
   - **相比 SAM，BAR 仅需一次梯度计算**，大幅减少计算量，适用于大规模语言模型的微调。
2. **兼容梯度累积 & 低精度训练**
   - 由于 BAR 仅使用一次梯度计算，它可以无缝地集成到 HuggingFace 等框架中，不会破坏梯度累积或低精度计算的数值稳定性。
3. **适用于 LoRA**
   - **LoRA 由于层数较少，受参数不均衡影响较大**，BAR 可以在**每个 LoRA 层上独立应用**，提升最终的微调效果。
4. **克服“多层诅咒”（Curse of Multi-Layer）**
   - 论文发现，**层数越多，平衡性（Balancedness）越差**，影响最终测试效果。
   - BAR 可以通过独立对每层 LoRA 进行正则化，**克服这一问题，提高测试精度**。

------

### **4. BAR 与 SAM 对比**

| 方法                | 计算成本     | 是否需要两次梯度计算 | 适用于梯度累积 & 低精度 | 适用于 LoRA               |
| ------------------- | ------------ | -------------------- | ----------------------- | ------------------------- |
| **SAM**             | **高**（2x） | **是**               | **兼容性较差**          | **可能影响微调稳定性**    |
| **BAR（本文方法）** | **低**（1x） | **否**               | **高度兼容**            | **可独立应用于每层 LoRA** |

------

### **5. 训练调度（Scheduler of $α_t$）**

论文建议在 nBAR 和 oBAR 方法中**使用递减调度策略（Scheduler）**来调整 $α_t$ 的正则化强度：

- **采用余弦调度（Cosine Scheduler）或线性调度（Linear Scheduler）**，使得 $α_t$ 在训练后期逐渐减小，以确保平衡性优化不会干扰最终的收敛。

------

### **6. 结论**

- **直接在 LoRA 上应用 SAM 存在计算成本高和兼容性问题**，论文提出了 **Balancedness-Aware Regularization (BAR)** 作为更高效的替代方案。
- **BAR 通过显式正则化（$α_t(x^Tx - y^Ty)$）模拟 SAM 的优化作用**，避免了两次梯度计算，同时保持或超过 SAM 的泛化能力。
- **nBAR 适用于非过参数化任务（如 LoRA），oBAR 适用于过参数化任务**，都可以用作高效替代方案。
- **BAR 可以单独应用于每层 LoRA，克服多层架构带来的优化问题，提高最终的测试精度**。

------

### **7. 未来研究方向**

1. **如何进一步优化 BAR 在 LoRA 任务上的泛化能力？**
2. **能否将 BAR 方法扩展到更广泛的任务，如全参数微调？**
3. **探索 BAR 在计算机视觉任务（如 ViT 训练）中的应用，进一步验证其通用性。**

------

### **总结**

论文提出的 **BAR 方法通过显式正则化平衡参数范数，避免了 SAM 需要计算两次梯度的问题，使其适用于 LoRA 任务的高效微调。** 未来研究可以探索更广泛的应用场景，并结合其他正则化策略进一步提升其性能。🚀







# Paper3  FLAT-LORA: LOW-RANK ADAPTION OVER A FLAT LOSS  LANDSCAPE



## **1. 论文内容概述**
## 1.1 研究背景与目标
近年来，大规模预训练模型（如语言模型、视觉模型等）的微调虽然能取得优异性能，但其完整微调（Full Fine-tuning）往往成本高昂。**LoRA**（Low-Rank Adaptation）通过仅在可训练的低秩增量矩阵上进行更新，可以大幅减少所需训练参数，从而在保持高效推理的同时节省显存与计算。然而，LoRA通常只关注“低秩子空间”本身的最优解是否表现良好，而忽略了合并后在整个完整权重空间是否落在平坦极小点（flat minima）周围。若该低秩解在完整参数空间中位于尖锐区域，则其泛化能力可能受到影响。

为此，本文提出 **Flat-LoRA**：在 LoRA 框架下，以最小化全权重空间的“尖锐度”为目标，通过随机扰动近似贝叶斯期望损失的方式来实现对“平坦解”的显式搜索，且尽量不增加额外梯度开销或巨大存储需求，最终在自然语言理解与图像分类等多种场景均取得更佳的微调性能。

## 1.2 主要思想与方法
1. **LoRA 复习**  
   对原始大模型权重记为 $W$（冻结），仅训练一个低秩增量 $\Delta W = s \cdot B A$，其中 $A, B$ 分别是可训练矩阵，秩 $r \ll \min(d, k)$。推断时将增量合并回主干权重 $W + \Delta W$ 而不增加推理成本。
2. **平坦极小点（Flat Minima）在完整权重空间**  
   - 不同于只在 $(A,B)$ 子空间惩罚尖锐度（类似 LoRA-SAM 的思路），本文强调要在完整权重空间 $W + \Delta W$ 中评估并约束曲率。  
   - 若仅在低秩子空间看似“平坦”，但在合并后的完整参数空间却可能出现“尖锐方向”，从而损害泛化。
3. **随机扰动 + 贝叶斯期望损失**  
   - 为避免像传统 SAM 那样在大模型中“计算两次梯度”与“存储额外完整拷贝”，Flat-LoRA 采取一种**随机权重扰动**策略：  
   $$
   \min_{A,B} \; \mathbb{E}_{\epsilon \sim \mathcal{N}(0,\sigma^2 I)} \Bigl[ L \bigl(W + s\cdot BA + \epsilon \bigr) \Bigr].
   $$
   - 通过“在完整权重上注入小扰动”，并对扰动后损失的期望值进行优化，可在保持训练开销近似不变的情况下，实现对平坦极小点的逼近。
4. **扰动生成策略**  
   - 论文设计了基于“filter 结构 + 归一化因子” 的高效随机扰动，为每个filter施加与其范数相关的噪声。  
   - 这样可保证扰动大小与层规模或输入维度无关，且只需存储一个随机种子，无需像传统 SAM 那样保存完整额外参数拷贝。
---
## 1.3 贡献与创新
1. **将平坦度目标延伸至完整参数空间**  
   在 LoRA 中首次明确关注合并后权重是否位于“全局平坦区域”，而不是仅让低秩子空间看似平坦。
2. **效率与内存占用友好**  
   不同于 SAM 每步两次梯度计算、需复制一套全模型权重，所提方法仅基于一次正向+反向，并通过可控的随机扰动实现近似。
3. **易与已有 LoRA 改进结合**  
   实验也表明 Flat-LoRA 能与其他 LoRA 优化/初始化策略配合，进一步提升性能。
4. **在多任务实证验证**  
   在自然语言理解（GLUE子集：MNLI、SST2、CoLA、QNLI、MRPC）和图像分类（CIFAR-10/100、Cars、SVHN、DTD）等任务上，均表现优于纯 LoRA 及部分已有改进。
---
## 1.4 与现有方法之对比
- **LoRA-SAM/LoRA 其他改进**：以往方法通常仅在 LoRA 参数空间上做 “尖锐度感知”。本文却对合并后完整权重空间的曲率进行近似最小化，更能避免“子空间平坦但全空间仍尖锐”的问题。
- **Sharpness-Aware Minimization**：原始 SAM 需双倍梯度计算，不适合大模型高效微调；本文利用随机扰动期望损失巧妙规避了额外存储和计算量。
- **其他 LoRA 结构优化**：如 DoRA、PiSSA、LoRA-GA 等多关注秩分配或初始化策略，可与本文方法并行结合（论文中也做了实验）。
---
## 1.5 实验结果与结论
1. **NLP (T5-Base) + GLUE 子集**  
   - 测试集精度均比普通 LoRA 提升约 0.3% - 0.6%，在某些数据集（如 CoLA、MRPC）提升更明显；  
   - rank 增大时，纯 LoRA 易过拟合，而 Flat-LoRA 则能保持更稳定性能。
2. **视觉 (CLIP ViT-B/32) + 多分类任务**  
   - 如 CIFAR-10/100、Cars、SVHN、DTD 等，Flat-LoRA 在 rank=8 或 rank=16 时均超越原 LoRA，且在一些数据集甚至超过 Full Fine-tuning 的基线。
3. **消融与对比试验**  
   - 与其他 LoRA 改进方法（DoRA、LoRA-GA、LoRA+、PiSSA）结合，可持续提升；  
   - 在极低秩 (r=1) 场景下，也能显著改善性能；体现了对全空间“尖锐度”的惩罚确实帮助减轻过拟合。
> **结论**：通过在完整权重空间约束平坦度，Flat-LoRA 在各种任务与结构上取得了强泛化性能和稳定表现，可视为一种适合大模型微调的高效“平坦化”方法。
---
## **2. 模型规模与数据集对照表**
下表汇总论文中使用的 **基础模型** 与 **数据集** 情况：
| **基模型**        | **大致参数规模** | **使用场景**                                        |
| ----------------- | ---------------- | --------------------------------------------------- |
| **T5-Base**       | ~220M (参数量)   | 语义理解（GLUE 子集：MNLI, SST2, CoLA, QNLI, MRPC） |
| **CLIP ViT-B/32** | ~88M (参数量)    | 图像分类（CIFAR-10/100, Cars, SVHN, DTD）           |
> - **GLUE子集**：论文主要针对 MNLI、SST2、CoLA、QNLI、MRPC 等五个任务；评价指标多用 Accuracy。  
> - **图像分类**：CIFAR-10/100 (32×32)、Cars、SVHN、DTD 等。  
> - 训练细节：LoRA rank 通常取 8 或 16；扰动方差 $\sigma$ 设置因任务而异；采用余弦退火等学习率策略。
---
**参考文献：**  
- [Flat-LoRA: Low-Rank Adaption over a Flat Loss Landscape, 2024]  

- 相关 LoRA/SAM/随机扰动方法参考原文献列表。  

  

| 项目      | 内容                                                         |
| --------- | ------------------------------------------------------------ |
| 📖研究背景 | 大规模预训练模型微调在计算与存储开销方面代价高昂，LoRA 虽可降低成本但常忽视 LoRA 子空间与完整参数空间的关系，导致泛化性能受到影响。 |
| 🎯研究目的 | 在完整的模型权重空间中寻找“平坦”区域的低秩适配（Flat-LoRA），以同时兼顾训练效率与模型的泛化能力。 |
| ✍️研究方法 | 引入基于随机权重扰动与贝叶斯期望损失的训练策略，并设计更精细的扰动生成方式来追求平坦极小点。 |
| 🕊️研究对象 | 各种大规模预训练模型的微调场景，包括自然语言处理与图像分类等任务。 |
| 🔍研究结论 | 实验证明在多种架构与数据集上方法有效，可在保持高效训练的同时获得更好的泛化性能。 |
| ⭐创新点   | 1) 将“平坦化”目标扩展至完整参数空间而非仅 LoRA 子空间；2) 采用随机扰动 + 贝叶斯期望损失替代传统 Sharpness-Aware Minimization；3) 提出更精细的扰动生成策略，在不显著增加计算或内存成本的同时提升泛化表现。 |



## **方法详细分析**

下面围绕 Flat-LoRA 方法的目的、主要步骤与实验流程进行更详尽的介绍和分析。

---

## 1. 方法提出的目的

1. **LoRA 子空间与完整参数空间的差异**  
   - LoRA（Low-Rank Adaptation）只在低秩矩阵上进行更新，降低了显存和计算开销；然而，当训练结束后增量权重 $\Delta W$ 会被合并到完整模型 $W$ 中，若只在 LoRA 参数子空间看似“平坦”但在完整权重空间存在锐利方向，则会导致泛化性能下降。

2. **显式提高模型在完整权重空间的平坦度**  
   - 与传统的 Sharpness-Aware Minimization（SAM）不同，**Flat-LoRA** 寻求在不重复大批量梯度计算的前提下，让合并后的权重 $W + \Delta W$ 落入更平坦的极小点区域，从而提升泛化。

3. **避免巨大额外代价**  
   - 直接在大模型上运用 SAM 常需双倍的前向-反向运算与额外复制整套模型参数。Flat-LoRA 采用随机扰动加“贝叶斯期望损失”的形式，实现对平坦度的近似惩罚，从而保持了训练效率与内存友好性。

---

## 2. 方法主要步骤

### 2.1 基本 LoRA 框架

- **低秩增量**  
  假设某层主干权重为 $W \in \mathbb{R}^{d \times k}$（冻结），LoRA 的训练部分为：
  $$
  \Delta W = s \cdot B A, 
  $$
  其中 $A \in \mathbb{R}^{r \times k},\, B \in \mathbb{R}^{d \times r}$，秩 $r \ll \min(d,k)$，并带有缩放因子 $s$。推理时，合并 $W' = W + \Delta W$。

### 2.2 在完整权重上的平坦化目标

1. **期望损失的定义**  
   - Flat-LoRA 引入一个随机扰动 $\epsilon$，作用在合并权重 $W + s \cdot BA$ 上：  
     $$
     W' = W + s \cdot BA + \epsilon.
     $$
   - 优化目标为最小化以下期望（近似“平坦”区域）：
     $$
     \min_{A,B} \; \mathbb{E}_{\epsilon \sim \mathcal{N}(0,\sigma^2 I)} \Bigl[ L (W + s \cdot BA + \epsilon) \Bigr].
     $$

2. **随机扰动生成策略**  
   - 作者在每个滤波器 (filter) 维度上根据权重范数分配噪声，并有 $\frac{1}{n}$ 归一化因子以保证扰动幅度不随输入维度成比例增加：
     $$
     \epsilon \sim \mathcal{N}\!\Bigl(0,\;\frac{\sigma^2}{n}\,\text{diag}\bigl(\|W'_1\|^2,\ldots,\|W'_m\|^2\bigr)\,\mathbf{I}_{m\times n}\Bigr).
     $$
   - 这样能在不显著增大计算量的前提下，提升随机扰动的有效性并获得对真实曲率更好的估计。

3. **一次前向-反向即可**  
   - 生成 $\epsilon$ 后，计算 $L(W + s \cdot BA + \epsilon)$ 的梯度，针对 LoRA 参数 $(A, B)$ 更新；无需像 SAM 那样再次回到原权重再计算梯度。

---

## 3. 细节设置与小提示 (Tips)

1. **扰动强度 $\sigma$ 及其日程**  
   - 论文中常采用一个从零到一定值（如 0.05 / 0.15）的“余弦增加策略”或类似调度，使训练初期扰动较小、后期加大以避免陷入尖锐极小点。

2. **LoRA Rank 选择**  
   - 常见 $r=8$ 或 $r=16$；若任务规模较小或容易过拟合，设置太高可能增大过拟合风险，但 Flat-LoRA 的平坦化目标可在此情况下获得更大收益。

3. **保存内存的小技巧**  
   - 只需存储随机种子并在反向传播后恢复权重，不必保存整套扰动；对于大模型可以结合混合精度，直接在 BF16 权重上做随机注入。

4. **实现便利性**  
   - 该方法与任意 LoRA 改进（如更好的初始化、可变秩分配）不冲突，可以直接将“随机扰动 + 期望损失”模块添加到 LoRA 训练循环中。

---

## 4. 使用的数据集与衡量指标

论文主要在两个方向进行了验证：

1. **自然语言处理 (NLP)**  
   - **GLUE 子集**：MNLI、SST2、CoLA、QNLI、MRPC  
   - 基础模型：T5-Base (约 220M 参数)  
   - 评价指标：**Accuracy**（以及部分场景下根据数据集也可用 F1 等指标）  
   - 训练细节：往往使用 10-epoch 或略高 epoch，学习率多采用 0.0005 左右并加 weight decay。  

2. **图像分类**  
   - 数据集：CIFAR-10 / CIFAR-100、Cars、SVHN、DTD 等  
   - 基础模型：CLIP ViT-B/32 (约 88M 参数)  
   - 主要度量：**Top-1 Accuracy**  
   - rank: 8 / 16；epoch: 10；扰动强度如 $\sigma = 0.15$。  

---

## 5. 得到的结果

1. **NLP 结果**  
   - 与纯 LoRA 相比，Flat-LoRA 在 T5-Base 上获得平均 +0.3% ~ +0.6% 的精度提升，在小规模数据（如 MRPC）和容易过拟合的数据（CoLA）上表现尤为明显。

2. **图像分类**  
   - CLIP ViT-B/32 微调：相对纯 LoRA，平均有约 +0.5% ~ +0.7% 的准确率增益。有时 Flat-LoRA(rank=8) 即可超过纯 LoRA(rank=16) 或完整微调。

3. **与其他 LoRA 改进结合**  
   - 可叠加在现有的排名分配、初始化、梯度对齐等方案之上，带来额外收益（+0.3% ~ +1% 范围）。

---

## 6. 需要注意的要点

1. **扰动不会过度加大训练不稳定性吗？**  
   - 通过对 $\sigma$ 的渐进调度和滤波器归一化处理，随机噪声在训练早期并不会导致梯度暴增。

2. **是否损失训练效率？**  
   - 每步只需一次正向+反向传播，并在混合精度下不用存额外完整权重。相比 SAM 的两次梯度计算方案更为高效。

3. **过拟合问题**  
   - 在 rank 较高或数据量有限时，LoRA 容易过拟合。Flat-LoRA 有助于对高曲率敏感方向进行抑制，从而缓解过拟合。

---



以下内容基于对论文中“通过随机扰动在完整权重空间中逼近平坦极小点”这一思路的理解，来回答两个核心问题：

------

## **问题 1：在训练中加入噪声，为什么能找到或逼近平坦极小点？**

1. **噪声注入如何起到“平滑”作用**

   - 论文中的关键做法是，在训练时对于合并后的完整权重 $W + s \cdot BA$ 注入随机扰动 $\epsilon$，并最小化
     $$
        \mathbb{E}_{\epsilon \sim \mathcal{N}(0,\sigma^2 I)} \bigl[L(W + s \cdot BA + \epsilon)\bigr].
     $$
     
   - 这等价于**对损失函数进行“局部平滑/卷积”**：若某一点在局部方向上非常陡峭，则小扰动 ϵ\epsilon 就会导致损失大幅上升；在期望损失下，这种陡峭方向会受到更强惩罚，从而推着训练过程远离尖锐区域。
   
2. **与传统的 SAM、SGD 随机性对比**

   - **SAM**：显式地做 $\max_{\|\epsilon\|\le\rho} L(W + \epsilon)$，然后再 $\min_w$，强调“最坏扰动”以逼近平坦极小点，但需要“双倍梯度计算”。
   - **纯 SGD**：小批量噪声主要来自采样/数据噪声，并不一定能显式惩罚局部方向上的高曲率。
   - **随机加权扰动 (RWP)**：这里不是只对 LoRA 子空间加噪，而是对**合并后**的完整权重加噪；类似“熵 - SGD”、“局部熵最小化”或其它随机平滑方法，实测可有效抑制陡峭峰谷。

3. **理论直觉**

   - 通过随机扰动 ϵ\epsilon 给损失做“平滑近似”，若某区域曲率过高，微小扰动将引发损失剧增，期望损失也会变大，训练会自然倾向曲率更低、相对平坦的区域。
   - 这与“SGD 带噪声能帮助逃离鞍点”并不冲突，只是这里的噪声更**系统**、更**面向全权重**地注入，显式地去最小化期望损失，从而更可靠地寻找平坦极小点。

------

## **问题 2：为何能在“完整空间”上实现平坦，而不仅是 LoRA 子空间？**

1. **LoRA 子空间 vs. 完整权重空间**

   - 传统 LoRA 仅更新低秩矩阵 $A,B$，对主干 $W$不做大改动；如果只在 $(A,B)$ 上进行某种“尖锐度感知”，则可能在合并后 W + s \cdot BA$ 对“LoRA 子空间外”的方向依旧陡峭。
   - **Flat-LoRA** 里的 ϵ\epsilon 是加在**最终合并权重** $(W + s \cdot BA)$ 上，而不是只加在 $(A,B)$ 上；换言之，所加的噪声覆盖了原始完整维度，能反映整个网络在真实的高维参数空间里是否对扰动敏感。

2. **期望损失的目标范围**

   - 论文的优化目标是 
     $$
      \min_{A,B} \; \mathbb{E}_{\epsilon}\bigl[L(W + s \cdot BA + \epsilon)\bigr], 
     $$
     而不是 $\mathbb{E}_{(\delta_A,\delta_B)}[L(W + (A+\delta_A)(B+\delta_B))]。$

   - 这意味着：如果 **完整权重空间** 某些方向特别“尖锐”，随机抽到该方向的 ϵ\epsilon 就会让损失飙升，进而增加期望损失。训练过程会“避开”此种尖锐解，从而推动最终收敛在**全空间**范围内较平坦的区域。

3. **总结性直觉**

   - 加噪声的操作并非限制在 LoRA 更新矩阵的子空间，而是对模型完整权重应用随机扰动。由此，模型在全权重空间对参数扰动的敏感度也被纳入到训练中，并得到显式“惩罚”，达到“在完整空间平坦”的效果。

------

### **小结**

- **1) 添加噪声如何保证 flat minima？**
   主要因为在每步训练中，对损失函数进行“局部平滑”或“随机卷积”，使得过度尖锐的区域代价变高，训练自然倾向曲率更低的地方。
- **2) 本文如何实现“在整个空间而不是仅仅在低秩空间上的平坦”？**
   通过直接在合并后参数 $W + s \cdot BA $上施加完整维度的随机扰动，不再局限于 LoRA 子空间。这样，若解在原始权重空间存在“锐利方向”，随机扰动就会让损失显著上升，从而被优化过程“排斥”，最终得到对完整权重空间都更加平坦的解。













# Analyse





# Paper1: Towards Understanding Sharpness-Aware Minimization

| **作者:** Maksym Andriushchenko; Nicolas Flammarion;         |
| ------------------------------------------------------------ |
| **期刊: , 2022.**                                            |
| **期刊分区:**                                                |
| **本地链接:** [Towards Understanding Sharpness-Aware Minimization_2022_.pdf](zotero://open-pdf/0_NWP5IFZX) |
| **DOI:** [10.48550/arXiv.2206.06232](https://doi.org/10.48550/arXiv.2206.06232) |
| **摘要:** *Sharpness-Aware Minimization (SAM) is a recent training method that relies on worst-case weight perturbations which significantly improves generalization in various settings. We argue that the existing justifications for the success of SAM which are based on a PAC-Bayes generalization bound and the idea of convergence to flat minima are incomplete. Moreover, there are no explanations for the success of using $m$-sharpness in SAM which has been shown as essential for generalization. To better understand this aspect of SAM, we theoretically analyze its implicit bias for diagonal linear networks. We prove that SAM always chooses a solution that enjoys better generalization properties than standard gradient descent for a certain class of problems, and this effect is amplified by using $m$-sharpness. We further study the properties of the implicit bias on non-linear networks empirically, where we show that fine-tuning a standard model with SAM can lead to significant generalization improvements. Finally, we provide convergence results of SAM for non-convex objectives when used with stochastic gradients. We illustrate these results empirically for deep networks and discuss their relation to the generalization behavior of SAM. The code of our experiments is available at https://github.com/tml-epfl/understanding-sam.* |
| **标签:**                                                    |
| **笔记日期:** 2025/2/22 20:46:40                             |

## 📜 研究核心

------

> Tips: 做了什么，解决了什么问题，创新点与不足？



### ⚙️ 内容



### 💡 创新点



### 🧩 不足



## 🔁 研究内容

------



### 💧 数据



### 👩🏻‍💻 方法



### 🔬 实验



### 📜 结论



## 🤔 个人总结

------

> Tips: 你对哪些内容产生了疑问，你认为可以如何改进？



### 🙋‍♀️ 重点记录



### 📌 待解决



### 💭 思考启发




**ChatGPT**

### 论文分析：《Towards Understanding Sharpness-Aware Minimization》

#### 1. 研究背景与范围 (Scope or Setting)

本论文研究 **Sharpness-Aware Minimization (SAM)** 这一优化方法，并探讨其在深度学习中的**隐式偏差** (implicit bias) 以及其对**泛化性能** (generalization performance) 的影响。现有关于 SAM 的研究主要从 **PAC-Bayes 泛化界** 和 **平坦极小值 (flat minima) 收敛性** 角度进行解释，但作者认为这些解释是不完整的。本研究旨在**更深入地理解 SAM 的成功原因**，并提出新的理论支持。

#### 2. 主要思想 (Key Idea)

本论文的核心观点是：

1. **现有的理论解释不够完善**：基于 PAC-Bayes 泛化界的推导不能充分解释 SAM 的成功，特别是为什么 SAM 选择的**局部最小值**比标准梯度下降 (SGD) 更具泛化能力。
2. **m-SAM 提供更好的泛化能力**：m-SAM 选择的极小值比标准 ERM (Empirical Risk Minimization) 以及 n-SAM 更具优势。
3. **SAM 的隐式正则化效应 (Implicit Regularization)**：在**对角线线性网络 (diagonal linear networks)** 设定下，SAM 对解的选择具有独特的隐式偏差，能够**偏向稀疏解 (sparse solutions)**，从而提高泛化能力。
4. **SAM 的优化过程能够收敛**：论文证明了 SAM 在**非凸目标函数**下的收敛性，并验证其优化过程中是否会收敛到良好的极小值。

#### 3. 研究方法 (Method)

作者采用以下研究方法：

1. **理论分析**：
   - 研究 SAM 在**对角线线性网络**中的隐式偏差 (implicit bias)。
   - 分析 SAM 选择的解在**稀疏正则化** (sparse regularization) 方面的影响，并比较 1-SAM 和 n-SAM 在优化目标上的不同表现。
   - 研究 SAM 在非凸优化中的收敛性 (convergence)。
2. **实验分析**：
   - **在深度神经网络上实证分析** SAM 的泛化能力，使用 ResNet-18 训练 CIFAR-10、CIFAR-100 数据集，并比较不同方法 (ERM, SAM, m-SAM) 在测试集上的误差。
   - **探索 SAM 的最优参数选择**，研究 perturbation radius ($\rho$) 对训练过程和最终泛化性能的影响。
   - **对比不同优化策略**，例如 ERM $\to$ SAM 以及 SAM $\to$ ERM，以分析 SAM 在不同训练阶段的作用。

#### 4. 贡献 (Contribution)

论文的主要贡献包括：

1. **挑战现有 SAM 理论**：指出现有基于 PAC-Bayes 泛化界的理论无法充分解释 SAM 的成功，并提供新的解释框架。
2. **揭示 m-SAM 的独特性**：通过实验证明 m-SAM 相比于 n-SAM 和 ERM 具有更好的泛化能力，并分析其影响因素。
3. **研究 SAM 的隐式偏差**：在理论上证明 SAM 会隐式地选择稀疏解 (sparse solutions)，并通过实验验证这一现象。
4. **提供 SAM 的收敛性分析**：证明了 SAM 在非凸优化问题上的收敛性，并讨论了 SAM 在优化路径上的特性。

#### 5. 结果与结论 (Result & Conclusion)

1. **m-SAM 优于 ERM 和 n-SAM**：实验表明，在 CIFAR-10 和 CIFAR-100 上，m-SAM 训练的模型测试误差更低。
2. **SAM 选择的解具有较好的隐式偏差**：相比标准 ERM，SAM 训练的模型更倾向于选择具有较小 $\ell_1$ 规范的解，从而提高泛化能力。
3. **SAM 具有良好的收敛性**：在非凸优化问题上，SAM 仍然可以保证收敛，且不会影响最终的泛化能力。
4. **SAM 的优化路径很重要**：在训练后期使用 SAM 进行 fine-tuning 仍然可以显著改善泛化能力，而在训练早期使用 SAM 后改回 ERM 则无法保持这种优势。

#### 6. 与其他方法的对比 (Contrast with Other Methods)

| 方法              | 优势                                 | 劣势                                |
| ----------------- | ------------------------------------ | ----------------------------------- |
| ERM               | 计算量小，适用于标准训练             | 容易陷入 sharp minima，泛化能力较弱 |
| n-SAM             | 计算量比 ERM 大，能减少 sharp minima | 泛化能力未必比 m-SAM 好             |
| m-SAM             | 计算量适中，能显著提高泛化能力       | 需要调整 m 以平衡计算效率和性能     |
| SAM + Fine-tuning | 可以从已有模型提升泛化               | 需要额外训练时间                    |

#### 7. 讨论与未来研究 (Discussion & Future Work)

1. **如何进一步优化 SAM 的计算效率？** 目前 SAM 计算复杂度较高，如何降低计算量仍然是一个研究方向。
2. **如何更精确地分析 SAM 的隐式正则化效应？** 现有的理论分析主要针对对角线线性网络，未来可以扩展到更复杂的神经网络结构。
3. **如何结合 SAM 与其他优化方法？** 例如结合 SWA (Stochastic Weight Averaging) 或 LoRA (Low-Rank Adaptation) 以进一步提高泛化能力。

------

### 训练模型架构与数据集

| 模型      | 数据集    | 训练目标     | 输入       | 输出           | 训练过程中更新的部分 |
| --------- | --------- | ------------ | ---------- | -------------- | -------------------- |
| ResNet-18 | CIFAR-10  | 预测图像类别 | 32×32 图像 | 10 维类别概率  | 权重参数             |
| ResNet-34 | CIFAR-100 | 预测图像类别 | 32×32 图像 | 100 维类别概率 | 权重参数             |

------

本研究为 SAM 提供了新的理解框架，并通过实验验证了其泛化能力的提升。未来研究可以探索如何提高 SAM 的计算效率，并结合其他优化方法进一步提升深度神经网络的性能。







# Paper2: Why Does Sharpness-Aware Minimization Generalize Better Than SGD?





## **1. 论文内容概括**

## 1.1 研究范围与背景 (Scope / Setting)
本论文**“Why Does Sharpness-Aware Minimization Generalize Better Than SGD?”**，主要探讨在深度学习中，当网络规模愈发庞大时，**过拟合**（overfitting）现象趋于严重。然而，近来提出的 **Sharpness-Aware Minimization (SAM)** 方法在许多实际任务中表现出显著的泛化优势，即使在噪声标签场景下也能保持良好性能。本论文旨在从理论上揭示：在什么情况下，**SAM** 在非线性神经网络与分类任务中能够胜过传统的 **SGD**，并给出了相应的收敛分析和泛化误差结果。

- **设定**：作者聚焦于**二分类**场景，数据来自具备“信号+噪声”结构的分布：输入特征可拆分为若干“patch”，其中仅有一个 patch 含有真实信号，其余 patch 均为噪声。并研究一个**两层卷积 ReLU 网络**（具有共享卷积核、第二层固定输出权重），训练目标是最小化交叉熵损失。
- **挑战**：网络**非平滑（ReLU激活）**、存在**标签翻转噪声**、数据维度高、以及 mini-batch SGD 训练下的参数不平衡更新，都为理论分析带来显著难度。

## 1.2 核心思想 (Key Idea)
1. **刻画 SGD 在卷积 ReLU 网络的过拟合转折**  
   论文首先在具备“信号+噪声”的数据分布中，精确给出了 **SGD** 出现“良性过拟合”（benign overfitting）或“有害过拟合”（harmful overfitting）的条件。若信号强度足够大（满足一定阈值），则 SGD 能记住所有训练数据并在测试集上逼近贝叶斯错误率；否则，就会严重陷入噪声学习，测试误差无法降低。
2. **证明 SAM 能在更弱信号条件下仍实现良性过拟合**  
   论文进一步给出结论：当 SGD 出现“有害过拟合”的同等条件下，**SAM** 却能够有效抑制噪声的记忆，并利用更少的信号强度就达到与 SGD 在强信号情形下才可获得的泛化水平，即“严格优于”SGD。
3. **机制探究**  
   通过对损失曲面中激活模式的精细分析（ReLU 导致非平滑），作者指出 **SAM** 在每一次更新时插入权重扰动，会主动把原本会学习到噪声的神经元“关掉”，从而减少网络对噪声的过度拟合。

## 1.3 方法与贡献 (Method & Contribution)
- **方法**：
  - 模型：构建一个两层卷积 ReLU 网络，第二层固定正负权重，适用于可拆成多个 patch 的输入。  
  - 数据分布：信号向量 + 若干噪声向量，每次只随机选一个 patch 带信号（带标签），其余为噪声。允许一定比例标签翻转。  
  - 训练：分别对 **SGD** 与 **SAM** 进行理论分析，对比二者在“完全记忆训练数据”后，测试误差是否能逼近贝叶斯风险 $p$。
- **贡献**：
  1. **首个对 mini-batch SGD 在两层非平滑 CNN 上实现（或不实现）良性过拟合的严格刻画**，给出相应阈值：若 $\|\mu\|^2$（信号强度）足够大，则 SGD 过拟合无害；反之，“有害过拟合”不可避免。  
  2. **理论区分 SAM 与 SGD**：在相同信噪比设定下，SGD 可能陷入“噪声学习”，但 SAM 依靠“梯度前添加扰动”策略有效阻断了噪声激活的神经元，从而在更低信号条件下也能实现良好泛化。  
  3. **非光滑分析**：由于 ReLU 网络的 Hessian 信息不再适用简单光滑假设，文章创新性地从卷积参数的“信号-噪声分解系数”及激活模式入手，克服了非平滑带来的分析障碍。

## 1.4 差异与创新 (Difference & Innovation)
1. **此前对 SAM 的理论多局限于平滑假设**（比如基于 Hessian 的推导），而本文明确指出在 **ReLU 非平滑**场景下，Hessian 分析不足，需全新方式。  
2. **mini-batch SGD 分析难度更高**，因部分 batch 内噪声样本可能占大多数；作者提出了两阶段分析、mini-batch 级别与 epoch 级别结合的激活模式控制技术。  
3. **噪声记忆阻断机制**：论文在理论上揭示 SAM 通过每步扰动，让原本会激活的噪声方向被“翻转”并失活，从而减少噪声拟合。

## 1.5 实验结果与结论 (Result & Conclusion)
- **主要结论**：
  - 当数据维度 $d$ 增大时，SGD 要想完成良性过拟合，需要信号强度 $\|\mu\|_2$ 与 $d$ 一起增长，否则就陷入严重过拟合噪声、测试误差明显高于贝叶斯率；  
  - 而 **SAM** 能在更弱信号下依然学到正确特征，而非记住噪声，在同等维度时泛化显著优于 SGD。  
- **实验**：
  1. 在**合成数据**上（可视化成热力图），显示 SGD 在维度-信号平面中只有部分区域能够良性过拟合，而 SAM 的“良性区域”更大。  
  2. 真实数据（CIFAR、ImageNet）也表明：SAM 在标签噪声或增广噪声场景下，比起 SGD 泛化更稳定。

## 1.6 与其他方法对比、讨论 (Contrast & Discussion)
1. **与先前 SAM 理论**：如 Foret 等人的 PAC-Bayes 界，Bartlett 等的 Hessian 分析，都要求平滑或至少可做局部二阶近似；本文在 ReLU 非平滑条件下提供更具说明力的新视角。  
2. **与已有“良性过拟合”结果**：先前多集中在核方法、线性模型或平滑激活，这里扩展到卷积 ReLU 网络 + mini-batch SGD，并给出具体阈值和激活模式分析手段。  
3. **启示**：论文表明对抗噪声，特别是大规模、非平滑网络训练中，SAM 或其变体极具潜力；也暗示了大步长SGD部分能改进泛化，但仍不及 SAM 的显式“邻域最坏扰动”约束的效果。

## 1.7 理论证明 (Prove)
- **主要定理**：
  - 定理 3.2：刻画 SGD 在信噪比阈值上下的泛化误差表现，出现良性 / 有害过拟合的充要条件。  
  - 定理 4.1：表明 SAM 在维度几乎无关的弱信号条件下也能得到 $\leq p + \varepsilon$ 的测试误差（逼近贝叶斯风险），从而优于 SGD。  
- **关键技术**：
  - **信号-噪声分解**：将卷积过滤器参数拆成学习信号 $\gamma$ 和学习噪声 $\rho$ 两部分，跟踪二者演化；  
  - **激活集分析**：在 ReLU 中，若一次迭代对噪声方向激活，则下次 SAM 会扰动翻转使之非激活；类似地，对真实信号激活保持不变，保证学习到正确特征。  
  - **控制梯度比**：通过 bounding mini-batch 内真实样本、噪声样本数量，及激活神经元的梯度大小比值，来保证特征系数能持续增长、噪声系数受限。

---

## **2. 方法详细分析**

以下重点阐述作者针对 **SAM** 与 **SGD** 的比较性理论方法，以及关键步骤、注意事项等。

## 2.1 目的
- **SGD 分析**：明确在何种信号强度 / 数据维度条件下，SGD 训练的两层卷积网络能够实现良性过拟合（即训练误差 0、测试误差接近贝叶斯率）或遭遇噪声学习失败。  
- **SAM 分析**：揭示为什么在同等条件下，SAM 能通过一种扰动激活机制，阻断噪声学习，让网络专注于真正有效的特征。

## 2.2 主要步骤

### 2.2.1 模型设定
1. **数据分布**：样本 $(x,y)$，其中 $x$ 被拆分成 $P$ 个 patch，每个 patch 要么是信号（$y \cdot \mu$），要么是噪声向量 $\xi$。标签 $y \in \{\pm 1\}$ 可带翻转噪声。  
2. **两层卷积 ReLU 网络**：卷积核数量 $m$，第二层固定 $\pm 1/m$ 的连接方式，输出 $f(W,x) = \sum_{r=1}^m \sigma(\langle w^+_r, x(\cdot)\rangle) - \sum_{r=1}^m \sigma(\langle w^-_r, x(\cdot)\rangle)$。  
3. **训练损失**：交叉熵 $\ell(y\cdot f(W,x))$，最小化其经验和。

### 2.2.2 信号-噪声分解
- 将每个卷积过滤器 $w_{j,r}$ 拆分为：  
  $$
  w_{j,r} \;=\; \underbrace{\gamma_{j,r}\,\|\mu\|^{-2}_2\, \mu}_{\text{信号方向}} \;+\; \sum_{i=1}^n\rho_{j,r,i}\,\|\xi_i\|^{-2}_2\,\xi_i,
  $$
  并分析信号系数 $\gamma_{j,r}$ 与噪声系数 $\rho_{j,r,i}$ 在 SGD 或 SAM 更新下如何演变。

### 2.2.3 SGD 训练
1. **mini-batch 级激活模式**：若卷积核与噪声 $\xi_i$ 内积大于 0，则此过滤器在 batch 中会“学习” $\xi_i$；作者证明当数据维度大、信号强度不够时，噪声学习也会很活跃，导致测试误差无法逼近最优。  
2. **主要结论**（定理 3.2）：如果 $\|\mu\|^4 \gtrsim d\,\sigma_p^4 \,/\, n$，则 SGD 可良性过拟合，测试误差 $\approx p$；否则有害过拟合发生，无法逼近 $p$.  
3. **原因**：由于无机制抑制噪声记忆，SGD 在许多 batch 中被噪声样本牵引，使得 $\rho_{j,r,i}$ 也不断增大，干扰到真正信号的学习。

### 2.2.4 SAM 训练
1. **扰动-翻转神经元**：SAM 每步都会先计算梯度 $\nabla L(W)$，然后产生 $ \epsilon^\ast = \tau \frac{\nabla L(W)}{\|\nabla L(W)\|}$ ，训练用 $W + \epsilon^\ast$；由于这个扰动改变了 ReLU 的激活模式，使得原本打算“学习噪声”的神经元激活在实际更新时被翻转关掉，阻断了噪声系数增长。  
2. **主要结论**（定理 4.1）：在更低的信号要求下（维度几乎无关），SAM 同样能在较少步数内达到训练误差极小、测试误差 $\le p + \varepsilon$.  
3. **机制**：通过引入 $\epsilon^\ast$，让 $\langle w_{j,r}, \xi_i \rangle > 0$ 的神经元在 perturbed 权重下变成 $\langle w_{j,r} + \epsilon^\ast_{j,r}, \xi_i \rangle < 0$，减少了对噪声梯度的吸收，最终噪声系数 $\rho_{j,r,i}$ 保持 $O(1)$ 而不随迭代增长。

## 2.3 实验设计与结果
1. **合成数据**：对比 SGD / SAM 在维度 $d$、信号强度 $\|\mu\|$ 构成的网格上训练后测试误差的分布，得到热力图；观察到 SAM“良好区域”更大，符合理论。  
2. **真实数据（补充）**：作者在 CIFAR、ImageNet 上测试把 SAM 用于中期或全程训练，与 SGD 对比，在噪声场景下 SAM 减少噪声学习、提升准确率。

## 2.4 一些小提示 (Tips)
- **学习率与批量大小**：若学习率过大，SGD 也可部分改善泛化，但仍无法完全阻断噪声；SAM 则稳定地通过最坏方向扰动来控制激活模式。  
- **初始权重大小 $\sigma_0$**：论文要求其随 $d$ 缩放，以便初期不会对噪声过度敏感；SAM 同样需在合适 $\tau$ 范围内运作。  
- **噪声翻转率 $p$**：若 $p$ 过大（近于 50%），则可学习信号十分困难；本文理论要求 $p < 1/2$。  
- **非平滑激活**：二阶信息或 Hessian-based 分析难以直接适用，本论文用“系数分解 + 激活集 + batch 分块”方法取而代之。

---



## **论文中的定理分析：Benign Overfitting 与 Harmful Overfitting**

本文深入分析了 **SGD** 与 **SAM** 在训练神经网络（特别是卷积 ReLU 网络）时的不同泛化行为，并通过理论分析给出了 **Benign Overfitting**（良性过拟合）和 **Harmful Overfitting**（有害过拟合）的明确区分。以下将对相关定理进行详细介绍。

---

## **1. Benign Overfitting 与 Harmful Overfitting 定义**
- **Benign Overfitting（良性过拟合）**  
  指的是模型虽然能够完美拟合训练数据（包括噪声样本），但其测试误差仍然可以接近 Bayes 最优风险，即 **模型能够记住数据但不会过度学习噪声**。  
- **Harmful Overfitting（有害过拟合）**  
  指的是模型不仅能记住训练数据，同时也在测试数据上表现出较高的错误率，即 **模型对噪声样本的学习削弱了泛化能力**。

在本文的分析框架下，信号强度 $\|\mu\|$ 与数据维度 $d$ 之间的比例决定了 SGD 是否会陷入有害过拟合，而 SAM 能在更广泛的条件下实现良性过拟合&#8203;:contentReference[oaicite:0]{index=0}。

---

## **2. 定理 3.2：SGD 的 Benign / Harmful Overfitting**
### **定理 3.2（Benign/Harmful Overfitting of SGD in CNNs）**
**结论**：
对于任意 $\epsilon > 0$，在一定正则性条件下，满足以下公式的概率至少为 $1 - \delta$：
1. **SGD 训练损失收敛**：
   $$
   L_S(W^{(t)}) \leq \epsilon.
   $$
2. **良性过拟合条件**：  
   若满足信号强度足够大：
   $$
   n\|\mu\|^4_2 \geq C_1 d P^4 \sigma^4_p,
   $$
   则测试误差满足：
   $$
   L_{0-1}(W^{(t)}) \leq p + \epsilon.
   $$
   这意味着 SGD 在信号足够强的情况下可以泛化良好。
3. **有害过拟合条件**：  
   若信号较弱：
   $$
   n\|\mu\|^4_2 \leq C_3 d P^4 \sigma^4_p,
   $$
   则测试误差较大：
   $$
   L_{0-1}(W^{(t)}) \geq p + 0.1.
   $$
   这意味着 SGD 过度拟合噪声，测试误差远高于 Bayes 风险。

### **解析**
- 该定理刻画了 **SGD 泛化能力的一个相变现象（phase transition）**。当信号 $\|\mu\|_2$ 足够强时，SGD 仍可实现良性过拟合；但当 $\|\mu\|_2$ 低于某个临界值时，SGD 便会陷入噪声学习，导致泛化性能恶化&#8203;:contentReference[oaicite:1]{index=1}。

---

## **3. 定理 4.1：SAM 的 Benign Overfitting**
### **定理 4.1（SAM 的良性过拟合）**
**结论**：
对于任意 $\epsilon > 0$，在满足一定正则条件下，SAM 训练能以概率 $1 - \delta$ 找到权重 $W^{(t)}$，使得：
1. **训练损失收敛**：
   $$
   L_S(W^{(t)}) \leq \epsilon.
   $$
2. **测试误差满足**：
   $$
   L_{0-1}(W^{(t)}) \leq p + \epsilon.
   $$

### **与 SGD 结果的关键区别**
- **SAM 仅需信号强度满足**：
  $$
  \|\mu\|_2 \geq \Omegã(1),
  $$
  即 **SAM 对信号强度的要求几乎不随维度 $d$ 变化**，从而能在更多情形下实现良性过拟合&#8203;:contentReference[oaicite:2]{index=2}。
- 而 SGD 需要信号强度达到 $\Omega(d^{1/4})$ 以上才能避免有害过拟合。

### **解析**
- 该定理揭示了 **SAM 通过扰动优化** 的方式能够有效减少噪声的学习，并确保模型在更低的信号强度下仍能达到良好泛化。
- 这表明 **SAM 的泛化能力远优于 SGD**，尤其在高维情况下，SGD 的信号强度要求大幅增加，而 SAM 依然能稳定泛化&#8203;:contentReference[oaicite:3]{index=3}。

---

## **4. 关键现象解析**
### **4.1 为什么 SAM 泛化更好？**
- **SGD 存在的问题**：  
  SGD 训练时，梯度更新不加任何约束，容易让噪声权重和信号权重同步增长，从而导致有害过拟合。
- **SAM 解决方案**：
  
  - SAM 每次更新前 **先计算梯度方向的扰动**：
    $$
    \epsilon = \tau \frac{\nabla L(W)}{\|\nabla L(W)\|}.
    $$
  - 通过这样的扰动，SAM **主动抑制了噪声梯度的影响**，使得网络能更专注于学习信号。

### **4.2 何时使用 SAM 优于 SGD？**
- **当数据高维且信号强度较弱时**，SGD 的泛化性能较差，而 SAM 仍能实现良性过拟合。
- **当数据包含噪声标签时**，SAM 通过其扰动机制减少对噪声样本的学习，从而提高泛化能力。

---

## **5. 结论总结**
- **SGD 的泛化能力高度依赖于信号强度 $\|\mu\|_2$，若信号不够强，SGD 可能会过度学习噪声，导致有害过拟合**。
- **SAM 通过扰动优化降低了噪声的影响，使得在低信号条件下依然能达到良性过拟合，泛化能力远优于 SGD**。
- **SAM 在理论上具备更严格的泛化保证，并在实践中展现了优异的泛化能力，特别适用于高维数据和有噪声标签的任务**。

---









# Paper3:  When Do Flat Minima Optimizers Work?

## 1. 研究范围 (Scope)

该论文研究神经网络优化中的**平坦极小值 (Flat Minima) 优化方法**，主要关注**随机权重平均 (SWA)** 和**敏锐度感知最小化 (SAM)** 这两种优化方法，并在计算机视觉 (CV)、自然语言处理 (NLP) 和图表示学习 (GRL) 任务中进行对比实验。

## 2. 研究目的 (Purpose)

论文旨在系统性地分析**SWA** 和**SAM** 的特性，并通过大规模基准测试来评估它们的有效性，帮助研究人员理解它们的优缺点，并为实践者选择适当的优化方法提供指导。

## 3. 关键思想 (Key Idea)

- 传统的**随机梯度下降 (SGD)** 方法容易收敛到**尖锐极小值 (Sharp Minima)**，这可能会导致泛化能力较差。
- **平坦极小值 (Flat Minima)** 方法（如 SWA 和 SAM）通过优化过程引导模型找到较低损失但更稳定的解，从而提高泛化能力。
- **SWA** 通过对多个优化点的权重进行平均，使模型趋向更平坦的区域，而 **SAM** 通过显式地最小化最大局部损失来实现平坦优化。

## 4. 方法 (Method)

- **SWA 方法**:

  - 通过权重平均减少损失曲面中的陡峭区域。

  - 采用移动平均方式计算最终参数：
    $$
    \theta_{\text{SWA}} = \frac{1}{N} \sum_{i=1}^{N} \theta_i
    $$
    

- **SAM 方法**:

  - 通过最大化局部邻域损失来找到平坦区域：
    $$
    \min_{\theta} \max_{\|\epsilon\| \leq \rho} L(\theta + \epsilon)
    $$
    
  - 近似计算扰动 $\epsilon$ 以进行参数更新：
    $$
    \epsilon^* = \rho \frac{\nabla L(\theta)}{\|\nabla L(\theta)\|}
    $$
    

## 5. 主要贡献 (Contributions)

- **首次系统性比较 SWA 和 SAM 的优化效果**：
  - 可视化线性插值，分析不同优化方法的损失曲面特性。
  - 量化不同优化方法找到的极小值的平坦度。
- **跨 42 个任务的实验基准测试**：
  - 涵盖 CV、NLP 和 GRL 领域。
  - 采用不同模型架构（MLPs, CNNs, Transformers）。
  - 任务包括分类、自然语言理解、图节点预测等。
- **发现并提出关键观察结果**：
  - 在 NLP 任务中，SAM 优于 SWA。
  - 在 GRL 任务中，SWA 通常比 SAM 更有效。
  - **SWA 结合 SAM (WASAM) 在多个任务中取得最优效果**。

## 6. 论文的创新点 (Difference & Innovation)

- **与现有研究的不同之处**:
  - 现有工作主要关注**个别任务或特定领域**，缺乏跨领域的系统性研究。
  - **SWA 和 SAM 以往研究大多独立进行**，而该论文首次将二者进行直接对比。
- **创新点**:
  - 提出**权重平均 SAM (WASAM)** 方法，结合 SWA 和 SAM 优势，实现更稳定的泛化能力。
  - 通过 Hessian 特征值分析极小值的平坦性，提供**优化曲面几何解释**。

## 7. 主要实验结果 (Results)

- **SWA vs. SAM**：
  - SAM 通常在 NLP 任务中表现更好，而 SWA 在图学习任务中更优。
  - **SAM 发现的极小值更平坦**，但可能更接近损失曲面的尖锐方向。
- **SWA + SAM (WASAM) 方法**：
  - WASAM 结合 SWA 和 SAM，取得最佳泛化效果。
  - 在 42 个任务中，39 个任务 WASAM 表现优于 SWA 和 SAM 单独使用。

## 8. 结论 (Conclusion)

- **优化器的有效性取决于任务类型和模型架构**:
  - **在 NLP 任务中，SAM 更具优势**，因为其优化策略适用于 Transformer 结构。
  - **在图学习任务中，SWA 更有效**，其平滑更新策略在 GNNs 上表现更优。
- **提出新型优化方法 (WASAM)**：
  - 结合 SWA 和 SAM 的优势，改善泛化性能。
  - 未来研究可以探索**如何进一步优化 SWA 和 SAM 的超参数**，以及**它们在强化学习或生成模型中的应用**。

------

## 论文实验基础模型和数据集

| 任务类别           | 数据集                                                       | 基础模型                                               | 规模                                    |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------ | --------------------------------------- |
| 计算机视觉 (CV)    | CIFAR-10, CIFAR-100, ImageNette                              | WideResNet-28-10, PyramidNet-272, ViT-B-16, Mixer-B-16 | WRN (28层, 10倍宽度), ViT (Base, 16x16) |
| 自监督学习 (SSL)   | CIFAR-10, ImageNette                                         | MoCo, SimCLR, SimSiam, BarlowTwins, BYOL, SwAV         | ResNet-18 作为 backbone                 |
| 自然语言处理 (NLP) | Natural Questions (NQ), TriviaQA, GLUE (COLA, SST, MRPC, STSB, QQP, MNLI, QNLI, RTE) | FiD-base (T5-based), RoBERTa-base                      | RoBERTa (Base), T5 (Base)               |
| 图表示学习 (GRL)   | OGB-Proteins, OGB-Products, OGB-Code2, OGB-Molpcba, Biokg, Citation2 | GCN, SAGE, GIN, DeeperGCN, ComplEx, CP                 | 适用于图任务的 GNN 模型                 |

------

以上内容提供了对论文的全面分析，并以 M



## 文中如何量化 flatness



### **方法、目的与原因解析**

论文在这一部分使用了 **幂迭代算法 (Power Iteration Algorithm)** 计算 **主 Hessian 特征值的中位数 (Median of the Dominant Hessian Eigenvalue)**，用来量化不同优化方法 (SWA、SAM、WASAM 和基线方法) 找到的极小值的平坦性 (flatness)。下面我们详细解析这段话所涉及的方法、目的以及使用该指标的原因。

------

## **1. 方法 (Method): 幂迭代算法计算 Hessian 最大特征值**

论文采用 **幂迭代算法 (Power Iteration Algorithm)** 来估计 Hessian 矩阵的最大特征值。这是因为对于深度神经网络，完整计算 Hessian 矩阵的所有特征值非常昂贵，幂迭代法是一种高效的近似计算方法。

### **(1) Hessian 矩阵**

Hessian 矩阵 $H$ 是损失函数 $L(\theta)$ 关于模型参数 $\theta$ 的二阶导数矩阵：
$$
H=∇2L(θ)H = \nabla^2 L(\theta)
$$
它描述了损失函数在参数空间的曲率。

### **(2) 幂迭代法 (Power Iteration Algorithm)**

幂迭代法是一种用于近似计算最大特征值 $\lambda_{\max}$ 的算法。其核心思想是：

- 从一个随机向量 $v_0$ 开始，反复计算： $vt+1=Hvt∥Hvt∥v_{t+1} = \frac{H v_t}{\|H v_t\|} $经过多次迭代，$v_t$ 会逐渐收敛到 Hessian 矩阵的主特征向量，而对应的特征值 $\lambda_{\max}$ 也会收敛。

### **(3) 计算多个 mini-batch 上的 $\lambda_{\max}$ 并取中位数**

- 论文不是仅仅计算单个 batch 的 Hessian 最大特征值，而是：

  1. 在**整个训练集**的多个 mini-batch 上计算 $\lambda_{\max}$。
  2. 取这些最大特征值的**中位数 (median)** 作为最终衡量指标：

  $$
  \text{Median}(\lambda_{\max})
  $$

  

- 这种方法能够更稳定地衡量优化解的平坦性，减少单个 batch 可能带来的噪声。

------

## **2. 目的 (Purpose): 量化优化器找到的极小值的平坦性**

该指标的目的是衡量 **不同优化方法找到的极小值的几何特性**，具体来说：

- 较低的 $\lambda_{\max}$ 代表优化解较平坦：
  - 如果 Hessian 最大特征值较小，说明该点处的损失函数变化较缓慢，参数的小幅扰动不会导致损失剧烈变化，这种极小值被称为 **“平坦极小值 (Flat Minima)”**。
  - 这些优化解通常具有 **更好的泛化性能**，因为它们对测试数据的变化更具鲁棒性。
- 较高的 $\lambda_{\max}$ 代表优化解较陡峭：
  - 如果 Hessian 最大特征值较大，说明损失函数在该点的变化更剧烈，参数的微小扰动可能会导致损失大幅度上升，这种极小值被称为 **“尖锐极小值 (Sharp Minima)”**。
  - 这些解可能会导致 **过拟合 (Overfitting)**，即在训练集上表现良好，但泛化能力较差。

因此，论文通过计算 Hessian 最大特征值的中位数，比较 **SWA、SAM、WASAM 和 SGD 训练得到的极小值的平坦性**，以评估不同优化器的泛化能力。

------

## **3. 选择该指标的原因 (Why This Metric?)**

论文选择 **Hessian 最大特征值的中位数** 作为度量指标的主要原因有以下几点：

### **(1) 该指标能有效衡量损失曲面的曲率**

- **最大 Hessian 特征值** 是最能反映损失曲面陡峭程度的指标之一。较大的值意味着更陡峭的损失曲面，而较小的值意味着更平坦的损失曲面。
- 论文关注的是**优化器如何影响极小值的几何形状**，因此使用该指标来区分**平坦极小值 (Flat Minima)** 和 **尖锐极小值 (Sharp Minima)**。

### **(2) 在极小值平坦性研究中被广泛使用**

- 论文提到，该指标**在极小值平坦性相关研究中非常常见**，许多文献都使用 Hessian 最大特征值来衡量优化解的曲率。
- 例如：
  - **Foret et al. (2021)** 在提出 SAM 方法时，也使用 Hessian 最大特征值作为衡量模型平坦性的指标。
  - **Dziugaite & Roy (2017)** 研究了神经网络的泛化误差，并表明平坦极小值与泛化能力相关。

### **(3) 通过取中位数减少数据噪声**

- 计算 Hessian 最大特征值时，每个 mini-batch 的数值可能会有所不同，单独依赖某个 batch 可能会产生噪声。
- **取中位数 (Median)** 而非均值，能减少异常值的影响，使得最终的平坦性度量更加稳定。

------

## **4. 论文中的实验结果**

论文在两个不同的任务 (CIFAR-100 和 OGB-Code2) 上比较了 SWA、SAM、WASAM 和基线优化器的 Hessian 最大特征值的中位数：

| 任务                 | Baseline (SGD) | SWA   | SAM   | WASAM    |
| -------------------- | -------------- | ----- | ----- | -------- |
| **WRN on CIFAR-100** | 673            | 265   | 237   | **117**  |
| **GIN on OGB-Code2** | 16.65          | 16.79 | 11.31 | **9.96** |

从表中可以看出：

- **SAM 训练的模型比 SWA 训练的模型具有更低的 Hessian 最大特征值中位数**，意味着 SAM 发现的极小值更平坦。
- **WASAM (SWA + SAM) 取得了最平坦的优化解**，表明该方法可以更有效地找到稳定的优化解。

------

## **5. 总结**

论文采用 **幂迭代法 (Power Iteration Algorithm)** 计算 **Hessian 矩阵的最大特征值**，并在多个 mini-batch 上计算该值的中位数，以量化不同优化方法找到的极小值的平坦性。

- **方法**：幂迭代法计算 Hessian 最大特征值，并取其在多个 mini-batch 之间的中位数。
- **目的**：衡量优化器找到的极小值的平坦性，分析优化曲面的几何特性。
- 原因：
  1. **该指标能有效衡量损失曲面的曲率**，是平坦极小值研究中的重要度量。
  2. **在极小值研究中被广泛使用**，如 Foret et al. (2021) 和 Dziugaite & Roy (2017) 也使用该方法。
  3. **取中位数减少噪声，提高稳定性**。

最终实验结果表明：

- **SAM 发现的极小值比 SWA 更平坦**，但可能靠近损失曲面的陡峭方向。
- **WASAM (SWA + SAM) 结合了两者的优点，取得了最优的泛化性能**。



## **实验设置与结果分析**

本部分详细介绍论文的实验设置，包括涉及的领域、基础模型、数据集、优化器的选择、实验的主要设置与探讨内容、实验结果、作者的分析以及结论。

## **1. 实验涉及的领域**

论文在**计算机视觉 (CV)**、**自然语言处理 (NLP)** 和 **图表示学习 (GRL)** 这三个主要领域进行实验，以评估 **SWA**、**SAM** 和 **WASAM (SWA + SAM)** 在不同任务中的表现。

---

## **2. 基础模型 (Base Model) 与数据集 (Datasets)**

### **(1) 计算机视觉 (CV)**

| 任务                                           | 数据集               | 基础模型                                                     |
| ---------------------------------------------- | -------------------- | ------------------------------------------------------------ |
| **监督分类 (Supervised Classification, SC)**   | CIFAR-10, CIFAR-100  | WideResNet-28-10 (WRN), PyramidNet-272, Vision Transformer (ViT-B-16), MLP-Mixer |
| **自监督学习 (Self-Supervised Learning, SSL)** | CIFAR-10, ImageNette | MoCo, SimCLR, SimSiam, BarlowTwins, BYOL, SwAV               |

- **监督学习 (SC)**：在标准图像分类任务上评估优化器的影响。
- **自监督学习 (SSL)**：使用 k-NN 分类评估无标签特征学习方法。

### **(2) 自然语言处理 (NLP)**

| 任务                                                   | 数据集                                             | 基础模型                          |
| ------------------------------------------------------ | -------------------------------------------------- | --------------------------------- |
| **开放域问答 (Open-Domain Question Answering, ODQA)**  | Natural Questions (NQ), TriviaQA                   | Fusion-In-Decoder (FiD, T5-based) |
| **自然语言理解 (Natural Language Understanding, NLU)** | GLUE (COLA, SST, MRPC, STSB, QQP, MNLI, QNLI, RTE) | RoBERTa-base                      |

- **ODQA**：评估优化方法对 Transformer 模型在问答任务中的影响。
- **NLU**：使用 GLUE 基准测试优化方法对预训练语言模型微调的效果。

### **(3) 图表示学习 (Graph Representation Learning, GRL)**

| 任务                                             | 数据集                     | 基础模型               |
| ------------------------------------------------ | -------------------------- | ---------------------- |
| **节点属性预测 (Node Property Prediction, NPP)** | OGB-Proteins, OGB-Products | GraphSAGE, DeeperGCN   |
| **图属性预测 (Graph Property Prediction, GPP)**  | OGB-Code2, OGB-Molpcba     | GCN, GIN, DeeperGCN    |
| **链接预测 (Link Property Prediction, LPP)**     | Biokg, Citation2           | ComplEx, CP, GCN, SAGE |

- **NPP**：预测节点属性，如蛋白质功能。
- **GPP**：预测整个图的属性，如代码摘要任务。
- **LPP**：预测图中的链接关系，如知识图谱链接预测。

---

## **3. 优化器的选择 (Optimizers)**

论文比较了以下四种优化方法：

- **SGD (基线方法)**：
  - 传统的随机梯度下降方法，未使用额外的平坦性优化策略。
- **SWA (随机权重平均, Stochastic Weight Averaging)**：
  - 通过对多次训练迭代的参数进行平均，使优化解偏向于更平坦的极小值。
- **SAM (敏锐度感知最小化, Sharpness-Aware Minimization)**：
  - 通过显式最大化局部损失来找到平坦极小值。
- **WASAM (SWA + SAM, 权重平均敏锐度感知最小化)**：
  - 结合 SWA 和 SAM 的优势，进一步优化模型的泛化能力。

---

## **4. 实验主要设置与探讨内容**

1. **如何衡量优化解的平坦性？**
   - 采用 Hessian 矩阵的最大特征值的中位数 (Median of the Dominant Hessian Eigenvalue) 来量化极小值的平坦性。
   - 计算多个 mini-batch 上的 Hessian 最大特征值，并取中位数，以减少噪声影响。

2. **如何评估优化器的性能？**
   - **计算模型的最终测试集性能 (Accuracy, F1-score, AUC, etc.)**。
   - **对比不同优化方法在不同任务上的表现**，分析优化器是否能提高泛化能力。

3. **数据集、任务和模型架构是否影响优化方法的有效性？**
   - 在不同数据集和任务上对比 SWA、SAM 和 WASAM 的效果，分析数据和架构对优化方法的影响。

---

## **5. 实验结果 (Results)**

### **(1) Hessian 特征值分析**

| 任务                 | Baseline (SGD) | SWA   | SAM   | WASAM    |
| -------------------- | -------------- | ----- | ----- | -------- |
| **WRN on CIFAR-100** | 673            | 265   | 237   | **117**  |
| **GIN on OGB-Code2** | 16.65          | 16.79 | 11.31 | **9.96** |

- **SAM 训练的模型比 SWA 训练的模型具有更低的 Hessian 最大特征值中位数**，意味着 SAM 发现的极小值更平坦。
- **WASAM (SWA + SAM) 取得了最平坦的优化解**，表明该方法可以更有效地找到稳定的优化解。

### **(2) 关键实验结果**

- **NLP 任务中，SAM > SWA**：
  - SAM 在 7/10 NLP 任务中表现更优。
  - SWA 在 NLP 任务上并不适用，有时甚至会降低性能。
- **GRL 任务中，SWA > SAM**：
  - SWA 在 10/12 GRL 任务中优于基线，而 SAM 仅在 4/12 任务中有提升。
- **WASAM 表现最稳定**：
  - 在 42 个任务中，39 个任务 WASAM 的结果优于 SWA 和 SAM 单独使用。

---

## **6. 结果分析 (Analysis)**

- **数据集和任务类型对优化方法的影响**：
  - **NLP 任务**：SAM 适用于 Transformer 结构，提升泛化能力。
  - **GRL 任务**：SWA 适用于 GNN，平滑更新更有效。
- **平坦极小值 (Flat Minima) 与泛化能力的关系**：
  - **较低的 Hessian 最大特征值意味着更好的泛化能力**。
  - **SAM 在 NLP 任务中找到更平坦的极小值**，而 **SWA 在 GRL 任务中更有效**。

---

## **7. 论文的最终结论**

- **优化器的有效性取决于任务类型和模型架构**：
  - **NLP 任务中，SAM 优于 SWA**，适用于 Transformer。
  - **GRL 任务中，SWA 优于 SAM**，适用于 GNN。
- **提出新型优化方法 (WASAM)**：
  - 结合 SWA 和 SAM，进一步改善优化解的平坦性，提高泛化能力。
- **Hessian 特征值分析提供优化曲面的几何解释**，支持平坦极小值的理论。



## **论文中的 9 大发现解析**

论文通过在多个任务上系统性地比较 **SWA**、**SAM** 和 **WASAM (SWA + SAM)**，总结了 9 个核心发现。这些发现揭示了优化器的适用性、数据集和模型架构的影响，以及优化方法的泛化能力。以下是对每个发现的详细解释。

---

## **Finding 1: 数据集对优化方法的影响**

**现象**：

- 在**不同数据集**上，优化方法的表现差异较大。例如：
  - 在 **OGB-Proteins** 数据集上，SWA 和 SAM 都不能显著提升性能。
  - 但在 **OGB-Products** 数据集上，SWA 和 SAM 显著优于基线方法。

**解释**：

- 数据集的**复杂性、噪声水平、类别分布**等因素可能影响优化器的有效性。
- 例如，在结构化信息较强的数据集上，SWA 可能更稳定，而在高度非结构化的数据集上，SAM 可能更有效。

---

## **Finding 2: 体系结构对优化方法的影响**

**现象**：

- 在**不同的模型架构**上，优化方法的有效性不同。例如：
  - 在 **Citation2** 链接预测任务中：
    - **使用 GCN (图卷积网络) 时**，SAM 相比 SWA **提升 >1.30%**。
    - **使用 SAGE (图采样方法) 时**，SWA 相比 SAM **提升 >1.15%**。

**解释**：

- **GCN 具有固定的邻居聚合方式**，更容易受损失曲面的影响，因此 SAM 更适用。
- **SAGE 使用可变邻域采样**，对噪声更鲁棒，因此 SWA 更有效。

---

## **Finding 3: SWA 在 NLP 任务上效果较差**

**现象**：

- 在 **10 个 NLP 任务**中：
  - **SAM 在 7 个任务上取得最佳性能**。
  - **SWA 在 4 个任务中导致性能下降**。

**解释**：

- Transformer 结构的优化动态与 CNN 或 GNN 不同，SWA 可能不适合。
- NLP 任务依赖 **长距离依赖和上下文信息**，局部参数平均 (SWA) 可能破坏 Transformer 训练的有效性。

---

## **Finding 4: SWA 在图表示学习 (GRL) 任务中优于 SAM**

**现象**：

- 在 **12 个 GRL 任务**中：
  - **SWA 在 10 个任务中优于基线**。
  - **SAM 仅在 4 个任务中优于基线**。

**解释**：

- **图神经网络 (GNN) 训练过程中，权重更新较为稳定**，SWA 适合权重平滑处理。
- SAM 需要计算梯度扰动，**在 GNN 任务中可能会过拟合局部结构**，导致泛化能力下降。

---

## **Finding 5: SWA 在 Transformer 上表现不佳**

**现象**：

- 在 ViT (Vision Transformer) 模型和 NLP 任务中，SWA **无法提升，甚至降低性能**。
- 相比之下，SAM 在 Transformer 上能提升泛化能力。

**解释**：

- Transformer 依赖 **层归一化 (LayerNorm) 和自注意力 (Self-Attention)** 机制，SWA 可能破坏权重的均衡性。
- SAM 通过损失敏感性优化，使模型更具鲁棒性，从而在 Transformer 上更有效。

---

## **Finding 6: SWA 和 SAM 也能改善自监督学习 (SSL)**

**现象**：

- 在 **自监督学习任务 (SimCLR, MoCo, BYOL, SwAV)** 上：
  - SWA 和 SAM **在部分任务上提升了 SSL 表现**。

**解释**：

- **SSL 任务的目标是学习稳定表征 (Stable Representations)**，而平坦极小值优化 (SWA, SAM) 可以减少表征的局部敏感性。
- 这表明 SWA 和 SAM 可能对 **对比学习 (Contrastive Learning) 和非对比学习 (Non-Contrastive Learning) 都有潜在提升**。

---

## **Finding 7: 并非所有任务都受益于平坦极小值优化**

**现象**：

- 在 **OGB-Proteins (蛋白质预测)** 和 **SSL 任务**上：
  - **SWA 和 SAM 无法显著提升性能**，甚至在部分任务上有所下降。

**解释**：

- **SSL 任务的优化目标与分类任务不同**，损失表面可能更复杂，优化器的影响较小。
- **OGB-Proteins 数据集具有较高的结构复杂性**，而 SWA 和 SAM 可能对低噪声环境更有效。

---

## **Finding 8: 平坦极小值优化的影响存在非对称性**

**现象**：

- **最差情况下，SWA 和 SAM 仅降低 < 0.30%**。
- **最佳情况下，SWA 和 SAM 可提高 >2.60%**。

**解释**：

- **这表明 SWA 和 SAM 的优化策略总体来说是安全的**，即使在某些任务上无效，负面影响也有限。
- **但在某些任务上，优化收益可以非常显著**，这解释了为什么 SWA 和 SAM 在许多研究中受到关注。

---

## **Finding 9: 结合 SWA 和 SAM (WASAM) 最稳定**

**现象**：

- **在 42 个任务中，39 个任务 WASAM (SWA + SAM) 优于单独的 SWA 或 SAM**。

**解释**：

- **SWA 使得优化轨迹更加平稳**，减少了局部震荡。
- **SAM 进一步增强了局部的鲁棒性**，使得最终的优化解更具泛化能力。
- **WASAM 结合了两者的优势**，在不同任务上表现最稳定，适应性最强。

---

## **总结**

论文的 9 个发现揭示了优化器对不同任务、数据集、模型架构的影响：

- **数据集和模型架构影响优化方法的有效性**（Finding 1 & 2）。
- **SWA 适用于 GNN，而 SAM 适用于 Transformer**（Finding 3, 4, 5）。
- **SWA 和 SAM 也适用于 SSL 任务**（Finding 6）。
- **优化方法的效果因任务不同，可能不会总是提升性能**（Finding 7）。
- **优化收益存在非对称性，最坏情况下影响有限**（Finding 8）。
- **结合 SWA 和 SAM 的 WASAM 取得最优结果**（Finding 9）。

这些发现为未来的优化器研究提供了重要的方向，表明优化方法应根据任务类型和模型架构进行选择和调整。



## **论文中的线性插值方法解析**

------

### **1. 线性插值方法的定义**

在该论文中，作者使用**线性插值（Linear Interpolation）**来研究不同优化方法找到的极小值（minima）在损失曲面上的相对位置，以及它们对模型泛化的影响。

线性插值的基本形式如下： 给定两个模型权重 $\theta_A$ 和 $\theta_B$，可以构造一个插值权重 $\theta_\alpha$：

$$\theta_\alpha = (1 - \alpha) \theta_A + \alpha \theta_B, \quad \alpha \in [0,1]$$

其中：

- 当 $\alpha = 0$ 时，$\theta_\alpha = \theta_A$；
- 当 $\alpha = 1$ 时，$\theta_\alpha = \theta_B$；
- 当 $\alpha$ 取其他值时，$\theta_\alpha$ 是 $\theta_A$ 和 $\theta_B$ 之间的一组插值参数。

为了分析损失曲面的变化，作者计算不同 $\alpha$ 下的损失值：

$$L(\theta_\alpha) = L((1 - \alpha) \theta_A + \alpha \theta_B)$$

并绘制损失曲线，以观察模型参数在损失空间中的行为。

------

### **2. 为什么线性插值能分析损失曲面？**

**(1) 评估损失极小值的平坦性**

- **如果$\theta_A$和$\theta_B$都处于一个宽泛的平坦极小值区域**，那么 $L(\theta_\alpha)$ 在插值过程中应该保持较小的变化，即损失函数呈现**平缓过渡**。
- **如果$\theta_A$和$\theta_B$处于不同的尖锐极小值区域**，那么 $L(\theta_\alpha)$ 可能会有**明显的损失峰（sharp peak）**，说明参数空间的损失变化剧烈。

**(2) 研究不同优化方法找到的极小值是否相似**

- **SWA 和 SAM 找到的极小值可能会有所不同**：
  - SWA 通过权重平均找到**更宽的极小值**，即插值路径上损失变化较小；
  - SAM 通过梯度扰动优化，找到**更具鲁棒性的极小值**，其插值损失曲线可能不同于 SGD。
- **若 SWA 和 SAM 之间的损失曲线变化平稳**：
  - 说明两者找到的极小值位于相对相近的平坦区域。
- **若损失曲线存在较大波动**：
  - 说明两者找到的极小值可能位于不同的优化轨迹。

**(3) 线性插值可视化优化路径**

- 通过绘制损失曲线，可以直观展示**不同优化方法如何影响极小值的分布**，从而解释哪些优化方法能够找到更平坦的解。
- 论文中可能还使用了**高维空间投影方法**，将多维参数空间的插值路径映射到可视化空间，以便分析。

------

### **3. 论文中的具体实验**

论文中使用线性插值方法来分析：

1. SWA 和 SGD 找到的极小值：
   - 观察它们之间的插值损失是否平滑，判断 SWA 选取的解是否更平坦。
2. SAM 和 SGD 找到的极小值：
   - 分析 SAM 是否能找到更具鲁棒性的极小值，并比较 SAM 选取的解和平坦性。
3. WASAM（SWA + SAM）找到的极小值：
   - 验证 WASAM 是否能同时结合 SWA 和 SAM 的优势，找到更好的平坦极小值。

------

### **4. 关键结论**

- SWA 和 SAM 找到的极小值在损失曲面上的位置不同：
  - SWA 的插值路径更平滑，表明其找到的极小值更平坦；
  - SAM 由于优化策略不同，可能会导致插值路径上的损失变化较大。
- WASAM 的插值路径最平滑：
  - 说明 WASAM 结合了 SWA 的权重平均特性和 SAM 的梯度优化特性，使得最终的极小值更平坦。

------

### **5. 总结**

- **线性插值是一种有效的损失曲面分析工具**，可以帮助研究者理解不同优化方法对模型参数空间的影响。
- **损失曲面的平坦程度与模型的泛化能力密切相关**，插值曲线较平滑的方法往往能找到更鲁棒的极小值。
- **本研究利用线性插值，验证了 SWA、SAM 和 WASAM 之间的差异**，并证明 WASAM 在多个任务上具有更好的表现。

------

### **附加：可能的改进方向**

- 研究更高维的插值方法，例如：
  - **多点插值（Multi-Point Interpolation）**，分析多个优化路径上的损失变化。
  - **曲线插值（Geodesic Interpolation）**，考虑非线性路径，以更准确地描述损失变化趋势。
- 结合**Hessian 特征值分析**，定量评估不同优化方法的曲面平坦性，而不仅仅依赖插值损失可视化。

以上分析详细解析了 **线性插值方法的作用、实验设计及结论**，并提出了未来的研究方向。



## **WASAM 方法解析**

------

### **1. WASAM 方法的定义**

**WASAM（Weight Averaged Sharpness-Aware Minimization）** 是本文提出的一种新型优化方法，旨在结合 **SWA（Stochastic Weight Averaging）** 和 **SAM（Sharpness-Aware Minimization）** 的优点，以进一步提升深度学习模型的泛化能力。

- **SWA** 的优势：通过**权重平均**找到更平坦的极小值（flat minima）。
- **SAM** 的优势：通过**对梯度进行扰动**，优化模型权重，使其找到更具鲁棒性的极小值，减少对训练数据的过拟合。

WASAM 通过对 **SAM 训练过程中不同阶段的权重进行平均**，从而保留 **SAM 的抗干扰能力** 和 **SWA 的平坦极小值特性**，最终获得更鲁棒的模型。

------

### **2. WASAM 如何结合 SWA 和 SAM？**

WASAM 通过 **对 SAM 训练得到的一系列模型权重进行随机权重平均** 来获得最终模型参数。该方法分为两个主要步骤：

#### **步骤 1：SAM 训练**

- 在模型训练过程中，采用 SAM更新权重：

  - 计算扰动梯度：

    $$g' = \nabla_{\theta} L(\theta + \rho \frac{\nabla_{\theta} L}{\|\nabla_{\theta} L\|})$$

  - 更新模型参数：

    $$\theta \leftarrow \theta - \eta g'$$

  - **此时模型逐步收敛到具有更强抗干扰能力的局部极小值**。

#### **步骤 2：对 SAM 训练的多个模型权重进行 SWA**

- 在 SAM 训练的后期，每隔 $K$轮采样一次模型权重：

  - 设定 $N$轮采样，保存多个权重：

    $$\Theta_{\text{SAM}} = \{\theta_{t_1}, \theta_{t_2}, ..., \theta_{t_N}\}$$

  - 计算平均权重：

    $$\theta_{\text{WASAM}} = \frac{1}{N} \sum_{i=1}^{N} \theta_{t_i}$$

  - **这个平均权重$\theta_{\text{WASAM}}$继承了 SAM 提供的鲁棒性，同时具备 SWA 提供的更平坦极小值特性**。

------

### **3. 为什么 WASAM 比 SAM 或 SWA 更优？**

WASAM 通过 **SAM 生成更鲁棒的模型权重，并通过 SWA 平滑这些权重**，因此在泛化能力方面优于单独使用 SAM 或 SWA：

| **方法**  | **主要特点**                            | **可能的缺点**                           |
| --------- | --------------------------------------- | ---------------------------------------- |
| **SWA**   | 通过权重平均找到更平坦的极小值          | 仅适用于标准训练，无法增强鲁棒性         |
| **SAM**   | 通过梯度扰动找到更鲁棒的极小值          | 计算量大，可能收敛到局部不够平坦的极小值 |
| **WASAM** | 结合 SWA 和 SAM，兼顾泛化性和平坦极小值 | 计算量比 SWA 略高                        |

**WASAM 的核心优势：**

1. **比 SAM 训练的模型更平滑**：SAM 可能会收敛到不够平坦的极小值，而 WASAM 通过 SWA 进一步平滑参数空间。
2. **比 SWA 训练的模型更鲁棒**：SWA 没有显式优化极小值的锐度，而 SAM 能够通过梯度扰动确保模型的泛化能力。
3. **比单独使用 SAM 更稳健**：SAM 可能会导致某些权重的局部极小值过于锐利，而 WASAM 通过权重平均减轻了这一问题。

------

### **4. WASAM 训练流程**

```
1. 采用 SAM 训练模型，确保找到更鲁棒的极小值
2. 在训练后期，每 K 轮存储一次模型权重
3. 在多个存储的权重上执行 SWA，计算加权平均
4. 使用 SWA 计算的最终模型进行推理
```

------

### **5. 关键实验结果**

在论文的实验中，WASAM 在 42 个任务中 **39 个任务的表现优于单独使用 SAM 或 SWA**，表明这种方法能够有效结合两者的优势。

实验发现：

- **在计算机视觉（CV）任务中，WASAM 的表现优于 SAM 和 SWA**；
- **在 NLP 任务中，WASAM 比 SAM 更稳定，且计算效率更高**；
- **在图神经网络（GNN）任务中，WASAM 保留了 SWA 提供的泛化优势**。

------

### **6. WASAM 的不足之处**

1. **计算成本仍然较高**：
   - WASAM 需要进行 SAM 训练（计算量较大），同时还需要执行 SWA 进行权重平均。
   - 可能的优化方向：
     - 采用 **分层 SWA（Layer-wise SWA）**，只对部分层进行权重平均；
     - 采用 **LookSAM 等优化策略**，减少梯度计算次数。
2. **对任务的适用性尚未完全探究**：
   - WASAM 适用于大多数 CV 和 NLP 任务，但在某些**小数据集**任务中，是否能有效提升泛化能力仍有待进一步研究。
   - 未来研究方向：
     - 研究 WASAM 在 **强化学习（RL）、元学习（Meta-learning）** 等任务中的表现。
3. **需要进一步优化超参数**：
   - **采样间隔$K$的影响**：如果 $K$ 过大，可能会错过关键优化区域；如果 $K$ 过小，可能会导致权重平均效果降低。
   - **$\rho$选择的影响**：SAM 的扰动半径 $\rho$ 可能需要对不同任务进行调整，以适应不同的数据分布。

------

### **7. 未来研究方向**

1. **减少计算量**：
   - 采用 **分层 SWA（Layer-wise SWA）**，只对部分层执行权重平均。
   - 结合 **LookSAM** 仅在部分训练轮次进行 SAM 计算，以减少计算成本。
2. **扩展到更多任务**：
   - WASAM 已经在 CV、NLP 和 GNN 任务上取得了不错的效果，未来可以探索其在**强化学习（RL）和元学习（Meta-learning）**等领域的适用性。
3. **超参数调优**：
   - 研究如何 **自适应调整 SAM 的扰动半径$\rho$**，避免手动调参的负担。
   - **探索不同的权重平均策略**，如**指数滑动平均（EMA）**，进一步优化最终模型。

------

### **8. 结论**

- **WASAM 结合了 SAM 的鲁棒性和 SWA 的平坦极小值特性，是一种更稳健的优化方法**。
- **实验结果表明，WASAM 在多个任务上优于单独使用 SAM 或 SWA**，并且在泛化能力方面具有更好的表现。
- **未来研究可优化 WASAM 计算开销，并探索其在其他任务中的适用性**。

------

📌 **代码开源（如果有）**：[GitHub - WASAM](https://github.com/xxxxxx) 🚀

------

本综述详细解析了 **WASAM 方法的核心思想、训练步骤、实验结果、关键注意点**，并以 Markdown 代码格式输出，确保可复制与二次编辑。







# Paper4 : Revisiting Catastrophic Forgetting in Large Language Model Tuning



## **论文分析：《Revisiting Catastrophic Forgetting in Large Language Model Tuning》**

#### **1. 研究背景与范围 (Scope or Setting)**

本论文关注**大语言模型（LLMs）在微调过程中出现的灾难性遗忘（Catastrophic Forgetting, CF）问题**，这一问题指的是模型在学习新任务或数据时，逐渐遗忘先前学到的知识，从而影响其泛化能力。现有研究主要从**数据增强（如回放方法）\**和\**模型优化**两个方向尝试解决CF问题，但前者通常需要额外的存储与计算开销，后者尚未充分研究优化方法与CF之间的关系。

#### **2. 研究目标 (Key Idea & Purpose)**

本研究的核心目标是：

1. **揭示灾难性遗忘与损失曲面（Loss Landscape, LLS）平坦度之间的直接联系**，从优化角度分析CF现象的根本原因。
2. **通过Sharpness-Aware Minimization（SAM）优化方法**来降低损失曲面的尖锐程度，从而减轻CF问题，并验证SAM的有效性。
3. **探索SAM方法与现有抗遗忘策略（如Wise-FT、Rehearsal）之间的互补性**，验证其是否能协同提高抗遗忘能力。

#### **3. 研究方法 (Method)**

本论文的研究方法包括三个部分：

1. **分析CF与损失曲面平坦度的关系**

   - 采用**损失曲面可视化**和**量化平坦度的指标（SC, AG, MAG）**，直观呈现损失曲面的变化如何影响模型遗忘程度。
   - 在Llama2-7B模型上，通过逐步增加数据集任务难度（如Alpaca → Open-Platypus → Auto-Wiki），观察损失曲面平坦度的变化与CF现象的严重性之间的关系。

2. **利用SAM优化方法来减缓CF**

   - 采用SAM方法，使得模型优化时不仅最小化损失值，还要**确保在一个小扰动范围内损失不会显著增加**：

   $$\min_w \max_{\|\epsilon\|_2 \leq \rho} f(w + \epsilon)$$

   - 通过泰勒展开对扰动项进行近似：

   $$\min_w f\left(w + \rho \frac{\nabla_w f(w)}{\|\nabla_w f(w)\|_2}\right)$$

   - 这一优化过程确保模型在平坦的极小值处收敛，从而提高泛化能力并减轻灾难性遗忘。

3. **在不同规模的LLMs上测试SAM的有效性**

   - 选取 **TinyLlama-1.1B, Llama2-7B, Llama2-13B** 三种不同规模的模型，评估SAM对不同大小模型在抗遗忘上的表现。
   - 使用多个基准数据集（Alpaca, ShareGPT52K, MetaMathQA, Open-Platypus等）进行对比实验。

#### **4. 主要贡献 (Contribution)**

1. **首次系统性揭示了损失曲面平坦度与灾难性遗忘之间的直接关系**，发现**损失曲面越尖锐，模型的遗忘程度越严重**。
2. **首次从优化角度**（而非数据增强或网络结构改进）**缓解LLMs的灾难性遗忘问题**，提出**使用SAM来优化损失曲面**，降低曲面尖锐程度，进而减轻CF。
3. 通过实验表明，**SAM可以与现有的抗遗忘方法（如Wise-FT, Rehearsal）互补，提高抗遗忘能力**。
4. **SAM在不同规模的LLMs上均有效，且随着模型规模的增加，CF问题更严重，而SAM的缓解效果更明显**。

#### **5. 研究结果 (Results & Conclusion)**

实验结果表明：

1. **灾难性遗忘与损失曲面平坦度高度相关**：
   - 随着训练数据难度增加（Alpaca → Open-Platypus → Auto-Wiki），损失曲面变得更尖锐，模型在MMLU等任务上的性能下降最多达**17.2%**。
   - 在MMLU、SuperGLUE、RACE等基准任务上的实验表明，**SAM可以有效减缓CF问题，提升模型性能**。
2. **SAM在不同模型规模上的表现**
   - **模型越大，灾难性遗忘问题越严重**，Llama2-13B的CF程度比TinyLlama-1.1B更高。
   - **SAM在大模型上抗遗忘效果更显著**，Llama2-13B采用SAM优化后性能提升近**10%**。
3. **SAM可以与现有方法（Wise-FT, Rehearsal）互补**
   - 单独使用SAM可以将CF程度降低约 **7.01%**。
   - 结合Wise-FT或Rehearsal后，抗遗忘效果进一步提升，可减少**9%以上**的遗忘问题。

最终，论文得出结论：**灾难性遗忘与损失曲面平坦度高度相关，采用SAM方法可以有效缓解CF，并且SAM可以与现有的抗遗忘方法协同提升模型性能**。

------

### **6. 研究的不足与未来改进方向**

#### **研究的不足之处**

1. **SAM的计算开销较大**
   - SAM需要两次前向传播，计算复杂度增加了一倍，这可能限制了其在大规模训练中的应用。
   - 未来可以研究更高效的平坦度优化方法，如**近似梯度扰动技术**。
2. **仅研究了优化角度的影响**
   - 本研究主要关注优化方法对CF的影响，而其他潜在的影响因素（如**数据选择策略、知识蒸馏方法**）尚未深入探讨。
   - 未来可以结合这些方法，与优化方法相结合，形成更完善的抗遗忘策略。
3. **未探讨SAM在长期训练中的影响**
   - 论文主要研究了短期微调（fine-tuning）阶段的CF，而**长期增量学习（如在线更新LLMs）**的抗遗忘能力仍有待探索。
4. **仅在标准数据集上测试，未涉及真实世界任务**
   - 本文主要在公开数据集（Alpaca, ShareGPT等）上测试，而未涉及**实际工业应用中的LLM更新场景**（如自动驾驶数据、金融模型的持续学习）。
   - 未来可在更复杂的应用环境下测试，并结合**在线学习与记忆重放机制**。

------

### **7. 未来研究方向**

1. **降低SAM的计算开销**
   - 采用**近似梯度方法**或**稀疏扰动策略**，减少计算复杂度。
   - 结合**LookSAM等轻量级SAM变种**，提升计算效率。
2. **结合数据增强与知识蒸馏**
   - 研究如何将**数据增强、知识蒸馏与SAM结合**，在降低遗忘的同时，保持模型效率。
3. **探索长期持续学习场景**
   - 在**长期在线训练或增量学习场景**下，研究如何优化SAM，使其适用于更复杂的训练流程。
4. **应用于更大规模的LLMs**
   - 在**GPT-4、Gemini等超大规模模型**上测试SAM的表现，并进一步优化其适用性。

------

## **总结**

本论文首次系统性分析了灾难性遗忘与损失曲面平坦度的关系，并提出使用**Sharpness-Aware Minimization（SAM）**来优化损失曲面，从而缓解CF问题。实验结果表明，**SAM不仅有效降低CF，还可与现有方法互补，提高抗遗忘能力**。尽管该方法计算复杂度较高，未来研究可以探索更高效的优化策略，并在长期在线学习场景中测试其可行性。











































# 个人思考

# Flat Minima 与优化理论



### **Flat Minima 与流形理论（Manifold Theory）、微分流形的关联**

Flat Minima 作为优化理论中的一个核心概念，其与**流形理论（Manifold Theory）**和**微分流形（Differentiable Manifolds）**有着深刻的数学联系。通过流形理论的视角，我们可以更好地理解 Flat Minima 的几何性质，以及其对神经网络优化、泛化能力的影响。

------

## **1. Flat Minima 的几何描述**

在深度学习的优化过程中，损失函数 L(θ)L(\theta) 的最小值往往不会是孤立的点，而是形成了一个**低维结构**，即在高维参数空间中的一个“流形”（manifold）。

具体来说：

- 在神经网络的高维参数空间中，损失函数通常具有多个局部最小值，并且**Flat Minima 形成了一个近似的低维流形**。
- 这些 Flat Minima 并非单个点，而是一个**局部近似平坦的子空间**，其中损失函数的梯度变化缓慢，即 Hessian 矩阵的特征值较小。

用数学语言描述，假设神经网络的损失函数 L(θ)L(\theta) 定义在一个高维参数空间 Rn\mathbb{R}^n 上：
$$
\theta^* = \arg \min_{\theta} L(\theta)
$$
那么，Flat Minima 可以被视为一个**嵌入在参数空间中的低维微分流形** M\mathcal{M}，其维度取决于 Hessian 矩阵的零特征值数量：
$$
\mathcal{M} = \{ \theta \in \mathbb{R}^n \mid L(\theta) \text{ 在 } \mathcal{M} \text{ 上变化缓慢}\}
$$


------

## **2. 微分流形的视角：Flat Minima 作为流形**

在微分流形理论中，一个**流形（manifold）**是一个在局部与欧几里得空间 $\mathbb{R}^m$ 同胚的集合。例如：

- **曲面（Surface）**是 3D 空间中的 2 维流形。
- **损失函数的最优解集**可以形成高维参数空间中的流形。

如果 Flat Minima 形成了一个低维流形$ \mathcal{M}$：

- **流形的切空间（Tangent Space）** $ T_{\theta^*} \mathcal{M} $由 Hessian 矩阵的零特征值对应的方向决定。
- **法空间（Normal Space）** 表示 Hessian 矩阵的非零特征值对应的方向，即损失函数变化剧烈的方向。

因此，Flat Minima 可以视为损失函数的等高流形（level manifold），即：
$$
\mathcal{M} = \{ \theta \in \mathbb{R}^n \mid L(\theta) = C, \quad \forall \theta \in \mathcal{M} \}
$$
其中 CC 是一个常数，表示在 Flat Minima 处损失值几乎不变。

### **Hessian 矩阵与流形曲率**

微分几何中，曲率可以用于描述流形的形状。Hessian 矩阵 HH 可以类比为**流形的二阶曲率张量**：

- 若 Hessian 具有多个接近零的特征值，则损失表面在这些方向上是**平坦的（Flat）**，意味着形成了 Flat Minima。
- 若 Hessian 具有较大的正特征值，则对应的方向是陡峭的（Sharp Minima）。

换句话说，Flat Minima 的流形**局部曲率较小**，因此模型的泛化能力更强。

------

## **3. 连接 Flat Minima、流形优化与神经网络**

在深度学习中，流形理论提供了一种理解神经网络优化的新视角：

### **(1) 流形优化（Manifold Optimization）**

- 传统优化方法如 SGD 在参数空间 $\mathbb{R}^n$ 中更新，但实际上，最优解可能**局限于一个低维流形** $\mathcal{M}$。
- 采用**流形优化方法**（如 Riemannian Gradient Descent, Riemannian Trust-Region）可以更有效地搜索 Flat Minima。

### **(2) 神经网络的 Loss Surface**

- 研究发现，**宽度较大的神经网络的损失表面更像是一个低维流形**，而不是单个孤立的最优点。
- 例如，ResNet、Transformer 这类大规模模型，其训练过程中找到的 Flat Minima 可能属于某个高维流形。

### **(3) 泛化能力**

- Flat Minima 形成的流形通常比 Sharp Minima 更大，即**存在较多的等价解**。
- 这意味着参数扰动不会导致损失函数显著变化，从而提高泛化能力。

------

## **4. Flat Minima 的微分几何应用**

基于流形理论，我们可以在神经网络优化中设计新方法，以找到更好的 Flat Minima：

### **(1) Hessian 正则化**

$$
L'(\theta) = L(\theta) + \lambda \cdot \text{trace}(H)
$$

通过增加 Hessian 矩阵的迹（Trace），鼓励找到较低曲率的区域，即 Flat Minima。

### **(2) Laplacian-Based Loss**

$$
L'(\theta) = L(\theta) + \lambda \cdot ||\nabla^2 L(\theta)||
$$

利用 Hessian 的二阶信息优化损失，使其沿着低维流形调整优化方向。

### **(3) SGD 与流形优化**

标准的 SGD 在高维空间中搜索最优解，而如果我们在流形上优化（如 Riemannian Gradient Descent），可以更高效地找到 Flat Minima。

------

## **5. 结论**

Flat Minima 及其几何性质可以通过**流形理论和微分流形**得到更深刻的理解：

1. **Flat Minima 形成一个低维流形**，而不是单个孤立点。
2. **Hessian 矩阵的特征值决定流形的曲率**，特征值越小，表示损失表面越平坦。
3. **流形优化方法**可以帮助更高效地搜索 Flat Minima，从而提高神经网络的泛化能力。

这表明，在神经网络优化中，我们可以借助**微分几何、流形优化**等数学工具，构建更鲁棒的优化方法，提升深度学习模型的性能。

------

🚀 **如果你对流形优化、神经网络几何结构等话题感兴趣，我们可以进一步讨论更高级的数学工具（如黎曼几何、谱分析等）在深度学习中的应用！**



## **从优化的角度看神经网络的权重更新：流形视角下的 SGD、SAM 及其与 Flat Minima 的关联**

神经网络的优化过程通常发生在**高维非凸损失空间**中，标准优化方法（如 SGD）在欧几里得空间 Rn\mathbb{R}^n 中进行。然而，流形优化（Manifold Optimization）提供了一种更具几何视角的方法，使得我们能够更深入地理解权重更新的本质，尤其是在**寻找 Flat Minima** 方面的作用。

------

## **1. 神经网络的优化空间：损失景观与流形**

神经网络的损失函数 L(θ)L(\theta) 定义在高维权重空间中，优化的目标是找到最优参数 θ∗\theta^* 以最小化损失。这个优化问题通常具有以下几何特性：

- **非凸性（Non-convexity）**：损失表面通常是复杂的、具有多个局部极小值。
- **高维等价类结构（High-dimensional Equivalence Classes）**：由于权重的对称性（如权重排列不影响输出），权重空间并不是真正的欧几里得空间，而更像是**某种流形（manifold）**。
- **Flat Minima vs. Sharp Minima**：某些最小值区域是平缓的（flat），某些是陡峭的（sharp）。Flat minima 具有较小的 Hessian 特征值，从而提高泛化能力。

因此，我们可以用**流形理论**来重新审视优化算法的行为，特别是 SGD 和 SAM 如何在流形上找到合适的 Flat Minima。

------

## **2. 流形优化视角下的 SGD**

### **(1) 传统 SGD 及其流形解释**

**随机梯度下降（SGD）** 是神经网络优化的核心方法，其更新规则为：
$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t) + \xi_t
$$
其中：

- $\eta $是学习率，
- $\xi_t$ 是随机梯度噪声。

在传统的欧几里得空间 $\mathbb{R}^n$ 中，SGD 直接沿着梯度方向下降。然而，从**流形视角**来看：

- SGD 在损失景观上的移动路径并不是严格的直线，而是受损失表面几何结构的影响，类似于沿着流形的测地线（geodesics）前进。
- 由于 SGD 引入了噪声项 $\xi_t$，它实际上是一种**随机近似优化**，能够跳出 sharp minima 并倾向于找到更广阔的 flat minima 区域。

### **(2) SGD 对 Flat Minima 的偏好**

研究表明，SGD 自然倾向于找到 Flat Minima（Smith et al., 2018）：

- SGD 的噪声项相当于在损失表面上进行一个**热动力学扰动**，可以跳出高曲率区域（Sharp Minima）。
- 通过**Fokker-Planck 方程**建模 SGD 可以发现，它的优化路径趋向于具有较低 Hessian 最大特征值的区域，即 Flat Minima。
- **大批量训练（Large-batch Training）** 由于减少了噪声，容易陷入 Sharp Minima，而小批量 SGD 则更有助于找到 Flat Minima。

**总结**：

- 在流形上，SGD 的优化路径并不遵循标准欧几里得梯度，而是在噪声作用下动态探索损失表面的低曲率区域。
- 这意味着 SGD 本质上是一种**流形上的随机优化方法**，天然偏好 Flat Minima。

------

## **3. SAM（Sharpness-Aware Minimization）与流形优化**

### **(1) SAM 算法**

**Sharpness-Aware Minimization（SAM）** 是一种旨在显式寻找 Flat Minima 的优化方法（Foret et al., 2021）。它的优化目标是：
$$
\min_{\theta} \max_{\|\epsilon\| \leq \rho} L(\theta + \epsilon)
$$
其中：

- $\epsilon$ 是一个微小扰动，$∥ϵ∥≤ρ\|\epsilon\| \leq \rho$ 限制了扰动的范围，
- 内部的最大化步骤会寻找损失函数的“最坏情况”扰动，
- 外部的最小化步骤调整 θ\theta 以降低这种最坏情况的损失。

**优化更新步骤**：

1. 计算梯度方向上的扰动： ϵ∗=ρ∇L(θ)∥∇L(θ)∥\epsilon^* = \rho \frac{\nabla L(\theta)}{\|\nabla L(\theta)\|}
2. 计算扰动后的梯度： g=∇L(θ+ϵ∗)g = \nabla L(\theta + \epsilon^*)
3. 更新参数： θt+1=θt−ηg\theta_{t+1} = \theta_t - \eta g

### **(2) SAM 在流形上的优化视角**

- SAM 通过**构造扰动**来探索损失景观的几何特性，类似于沿着流形的局部测地线搜索最优路径。
- 其内部最大化步骤在一个小球邻域内找到损失的最大值，这可以看作是沿着损失景观的曲率方向进行优化。
- SAM 可以被看作是 SGD 在流形上进行的“曲率调整”，即通过显式寻找低曲率区域，强制优化过程朝向 Flat Minima。

**与 SGD 的对比**：

| **方法** | **优化方式**               | **流形解释**               | **Flat Minima 选择性**                 |
| -------- | -------------------------- | -------------------------- | -------------------------------------- |
| **SGD**  | 沿梯度下降，有噪声         | 在流形上进行随机梯度下降   | 自然偏向于 Flat Minima，但不是显式优化 |
| **SAM**  | 额外寻找损失曲率较小的方向 | 在流形上执行“局部鲁棒优化” | 通过最大化局部扰动主动寻找 Flat Minima |

### **(3) SAM 在高曲率流形上的行为**

如果我们考虑神经网络优化空间为一个高维黎曼流形 MM，则：

- SGD 沿着该流形的测地线随机行进，受损失景观的曲率影响。
- SAM 通过主动寻找损失较平坦的区域，相当于对流形上的曲率进行“平滑”处理，使得优化路径朝向低曲率方向。

这种几何视角解释了**为什么 SAM 能够显著提高神经网络的泛化能力**。

------

## **4. Flat Minima、SGD 和 SAM 的几何关系**

从优化几何的角度，SGD 和 SAM 在神经网络的流形优化上具有不同的偏向性：

1. **SGD 依靠噪声进行“隐式正则化”**，偏向于找到 Hessian 特征值较小的区域（Flat Minima）。
2. **SAM 通过局部扰动“显式优化”**，直接将优化方向引导至低曲率区域。
3. 流形几何解释：
   - SGD 在高维流形上进行随机优化，类似于一个带噪声的梯度流（stochastic gradient flow）。
   - SAM 则在流形上调整优化方向，以更明确地避免高曲率点，类似于曲率感知的优化（curvature-aware optimization）。

这种几何视角解释了为什么**SAM 能够改进 SGD 在某些任务上的性能，尤其是在泛化能力上的提升**。

------

## **5. 总结**

1. **SGD 是一种流形上的随机梯度优化**，其噪声特性有助于寻找 Flat Minima。
2. **SAM 通过显式优化曲率，在流形上调整优化方向**，更有效地选择低曲率区域。
3. SGD 和 SAM 的优化路径不同：
   - SGD 依赖于优化噪声，在流形上进行随机搜索。
   - SAM 通过局部扰动调整优化方向，主动选择平坦区域。

这一流形视角不仅帮助理解 Flat Minima 的选择性，也提供了一种优化算法设计的新思路🚀。



## 从优化的视角看神经网络的权重更新，从流形或黎曼流形的角度介绍GD, SGD，SWA，SAM 算法与 flat minial 的关系

以下内容将从优化的角度出发，结合流形（manifold）或黎曼流形（Riemannian manifold）的观点，对常见的几种训练方法——包括普通（批）梯度下降（GD）、随机梯度下降（SGD）、随机权重平均（SWA）以及尖锐度感知的最小化（SAM）——进行梳理与评述，并探讨这些方法与 **flat minima**（平坦极小点）之间的关联。文风将偏向“论文评审专家”的学术讨论方式，希望对相关读者理解其几何机理有所帮助。

------

## 1. 前言与背景

在神经网络的训练过程中，优化算法的核心目标是寻找能在训练集上将损失函数降到足够低、并在测试集上保持良好泛化性能的参数解。近年大量研究提出，“平坦极小点”（flat minima）往往能带来更好的泛化。与此同时，随着流形与黎曼几何理论在机器学习中的推广，人们逐渐意识到：在高维的参数空间 Rn\mathbb{R}^n 中，“损失地形”（loss landscape）可以用更加精细的几何学视角来研究，从而为模型的泛化和稳定性提供理论支撑。

下文将依次回顾与分析以下几种典型算法：

- **GD（Gradient Descent，批梯度下降）**
- **SGD（Stochastic Gradient Descent，随机梯度下降）**
- **SWA（Stochastic Weight Averaging）**
- **SAM（Sharpness-Aware Minimization）**

并基于流形/黎曼流形的角度，阐述它们如何影响或逼近平坦极小点。

------

## 2. GD：在欧几里得与黎曼流形上的梯度下降

### 2.1 传统欧几里得角度

在最朴素的形式中，批梯度下降（GD）假设我们可以获得所有训练样本的精确梯度，每次更新的形式为
$$
\theta \leftarrow \theta - \alpha \nabla \mathcal{L}(\theta),
$$
其中 $\alpha $是学习率，$\mathcal{L}(\theta)$ 表示损失函数。该更新规则将参数向梯度反方向移动，从而逐步降低损失值。对于规模适中的问题，GD 能较稳定地收敛到某个局部极小点。

### 2.2 黎曼流形视角

若考虑将参数空间“升格”为带度量的黎曼流形，则需要为参数$ θ\theta$ 引入合适的度量张量 $G(θ)\mathbf{G}(\theta)$。在信息几何中，$G(θ)\mathbf{G}(\theta) $常可取 Fisher 信息矩阵或其他二阶近似（如 Hessian）。这样一来，GD 的更新可以写作
$$
\theta \leftarrow \theta - \alpha \,\mathbf{G}(\theta)^{-1}\nabla \mathcal{L}(\theta),
$$
即所谓的自然梯度或 Riemannian 梯度下降（在某些文献中也有更精细的讨论）。该方法可以理解为在曲率复杂的流形上沿测地线方向做“最自然”的移动，从而更好地适配局部曲率分布。

### 2.3 与 Flat Minima 的关联

- **缺点**：纯批量的 GD，因其缺少随机扰动，在非凸高维问题中可能易陷于陡峭极小点或鞍点。
- **优点**：若网络规模中等，批量梯度更能准确地朝向真正的全局“陡峭/平坦”区域前进。若再加上合适的正则项或二阶信息，则在一定程度上能朝更平坦的极小区靠拢。

然而在大规模训练场景下，仅用 GD（全数据计算梯度）往往并不现实，人们更多依赖分批随机的更新方式，即 SGD。

------

## 3. SGD：随机扰动与平坦极小点

### 3.1 基本原理

SGD（或 Mini-batch SGD）采用小批量样本来近似计算梯度，使得更新公式变得噪声化：
$$
\theta \leftarrow \theta - \alpha \nabla \mathcal{L}_\text{mini-batch}(\theta).
$$
这种内在随机性常被视作在训练过程中“注入了噪声”，帮助模型逃离一些陡峭的局部极小点或鞍点，并有更大概率进入平坦区域。

### 3.2 流形视角下的随机优化

当我们将 SGD 的更新轨迹想象为在高维流形上进行随机梯度流动（stochastic gradient flow）时：

1. 噪声带来局部的随机扰动，可理解为在曲率不均衡的区域中进行自适应探索。
2. 训练后期，学习率逐步缩小时，若损失曲面存在大体积、曲率较低的平坦谷地，则随机梯度通常更容易“留在”这些区域，而不是陡峭极小点。

在理论上，部分鞍点与尖锐极小点往往需要更精确的梯度（或更大梯度幅度）才能维持收敛；一旦存在噪声扰动，就更容易使训练点跳出这些局部束缚，长远来看反而有利于获得更好的泛化性能。

### 3.3 与 Flat Minima 的关联

- **有利性**：SGD 的随机性是帮助网络“泛化”的重要原因之一，实证表明其最终收敛解往往更偏向平坦极小点。
- **局限性**：过大或过小的批量大小都会影响收敛到平坦区的概率；此外，如果学习率或批量规模设置不当，也可能错过更大的平坦谷地。

------

## 4. SWA：随机权重平均及其几何机理

### 4.1 算法概述

SWA（Stochastic Weight Averaging）核心思路是：

1. 在训练过程后期，保存多次迭代得到的网络参数 θt\theta_t。
2. 用简单平均或加权平均，将这些权重合成为最终模型参数。

具体形式可写作
$$
\theta_{\text{SWA}} = \frac{1}{T}\sum_{t=1}^{T} \theta_{t},
$$
其中$ θt\theta_t$ 是在训练后期、若干固定间隔保存的模型。

### 4.2 流形/黎曼几何的解释

在高维损失表面中，SWA 相当于在参数空间内对若干个相距不太远的权重点进行平均，从而“跨越”了多条微小的损失沟壑。这种平均化后得到的参数往往坐落于这些点所对应损失面的低谷区的“中心位置”。

- **几何视角**：
   如果将训练过程看成在等损失曲面附近振荡，则不同时间步的参数会分布在某个“谷地”周围。对这些参数做平均，往往能移到曲率更缓、体积更大、对局部扰动不敏感的区域，可视为在流形上找到一个“折中”点。

### 4.3 与 Flat Minima 的关联

研究和实验都表明，SWA 能显著提升泛化性能，这通常被解释为“令优化更偏向于平坦极小点”。从损失面的角度看，SWA 本质上将多个邻近极小值的区域融合，找到一个相对居中的参数解；若我们量化 Hessian 的特征值，往往也可观察到该平均参数在局部曲率上呈现“较低、较均匀”的趋势。

------

## 5. SAM：尖锐度感知的最小化与几何曲率

### 5.1 SAM 的核心思路

SAM（Sharpness-Aware Minimization）在更新权重时，不仅关注当前梯度，还会在参数附近寻找导致损失上升的“最坏扰动”并将其纳入优化目标。具体而言，SAM 定义了新的目标函数：
$$
\mathcal{L}_{\text{SAM}}(\theta) = \max_{\|\epsilon\|\leq\rho} \mathcal{L}(\theta+\epsilon),
$$
并在每次迭代中近似解决这个“最大化问题”，将结果纳入更新方向中。这样可以有效惩罚那些对于小扰动就会显著增大损失的“陡峭”点，迫使参数更倾向于曲率较缓的区域。

### 5.2 流形视角

从流形与曲率的角度来看，SAM 相当于在每一步更新前，主动探索在θ\theta邻域内曲率最陡的一侧（即对损失最敏感的方向），然后对该方向进行更强的“离开”或“抑制”。如果说传统的梯度下降是被动地顺着梯度更新，那么 SAM 则是在曲率凸起最严重的方向上做了额外修正。

- **几何含义**：通过对局部极值区曲率敏感的方向进行额外惩罚，使优化轨迹更容易滑向那些曲率更平坦、对小扰动不敏感的区域。

### 5.3 与 Flat Minima 的关联

SAM 的主要目标便是逼近或“强制”寻找平坦极小点。在众多实证研究中，SAM 常被证明能显著提高测试精度、稳定性和对抗鲁棒性，其作用机理正是主动规避“陡峭极小点”，鼓励在参数空间找到更大的低损失盆地。

------

## 6. 结论与展望

通过上文的分析，我们可以将 **GD**、**SGD**、**SWA**、**SAM** 这几种方法从传统欧几里得空间扩展到流形/黎曼流形的角度进行解读，进一步阐明它们与 **flat minima**（平坦极小点）的关系：

1. **GD（批量梯度下降）**：在流形视角下可视为最朴素的沿梯度方向下降，但缺乏随机扰动，易陷于局部尖锐极小点，除非借助二阶信息或正则化。
2. **SGD（随机梯度下降）**：噪声扮演“逃离陡峭极小点”的角色，帮助模型更大概率收敛到较平坦的区域。
3. **SWA（随机权重平均）**：在参数空间中对不同训练步的权重进行平均，相当于在曲率相对缓和的谷地中心落点，并在实证中表现出良好的泛化能力。
4. **SAM（尖锐度感知最小化）**：通过显式地惩罚那些对局部扰动敏感的方向，可视为对曲率进行“主动塑形”，直接朝平坦极小值区逼近。

未来研究的重点包括：

- **在超大规模模型中实现更高效的几何更新**：如近似二阶方法、块对角或低秩近似等。
- **与统计学习理论的统一视角**：融入 PAC-Bayesian 或随机微分方程（SDE）的分析框架，量化平坦极小点的概率与泛化界。
- **在对抗训练、元学习、强化学习等场景的应用**：不同类型的损失或策略优化都存在曲率分布差异，如何有效地寻找平坦区将成为进一步提升性能的关键。

总而言之，从流形或黎曼流形角度审视神经网络的权重更新，能更深刻地理解为什么这些优化方法能够（或不能）找到对小扰动不敏感、泛化性强的解，也能为后续算法设计和理论研究提供更加丰富的几何工具与思想。





# **优化方法对比：SAM、L2 正则化、SGD、Adam 在梯度更新和噪声约束方面的联系与区别**

深度学习优化方法在训练过程中面临两个核心挑战：
1. **如何有效更新梯度** 以优化损失函数 $L(\theta)$；
2. **如何控制噪声**，避免过拟合或梯度方向的不稳定性。

本文将分析 **SGD、Adam、L2 正则化** 和 **SAM（Sharpness-Aware Minimization）** 这四种方法在目标、更新策略和对噪声的处理方式上的联系与区别。

---

## **1. Stochastic Gradient Descent (SGD)**
### **1.1 目标**
SGD 通过计算损失函数的梯度 $\nabla L(\theta)$ 并沿梯度方向更新参数：
$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t).
$$
其中：
- $\eta$ 是学习率，控制更新步长；
- $\nabla L(\theta_t)$ 是损失函数关于参数 $\theta$ 的梯度。

### **1.2 设计思路**
- 直接沿着当前批量数据的梯度方向进行更新；
- 通过 mini-batch 采样减少计算开销，同时引入一定程度的噪声，使优化过程更具随机性。

### **1.3 对噪声的处理**
- **噪声的自然存在**：由于 SGD 在每次更新时仅使用一部分样本（mini-batch），计算的梯度具有一定噪声，这种噪声有助于逃离鞍点，但可能导致不稳定训练。
- **无显式约束**：SGD 本身不对梯度噪声进行显式控制，而是依赖 batch size、学习率等因素间接影响噪声水平。

### **1.4 约束的方式**
- **无额外正则化项**：SGD 本身不会对参数的变化范围进行限制，优化过程中参数可以自由变化。

---

## **2. Adam**
### **2.1 目标**
Adam（Adaptive Moment Estimation）是一种**自适应学习率优化方法**，其目标是：
- 结合 **SGD** 的易实现性和 **RMSProp** 的自适应特性，使得参数更新速度更稳定；
- 通过动量积累和梯度平方缩放，使不同参数能按照自身的变化速率进行调整。

### **2.2 设计思路**
Adam 结合了一阶矩估计（动量）和二阶矩估计（梯度方差）：
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(\theta_t),
$$
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L(\theta_t))^2.
$$
其中：
- $m_t$ 是梯度的一阶动量（类似于动量优化）；
- $v_t$ 是梯度的二阶动量（类似于 RMSProp 中的梯度方差估计）；
- **最终更新规则**：
  $$
  \theta_{t+1} = \theta_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}.
  $$

### **2.3 对噪声的处理**
- **自适应调整学习率**：梯度较大的方向，学习率会自动缩小；梯度较小的方向，学习率会自动放大，从而减少梯度振荡，使优化更加稳定。
- **对梯度噪声更鲁棒**：由于二阶动量的引入，Adam 可以更有效地适应不同方向的梯度变化，从而降低梯度更新中的噪声。

### **2.4 约束的方式**
- **间接限制噪声**：通过自适应调整学习率，而非显式的正则化项来约束梯度更新。
- **易陷入局部极小值**：由于对梯度变化进行了平滑处理，Adam 可能在训练后期过快收敛于局部极小点，而缺乏 SGD 那种逃离鞍点的能力。

---

## **3. L2 正则化（权重衰减）**
### **3.1 目标**
L2 正则化（也称为权重衰减，Weight Decay）在优化目标中增加了一项正则化项：
$$
\min_{\theta} L(\theta) + \lambda \|\theta\|^2_2.
$$
对应的梯度更新：
$$
\theta_{t+1} = \theta_t - \eta \left( \nabla L(\theta_t) + \lambda \theta_t \right).
$$

### **3.2 设计思路**
- 通过在损失函数中添加 $\lambda \|\theta\|^2_2$ 项，迫使参数 $\theta$ 保持较小的范数，减少模型复杂度，从而提高泛化能力。

### **3.3 对噪声的处理**
- **隐式减少噪声影响**：L2 正则化会抑制权重的绝对值，从而避免模型对噪声样本过度拟合。
- **对梯度没有直接影响**：与 SAM 不同，L2 正则化不会对梯度方向做额外的扰动或调整，而是直接在参数空间约束模型。

### **3.4 约束的方式**
- **通过参数范数约束梯度更新**：不会直接修改梯度，而是通过抑制权重的增长间接控制模型复杂度。

---

## **4. SAM（Sharpness-Aware Minimization）**
### **4.1 目标**
SAM 直接优化损失函数的**尖锐度（sharpness）**，目的是找到平坦极小点：
$$
\min_{\theta} \max_{\|\epsilon\|\leq \rho} L(\theta + \epsilon).
$$
- 其中 **内层最大化** $\max_{\|\epsilon\|\leq \rho} L(\theta + \epsilon)$ 计算“最坏方向”的损失变化。
- **外层最小化** $\min_{\theta}$ 则优化参数 $\theta$，使得其在局部区域内对扰动不敏感，即找到较“平坦”的极小点。

### **4.2 设计思路**
SAM 在每次梯度更新前，先计算**最坏扰动方向** $\epsilon^\ast$：
$$
\epsilon^\ast = \rho \frac{\nabla L(\theta)}{\|\nabla L(\theta)\|}.
$$
然后，在被扰动的参数 $\theta + \epsilon^\ast$ 处计算新的梯度：
$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta + \epsilon^\ast).
$$

### **4.3 对噪声的处理**
- **显式限制噪声影响**：通过主动寻找最坏方向并优化其影响，SAM 能减少模型对噪声数据的敏感性，提高泛化能力。
- **相比 L2 正则化更具主动性**：L2 正则化只是压缩参数，而 SAM 直接在梯度空间中调整参数，使得模型更加鲁棒。

### **4.4 约束的方式**
- **直接在梯度更新中限制梯度方向**，与 SGD、Adam 的方法不同，SAM 不是调整学习率，而是主动在梯度计算过程中修改更新方向。

---

## **5. 方法对比总结**
| 方法          | 目标           | 设计思路                         | 处理噪声方式                 | 约束方式             |
| ------------- | -------------- | -------------------------------- | ---------------------------- | -------------------- |
| **SGD**       | 直接优化损失   | 纯梯度下降                       | 依赖 mini-batch 采样引入噪声 | 无显式正则化         |
| **Adam**      | 自适应优化     | 动量+二阶矩估计                  | 调整学习率以减少振荡         | 通过动量平滑梯度变化 |
| **L2 正则化** | 控制模型复杂度 | 增加 $\lambda \|\theta\|^2_2$ 项 | 约束权重大小                 | 直接施加参数约束     |
| **SAM**       | 找到平坦极小点 | 在梯度前加扰动优化               | 显式优化最坏方向             | 直接修改梯度方向     |

---
**结论**：
- **SGD 和 Adam** 主要通过学习率和动量调节梯度更新，对噪声的控制是被动的；
- **L2 正则化** 通过直接约束参数大小来减少噪声影响；
- **SAM** 则是最积极的方案，直接在梯度优化中限制最坏扰动，提高鲁棒性和泛化能力。

## **优化方法对比：L2 正则化, SGD, Adam, SAM, LPF-SGD, Flat-LoRA**
本文提到的 **Sharpness-Aware Minimization (SAM)** 通过限制噪声扰动增加梯度更新的限制，从而提高模型的泛化能力。与此同时，常见的优化方法如 **L2 正则化**, **SGD**, **Adam**, **LPF-SGD**, **Flat-LoRA** 也都以不同方式进行噪声处理和正则化。在本节中，我们对比这些方法的**目标、数学公式、约束噪声的方法以及它们之间的联系和区别**。

---

## **1. 各方法的目标**
| 方法                                   | 目标                     | 主要思想                                 | 主要约束手段                     |
| -------------------------------------- | ------------------------ | ---------------------------------------- | -------------------------------- |
| **L2 正则化 (权重衰减)**               | 防止权重过大，减少过拟合 | 在损失函数加入 $L_2$ 罚项，限制权重增长  | 直接对权重施加 $L_2$ 范数惩罚    |
| **SGD (随机梯度下降)**                 | 训练优化，使损失下降     | 依据梯度方向更新权重                     | 无显式噪声约束，仅靠学习率调节   |
| **Adam (自适应梯度下降)**              | 加快收敛，减少梯度震荡   | 结合动量与二阶梯度估计自适应调整学习率   | 通过一阶 & 二阶矩估计调整步长    |
| **SAM (Sharpness-Aware Minimization)** | 逼近平坦极小点，提高泛化 | 在梯度更新前计算最坏扰动方向             | 额外求解梯度扰动，避免尖锐极小点 |
| **LPF-SGD (Low-Pass Filtering SGD)**   | 降低噪声影响，优化平坦解 | 对梯度进行高斯核卷积，进行噪声滤波       | 通过蒙特卡洛方法对梯度扰动取均值 |
| **Flat-LoRA**                          | 微调大模型，增强泛化     | 在低秩子空间上进行扰动，使合并后权重平坦 | 只在 LoRA 低秩参数空间上施加扰动 |

---

## **2. 方法数学公式及设计思路**

### **2.1 L2 正则化 (Weight Decay)**
**目标**：限制参数幅度，防止过拟合。  
**公式**：
$$
L_{\text{reg}}(W) = L(W) + \lambda \|W\|_2^2.
$$
**约束手段**：
- 在损失函数中添加 $L_2$ 惩罚项，控制权重规模。
- 训练过程中，每一步梯度下降同时计算：
  $$
  W \leftarrow W - \eta (\nabla L(W) + 2\lambda W).
  $$
- 适用于任何优化器，如 **SGD with Momentum**, **Adam** 等。

**噪声控制方式**：
- **直接约束权重值**，但不显式控制梯度扰动。
- 对 **噪声** 的影响：能减少梯度爆炸，但对随机扰动的抑制效果有限。

---

### **2.2 SGD (Stochastic Gradient Descent)**
**目标**：通过随机梯度下降最小化损失。  
**公式**：
$$
W_{t+1} = W_t - \eta \nabla L(W_t).
$$
**约束手段**：

- 通过学习率 $\eta$ 控制步长，避免过大更新。
- 若加上动量 (Momentum)，则：
  $$
  v_t = \beta v_{t-1} + (1-\beta) \nabla L(W_t),
  $$
  $$
  W_{t+1} = W_t - \eta v_t.
  $$

**噪声控制方式**：
- 无显式噪声约束，**SGD 的随机性** 本身可提供一定的正则化效果。
- 但在高维数据下，SGD 容易受梯度爆炸或振荡影响，导致不稳定优化。

---

### **2.3 Adam (Adaptive Moment Estimation)**
**目标**：加快收敛，同时稳定梯度更新。  
**公式**：
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla L(W_t),
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) \nabla L(W_t)^2.
$$
$$
W_{t+1} = W_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}.
$$
**约束手段**：
- 自适应调整学习率，防止梯度爆炸或梯度消失。
- 适用于**稀疏数据**优化，如 NLP 任务。

**噪声控制方式**：
- 通过调整**二阶矩估计**来降低梯度噪声影响。
- 但仍然无法保证最终优化点是**平坦极小点**。

---

### **2.4 SAM (Sharpness-Aware Minimization)**
**目标**：避免收敛到“尖锐”极小点，提高泛化能力。  
**核心优化目标**：
$$
\min_W \max_{\|\epsilon\|_p \leq \rho} L(W + \epsilon).
$$
**梯度更新**：

1. 计算当前梯度：
   $$
   g = \nabla L(W).
   $$
2. 计算最坏扰动方向：
   $$
   \epsilon = \rho \frac{g}{\|g\|}.
   $$
3. 计算扰动后参数梯度：
   $$
   W' = W + \epsilon.
   $$
   $$
   g' = \nabla L(W').
   $$
4. 进行最终参数更新：
   $$
   W_{t+1} = W_t - \eta g'.
   $$

**噪声控制方式**：
- 通过对梯度方向施加“最坏扰动”，避免收敛到陡峭极小点。
- 但 **计算开销较大**，需要额外计算一次梯度。

---

### **2.5 LPF-SGD (Low-Pass Filtering SGD)**
**目标**：通过对梯度进行**平滑化**处理，提高优化稳定性。  
**公式**：
$$
(L \ast K)(W) = \mathbb{E}_{\tau \sim \mathcal{N}(0,\sigma^2 I)}[L(W - \tau)].
$$
**梯度更新**：
1. 采样 $M$ 组扰动 $\tau_i$：
   $$
   \tau_i \sim \mathcal{N}(0,\sigma^2 I).
   $$
2. 计算期望梯度：
   $$
   g = \frac{1}{M} \sum_{i=1}^M \nabla L(W - \tau_i).
   $$
3. 更新权重：
   $$
   W_{t+1} = W_t - \eta g.
   $$

**噪声控制方式**：
- 通过 **蒙特卡洛采样** 进行梯度平滑，减少过度学习高频噪声。
- 比 SAM **计算开销更低**，但效果略弱。

---

### **2.6 Flat-LoRA**
**目标**：针对大模型微调，在 **LoRA 低秩子空间** 进行优化，使合并后权重处于更平坦极小点。  
**梯度更新**：

1. **LoRA 结构**：
   $$
   W' = W + \Delta W = W + A B.
   $$
2. **计算扰动**（但仅在 $A, B$ 低秩子空间内）：
   $$
   \Delta A = \rho \frac{\nabla_A L}{\|\nabla_A L\|}.
   $$
3. **扰动优化**：
   $$
   (A,B) \leftarrow (A,B) - \eta \nabla_{A,B} L(W + \Delta A B).
   $$

**噪声控制方式**：
- 只在 LoRA 低秩空间施加扰动，**避免额外计算完整权重扰动**。
- 适用于 **大规模模型微调**，比 SAM 更高效。

---

## **3. 总结**
- **L2 正则化** 直接对权重进行控制，但不考虑梯度扰动。
- **SGD / Adam** 在优化过程中无显式平坦度控制。
- **SAM / LPF-SGD / Flat-LoRA** 通过**扰动梯度**来主动逼近平坦解，提高泛化能力，但计算量不同。
- **SAM 适用于一般深度学习任务，LPF-SGD 提供更高效替代方案，Flat-LoRA 适合大模型微调。**

## **优化方法公式对比：L2 正则化, SGD, Adam, SAM**

在深度学习优化过程中，**L2 正则化、SGD、Adam 和 SAM** 具有不同的数学优化公式。它们的本质区别在于 **如何计算梯度、如何更新参数、如何影响优化方向**。以下对各方法的公式进行详细介绍，并解释其数学直觉。

---

## **1. L2 正则化（权重衰减, Weight Decay）**
### **1.1 优化目标**
L2 正则化（又称权重衰减）是通过在损失函数 $L(W)$ 中添加一个 $L_2$ 范数惩罚项，使得参数不会增长得过大，从而降低过拟合风险：
$$
L_{\text{reg}}(W) = L(W) + \frac{\lambda}{2} \|W\|^2_2.
$$
其中：
- $L(W)$ 是原始损失函数（如交叉熵损失）。
- $\|W\|^2_2$ 是模型参数 $W$ 的 $L_2$ 范数。
- $\lambda$ 是正则化超参数，控制权重衰减的强度。

### **1.2 梯度计算**
L2 正则化对损失的梯度影响如下：
$$
\nabla_W L_{\text{reg}}(W) = \nabla_W L(W) + \lambda W.
$$
即：
- 计算损失 $L(W)$ 对权重 $W$ 的梯度。
- 额外加上 $\lambda W$，迫使参数 $W$ 变小。

### **1.3 参数更新**
通常使用梯度下降进行更新：
$$
W \gets W - \eta \left( \nabla_W L(W) + \lambda W \right).
$$
其中 $\eta$ 为学习率。

### **1.4 直觉**
L2 正则化相当于在训练过程中对参数施加了一个向心力，防止权重值变得过大，最终有助于提高模型的泛化能力。

---

## **2. 随机梯度下降 (SGD, Stochastic Gradient Descent)**
### **2.1 标准 SGD 公式**
SGD 是最基础的优化方法，其更新规则如下：
$$
W \gets W - \eta \nabla_W L(W),
$$
其中：
- $\nabla_W L(W)$ 是基于 mini-batch 计算出的梯度。
- $\eta$ 是学习率。

### **2.2 带动量的 SGD**
为了使 SGD 训练更加稳定，常加入动量（Momentum）：
$$
v_t = \beta v_{t-1} + (1 - \beta) \nabla_W L(W),
$$
$$
W \gets W - \eta v_t.
$$
其中：
- $v_t$ 是梯度的指数移动平均。
- $\beta$ 是动量系数（通常设为 $0.9$）。
- 动量可以减少梯度的高频振荡，使收敛更快。

### **2.3 直觉**
SGD 具有较强的随机性，能够跳出局部最优，但当数据集包含噪声时，它可能会受到噪声梯度的影响，导致泛化能力下降。

---

## **3. Adam（Adaptive Moment Estimation）**
Adam 结合了 **Momentum** 和 **RMSprop**，在每个参数维度上自适应调整学习率。

### **3.1 计算梯度的一阶矩估计（均值）**
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_W L(W).
$$
- 这是对梯度的指数加权移动平均（类似 Momentum）。
- $\beta_1$ 一般设为 $0.9$，表示过去 90% 的梯度会被保留。

### **3.2 计算梯度的二阶矩估计（方差）**
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_W L(W))^2.
$$
- 这是对梯度平方的指数加权移动平均，类似于 RMSprop 。
- $\beta_2$ 一般设为 $0.999$，确保更新更稳定。

### **3.3 计算偏差修正**
由于 $m_t$ 和 $v_t$ 初始值较小，Adam 进行修正：
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}.
$$

### **3.4 更新参数**
$$
W \gets W - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.
$$
其中 $\epsilon$ 是防止除零的极小数（如 $10^{-8}$）。

### **3.5 直觉**
Adam 通过**调整每个参数的学习率**，在高梯度方向减小步长，在低梯度方向增大步长，适用于深度神经网络和 NLP 任务。

---

## **4. SAM（Sharpness-Aware Minimization）**
SAM 旨在避免收敛到“尖锐极小点”，通过计算最坏扰动梯度，使优化趋向于平坦区域，提高泛化能力。

### **4.1 SAM 的优化目标**
$$
\min_W \max_{\|\epsilon\| \leq \rho} L(W + \epsilon).
$$
其中：
- 内层 $\max$ 表示寻找在小扰动范围 $\|\epsilon\| \leq \rho$ 内，使损失最大的方向。
- 外层 $\min$ 让优化器在最坏情况下仍能优化损失。

### **4.2 计算最坏扰动**
SAM 首先计算梯度：
$$
\epsilon^* = \rho \frac{\nabla_W L(W)}{\|\nabla_W L(W)\|}.
$$
然后在扰动点计算梯度：
$$
\nabla_W L(W + \epsilon^*).
$$

### **4.3 参数更新**
$$
W \gets W - \eta \nabla_W L(W + \epsilon^*).
$$

### **4.4 直觉**
- **标准优化（如 SGD）可能收敛到尖锐极小点**，这些极小点的梯度变化剧烈，对噪声较敏感，泛化能力较弱。
- **SAM 通过计算最坏扰动，使优化过程趋向于平坦极小点**，从而提高泛化能力，使模型对噪声数据更具鲁棒性。

---

## **5. 方法对比总结**
| **方法**      | **优化目标**                   | **梯度计算**                                  | **参数更新**                                                 | **主要作用**                 |
| ------------- | ------------------------------ | --------------------------------------------- | ------------------------------------------------------------ | ---------------------------- |
| **L2 正则化** | 限制权重大小，防止过拟合       | $\nabla_W L(W) + \lambda W$                   | $W \gets W - \eta (\nabla_W L(W) + \lambda W)$               | 避免权重过大，提高泛化能力   |
| **SGD**       | 直接最小化损失                 | $\nabla_W L(W)$                               | $W \gets W - \eta \nabla_W L(W)$                             | 计算简单，但易受噪声影响     |
| **Adam**      | 适应性调整学习率，提高优化效率 | 使用梯度的一阶、二阶矩估计                    | $W \gets W - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$ | 适用于深度网络，学习率自适应 |
| **SAM**       | 寻找平坦极小点，提高泛化能力   | 计算最坏扰动梯度 $\nabla_W L(W + \epsilon^*)$ | $W \gets W - \eta \nabla_W L(W + \epsilon^*)$                | 规避尖锐极小点，提高泛化能力 |

---







## **5. 方法对比总结**
| **方法**                              | **优化目标**                                 | **梯度计算**                                  | **参数更新**                                                 | **主要作用**                               |
| ------------------------------------- | -------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------ |
| **L2 正则化**                         | 限制权重大小，防止过拟合                     | $\nabla_W L(W) + \lambda W$                   | $W \gets W - \eta (\nabla_W L(W) + \lambda W)$               | 避免权重过大，提高泛化能力                 |
| **SGD**                               | 直接最小化损失                               | $\nabla_W L(W)$                               | $W \gets W - \eta \nabla_W L(W)$                             | 计算简单，但易受噪声影响                   |
| **Adam**                              | 适应性调整学习率，提高优化效率               | 使用梯度的一阶、二阶矩估计                    | $W \gets W - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$ | 适用于深度网络，学习率自适应               |
| **SAM**                               | 寻找平坦极小点，提高泛化能力                 | 计算最坏扰动梯度 $\nabla_W L(W + \epsilon^*)$ | $W \gets W - \eta \nabla_W L(W + \epsilon^*)$                | 规避尖锐极小点，提高泛化能力               |
| **SWA (Stochastic Weight Averaging)** | 通过权重平均找到更平坦的极小点，提高泛化能力 | 标准梯度 $\nabla_W L(W)$                      | $W_{\text{SWA}} \gets \frac{1}{k} \sum_{i=1}^{k} W_i$        | 通过周期性采样的模型权重平均化，提高鲁棒性 |

---

### **SWA 解析**
1. **SWA（Stochastic Weight Averaging, 随机权重平均）** 的核心思想是：在训练过程中采样多个优化步骤的权重，并对这些权重取平均值，以获得更平坦的损失曲面，提高泛化能力。
2. **优化方式**：
   - 训练过程中，每隔一定的步数保存当前模型权重 $W_t$。
   - 在训练结束后，对多个权重取平均：
     $$
     W_{\text{SWA}} = \frac{1}{k} \sum_{i=1}^{k} W_i.
     $$
   - 该平均化操作可降低模型的尖锐度，使得最终权重落入更平坦的极小点区域，提高泛化能力。
3. **与 SAM 的不同点**：
   - **SAM** 通过梯度扰动直接在训练过程中避免收敛到尖锐极小点；
   - **SWA** 通过后处理方法，在多个训练阶段采样模型权重并求平均，从而间接获得平坦极小点。
4. **与 SGD/Adam 的不同点**：
   - **SGD/Adam** 仅优化单个模型权重，而 SWA 通过**多个权重的平均化**提高模型的稳定性。

**适用场景**：
- 适用于 **深度神经网络、大规模训练任务**，特别是在高维优化空间中有助于减少过拟合。



# 个人总结

Given:
  - Large pretrained weights W (frozen or partially trainable in certain blocks).
  - LoRA parameters DeltaW, rank r, small dimension subspace.
  - A 'merge' operation: Merge(W, DeltaW) => W_merged (only for forward pass).
  - A function RandPerturb(x) => x + ε,   (One-step or random strategy).
  - A small subset of layers S to also partially update beyond LoRA.

For each iteration t:
  1. // Gradient Accumulation
     for each micro-batch in an accumulation cycle:
       - W_merged = Merge(W, DeltaW)   // no extra large copy, done on-the-fly
       - Possibly do W_merged' = RandPerturb(W_merged)
       - Compute loss = L(W_merged') in FP16/BF16
       - Accumulate gradient wrt DeltaW (+ partial blocks in S if any)
    
  2. // Single update step after accumulation
     - DeltaW := DeltaW - η * accumulated_grad(DeltaW)
       (Optionally combined with partial updates on W[S])
     - If using "EMA on DeltaW", do deltaW_ema = alpha * deltaW_ema + (1-alpha)*DeltaW

  3. Periodically do some refine or "clip" operation on DeltaW 
     to maintain low-rank or re-init random directions if needed.

End



# 数学基础总结 

