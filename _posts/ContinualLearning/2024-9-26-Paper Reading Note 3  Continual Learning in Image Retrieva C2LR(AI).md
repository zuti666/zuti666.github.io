---
layout: post
title:  Paper Reading Note3---Continual Learning in Image Retrieval C2LR
categories: [Paper Reading Note, Continual Learning, Image Retrieval ] 
description: Paper Reading Note3---Continual Learning in Image Retrieval AI  Summary 
keywords: [Paper Reading Note, Continual Learning, Image Retrieval, AI Summary ] 

---



# CL2R: Compatible Lifelong Learning Representations 

## 论文信息

**标题**: CL2R: Compatible Lifelong Learning Representations  

## 设定

本文研究了终身学习（Lifelong Learning, LL）中的兼容表示问题，特别是在视觉搜索任务中，如何在增量学习过程中保持特征表示的兼容性。

为了在增量学习过程中实现兼容特征表示，本文提出了一种新的训练程序，鼓励所学习特征的全局和局部平稳性。平稳性是指特征的统计性质不会随着时间而改变，这使得新学习的特征能够与之前学习的特征互操作。通过这种方式，模型能够在不遗忘旧特征的情况下，逐步学习新特征。

### 目标

目标是设计一个训练过程来学习模型 $ \phi_t $，使得经过该模型变换的任何查询图像都可以通过某种距离函数 


$$
 \text{dist} : \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}^+ 
 $$ 

执行视觉搜索，从而识别出与查询特征 $F_Q$ 最近的特征 $F_G$，而不会遗忘先前的特征表示，也无需重新计算 


$$
F_G = { f \in \mathbb{R}^d | f = \phi_t(x), \forall x \in I_G }
$$


（即不需要重新索引）。

如果这一点成立，那么生成的表示 $ \phi_t $ 就被认为与 $ \phi_k $ 是终身兼容的。CL2R 问题的主要挑战在于同时缓解灾难性遗忘并学习一种与先前模型兼容的表示。

### 创新点

1. **兼容终身学习表示（CL2R）**: 提出了一个新概念，即在终身学习范式下进行兼容表示学习，旨在解决灾难性遗忘和特征兼容性之间的冲突。
2. **平稳性训练策略**: 提出了一种新的训练程序，鼓励全局和局部平稳性，通过回放策略（rehearsal）和特征平稳性联合解决灾难性遗忘和特征兼容性问题。
3. **新评估指标**: 定义了一套新的指标来专门评估在灾难性遗忘下的兼容表示学习性能。

## 方法

### 方法的设计思想

CL2R（Compatible Lifelong Learning Representations）方法的设计思想旨在解决增量学习中灾难性遗忘（catastrophic forgetting）和特征表示兼容性之间的冲突。其核心思想是通过引入全局和局部平稳性，来保证在学习新任务时，新旧特征表示能够保持兼容性。该方法依赖于以下几个关键设计思想：

#### **全局平稳性（Global Stationarity）**

**全局稳定性**是 CL2R 提出的一种方法，旨在确保新旧任务的特征表示在**全局特征空间中保持一致性**，即从整体上限制新旧模型生成的特征表示发生显著变化。

**具体做法**：

- **固定分类器原型（Fixed Classifier Prototype）**：CL2R 引入了一个固定的分类器原型——**Simplex固定分类器**。该分类器的原型向量在特征空间中保持不变，用于引导新模型的特征方向与旧模型的特征方向保持一致。
- **特征方向对齐**：通过固定的分类器原型，CL2R 强制新任务的特征向量与旧任务特征向量的方向对齐，从而减少特征方向的偏移。由于特征空间中的特征向量表示的是样本的高维嵌入，这一方法确保了新旧模型特征空间在宏观上保持一致性。

**作用**：

- 通过引入全局特征的稳定性约束，CL2R 保证了不同任务的特征表示在全局特征空间中的方向保持一致，从而减少特征表示的大规模漂移。全局稳定性约束尤其适合处理新旧任务之间特征表示可能存在的显著差异。

#### **局部平稳性（Local Stationarity）**

除了全局上的特征对齐，CL2R 还提出了**局部稳定性**机制，旨在保证新任务学习时，每个样本的特征在新旧模型中的局部表示保持一致。

**具体做法**：

- 特征蒸馏损失（Feature Distillation Loss）

  ：CL2R 使用特征蒸馏机制来保证局部特征一致性。特征蒸馏是通过让新模型模仿旧模型在特定样本上的特征表示来实现的。对于每个样本 $x_i$，新模型的特征表示 $f_{new}(x_i)$ 需要与旧模型的特征表示 $f_{old}(x_i)$ 保持一致，公式如下：

  

$$
L_{distill} = \frac{1}{N} \sum_{i=1}^{N} \| f_{new}(x_i) - f_{old}(x_i) \|^2
$$



通过最小化新旧模型特征向量之间的 L2 范数差异，确保新模型在局部样本上的特征与旧模型保持一致。

   **作用**：

   - **局部特征稳定性**确保了每个样本在特征空间中的局部嵌入不会发生显著漂移。即使模型在学习新任务时，个别样本的特征表示也能够与旧任务中的表示保持一致，从而增强特征表示的兼容性。局部稳定性可以减少对单一样本的灾难性遗忘，有助于提升整体兼容性。

**回放策略（Rehearsal Strategy）**:  为了进一步增强模型对旧任务的记忆能力，CL2R 还结合了**回放策略**，即在训练新任务时，模型还会使用部分旧任务的样本来帮助保持旧任务特征的稳定性。通过使用情景记忆（episodic memory），保存之前任务中的部分样本，并在训练新任务时使用这些样本。这样可以在增量学习过程中，通过重新使用这些样本来强化对旧任务的记忆，从而缓解灾难性遗忘。

**具体做法**：

- **记忆库（Memory Buffer）**：CL2R 维护一个记忆库，其中存储了旧任务中的一部分样本。每次训练新任务时，模型不仅使用新任务的数据，还会从记忆库中采样旧任务的样本，并将其与新任务一起训练。
- **联合训练**：通过同时使用新任务样本和记忆库中的旧任务样本进行训练，模型能够在学习新任务时，保留对旧任务的记忆。这样可以有效减少灾难性遗忘，确保新模型与旧模型特征之间的兼容性。

**作用**：

- **回放策略**通过结合旧任务样本和新任务样本进行联合训练，进一步提升了特征表示的兼容性和稳定性。特别是在应对长序列的增量学习任务时，回放策略可以有效减少灾难性遗忘。

### 原理

CL2R方法的核心原理在于确保特征表示的平稳性，这使得新旧特征表示在特征空间中具有较高的兼容性。具体来说，CL2R利用以下两种主要机制来实现兼容性：

1. **全局特征平稳性**:  使用一个固定的分类器原型（$d$-Simplex Regular Polytope）来约束特征表示，使得学习的特征始终与固定的原型对齐。通过这种方式，可以实现特征表示的全局平稳性，从而保证不同任务间特征表示的兼容性。

2. **局部特征平稳性**:  特征蒸馏（Feature Distillation）用于在当前模型与前一个模型之间保持特征表示的一致性。通过最小化新旧模型在特征空间的差异，CL2R方法有效地防止了由于模型更新而导致的特征表示变化，从而实现了局部特征平稳性。

### 与其他方法的区别

CL2R方法与其他增量学习方法的主要区别在于其独特的特征表示兼容性设计：

- **传统增量学习方法（如LwF, LUCIR, BiC, PODNet）**：这些方法主要关注减少灾难性遗忘，通过知识蒸馏和经验重放（experience replay）等策略来保持对旧任务的记忆，但它们并未显式关注特征表示的兼容性，**通常在特征空间中会有显著的变化**。
- **FAN 和 BCT 方法**：这些方法试图在特征空间中保持兼容性，但通常**通过固定模型或映射函数的方式来实现**。然而，它们的性能会随着任务数量的增加而下降，因为它们在适应新任务的过程中难以有效处理大量的特征映射变化。
- **CL2R 方法**：与上述方法不同，CL2R不仅通过知识蒸馏来缓解灾难性遗忘，还通过全局和局部平稳性设计来确保新旧特征表示的兼容性。这使得CL2R方法在不需要重新索引的情况下实现高效的增量学习，尤其适用于大规模视觉搜索任务。

### 技术细节

#### **全局平稳性训练策略**

**固定分类器原型**: 使用$d$-Simplex正多面体作为固定分类器原型，定义为



$$
   W = \{e_1, e_2, \ldots, e_{d-1}, \alpha \sum_{i=1}^{d-1} e_i\}，
$$

   

其中 $\alpha = 1 - \sqrt{\frac{d+1}{d}}$ ， $e_i$ 是 $R^{d-1}$ 中的标准基。

   **损失函数**: CL2R使用改进的交叉熵损失，公式如下：



$$
L_t = - \frac{1}{|T_t|} \sum_{x \in T_t} \log \left( \frac{\exp(w_{y_i}^\top \cdot \phi(x))}{\sum_{j \in K_s} \exp(w_j^\top \cdot \phi(x)) + \sum_{j \in K_u} \exp(w_j^\top \cdot \phi(x))} \right)
$$

   

   其中，

   $T_t$是当前任务的训练集，

   $K_s$是到当前任务为止已经学习的类别集合，

   $K_u$是未来 未见过的类别集合

   $w^T_{(\cdot)}$  是固定分类器$W$ 的类原型

   $W$ 是固定分类器的权重矩阵，该分类器在模型训练期间不进行学习

   $y_i$ 是有监督学习的标签

   $\phi$ 是将查询图像(query image)转换为特征向量 (feature vectors)

   

   

   

#### **局部平稳性训练策略**

- **特征蒸馏损失（Feature Distillation Loss, FD）**: 用于在局部特征空间中保持新旧模型之间的特征表示一致性。FD损失在每个任务$t$上，仅在情景记忆$M_t$中存储的样本上进行评估：

$$
L_{\text{FD}}^M = \frac{1}{|M_t|} \sum_{x_i \in M_t} \left( 1 - \frac{\phi_t(x_i) \cdot \phi_{t-1}(x_i)}{\|\phi_t(x_i)\| \|\phi_{t-1}(x_i)\|} \right)
$$

其中，$\phi_t$表示当前任务的模型，$\phi_{t-1}$表示前一个任务的模型。



#### **最终损失函数**

- **综合损失函数**: 最终优化的损失函数是全局和局部对齐提供的两个损失之和：

$$
L = L_t + \lambda L_{\text{FD}}^M
$$

其中，$\lambda$用于平衡全局和局部对齐的贡献。

## 实验

### 实验设置与参数

- **数据集**: 实验在多个基准数据集上进行，包括 CIFAR10, ImageNet201, ImageNet100, Labeled Face in the Wild (LFW) 和 IJB-C。
- **模型**: 使用 ResNet-32, ResNet-18 和 ResNet-50 作为特征提取器，分别在不同的任务中进行评估。
- **优化器**: 使用 SGD 优化器，初始学习率为 0.1，权重衰减为 2×10^-4。学习率在特定 epoch 后衰减。
- **训练过程**: 训练程序基于增量微调策略（incremental fine-tuning），每个任务训练多个 epoch，使用回放策略以缓解灾难性遗忘。

### 对比方法

本文将所提出的方法与多种现有的增量学习方法进行比较，包括：

- **Learning without Forgetting (LwF)**: 通过知识蒸馏来防止遗忘。
- **LUCIR**: 使用特征蒸馏减少灾难性遗忘的增量学习方法。
- **BiC**: 通过学习额外的线性层来重新校准输出概率。
- **PODNet**: 使用基于空间的蒸馏损失来约束每个残差块之后的中间特征统计。
- **FOSTER**: 解决兼容性问题的方法，通过训练线性层将不断增长的特征向量映射到固定维度。
- **FAN** 和 **BCT**: 提供显式机制来解决特征兼容性问题的方法。

### 结果与结论

实验结果表明，所提出的 CL2R 训练程序在多个基准数据集和不同的增量学习任务下都表现出色，相较于基线和最新的研究方法，显著减少了灾难性遗忘，同时保持了高水平的特征兼容性。尤其在面对多任务学习时，CL2R 训练程序能够有效地学习兼容特征表示，而无需频繁重新索引图库

### 结论

CL2R方法通过设计特征表示的全局和局部平稳性，成功实现了在增量学习过程中的特征兼容性。相较于其他方法，CL2R不仅有效减少了灾难性遗忘，还在大规模视觉搜索任务中无需重新索引就实现了高效的增量学习。这种方法的创新设计为未来的终身学习和特征表示兼容性研究提供了新的思路和工具。







# Simplex 



## Simplex 固定分类器是什么？

**Simplex 固定分类器（Simplex Fixed Classifier）** 是 CL2R 提出的一个关键机制，用于在持续学习中保持**全局特征稳定性（Global Stationarity）**。它通过一个固定的、几何上对称的分类器原型，来约束特征空间中的特征表示，使得新旧任务的特征能够在全局方向上保持一致。

在具体实现中，Simplex 固定分类器通过构造一个“Simplex”几何结构，其中每个类的原型向量在特征空间中以对称方式排列。这种结构确保了新旧任务特征嵌入方向的一致性，减少了特征空间的漂移。

## 什么是 Simplex 几何结构？

**Simplex** 是一个几何学术语，表示一种**对称的几何形状**。在 $d$ 维空间中，Simplex 是由 $d+1$ 个顶点构成的多面体，其特点是这些顶点在几何上是对称的，且每两个顶点之间的距离相等。

在 Simplex 固定分类器中，每个类的特征原型向量（prototype vector）被设计成一个 Simplex 结构，这意味着：

- 每个类的原型向量在特征空间中以对称的方式排列；
- 不同类的原型向量之间的角度保持恒定，从而在空间中形成均匀分布。

## Simplex 固定分类器的工作原理

在 CL2R 中，Simplex 固定分类器通过固定这些几何对称的类原型向量来引导新任务的特征表示，使它们的方向与旧任务保持一致。这种做法的主要目的是通过确保特征向量的方向稳定，减少新任务特征对旧任务特征空间的干扰，进而实现全局特征稳定性。

### 具体实现方式：

1. **固定类原型向量**：
   Simplex 分类器的每个类原型向量 $\mu_i$ 被固定为位于 Simplex 几何形状的一个顶点。通过这样设计，CL2R 中所有的类原型向量在训练过程中保持不变，即不随模型的更新或新任务的加入而改变。这种设计确保了特征空间中的全局几何结构在整个持续学习过程中保持稳定。

2. **新任务特征对齐**：
   在训练新任务时，新任务的特征表示 $f_{new}(x)$ 会被引导与 Simplex 分类器的固定原型向量对齐。这意味着新任务的特征表示必须符合原有几何结构的方向，从而避免新任务特征方向的偏移。

3. **全局特征方向一致性**：
   由于 Simplex 分类器的原型向量是对称且固定的，CL2R 能够确保新旧任务特征的方向在特征空间中保持一致。这样，即使新任务学习到了新的特征，整体特征空间的方向性依然被保持，减轻了特征漂移的问题。

### 数学形式：

假设 $C$ 是类的数量，Simplex 分类器的类原型向量 $\mu_1, \mu_2, \dots, \mu_C$ 是对称排列的，表示每个类的中心。对于一个输入样本 $x_i$，模型生成的特征向量 $f(x_i)$ 被引导与 Simplex 分类器中的某一个类原型 $\mu_k$ 对齐。整个过程通过最小化以下损失来实现：


$$
L_{global} = \sum_{i} \| f(x_i) - \mu_{y_i} \|^2
$$



其中，$y_i$ 是样本 $x_i$ 的真实标签，$\mu_{y_i}$ 是该样本所属类别的原型向量。

通过最小化这个损失函数，模型强制新生成的特征与对应类的固定原型对齐，从而确保特征方向的稳定性。

### 为什么原型向量保持不变？

在 CL2R 中，**Simplex 分类器的原型向量保持不变**，是为了确保整个特征空间的全局方向在引入新任务时不会发生显著变化。主要原因和效果如下：

1. **减少特征空间的漂移**：
   在持续学习过程中，如果每次引入新任务时都调整类的原型向量，那么特征空间会发生较大变化。这会导致模型在处理旧任务时产生灾难性遗忘。因此，通过固定原型向量，可以确保特征空间的全局几何结构保持稳定，减少新任务对旧任务的干扰。

2. **特征对齐和一致性**：
   原型向量的固定允许新任务的特征表示与旧任务的特征表示保持方向一致。即使模型在学习新任务时生成了新的特征，这些特征也会被引导与原有的固定方向对齐，从而确保特征的一致性。

3. **对称性带来的稳健性**：
   Simplex 分类器的几何对称性确保了每个类的原型向量之间的角度是对称的，并且这种对称结构不会因为新任务的引入而被破坏。这种对称性使得特征空间在全局上更加稳健和一致，有助于提升模型的鲁棒性。

### 如何实现 Simplex 分类器的固定原型？

要实现 Simplex 分类器的固定原型，可以通过以下步骤：

1. **构造 Simplex 原型向量**：
   在 $d$ 维空间中，选择 $C$ 个类原型 $\mu_1, \mu_2, \dots, \mu_C$，使得这些向量之间的角度均匀分布。几何上，这可以通过解等距离问题（等距顶点）来构造 Simplex 结构。每个原型向量的长度可以标准化为单位向量。

2. **固定原型向量**：
   一旦类原型向量构造完成，便在整个训练过程中固定不变。无论是旧任务还是新任务，所有任务的特征向量都会与这些固定的原型向量对齐。

3. **特征对齐损失**：
   在模型训练中，通过最小化样本特征与对应类原型向量之间的差异来进行训练。这样，即使学习新任务，特征空间的整体方向和结构依然被保持。

## 总结

**Simplex 固定分类器**是 CL2R 提出的一个关键机制，用于保持特征空间的全局稳定性。它通过几何对称的 Simplex 结构来固定分类器的类原型向量，确保新旧任务的特征表示在全局方向上保持一致性。这种方法的主要优点在于：

- **减少特征漂移**：通过固定类原型向量，确保新旧任务特征的全局方向一致，从而减少特征空间的漂移。
- **特征表示兼容性**：无论是旧任务还是新任务，所有任务的特征都与这些固定原型向量对齐，增强了特征表示的兼容性。

这种方法为持续学习中的特征表示提供了全局一致性，有助于在引入新任务时减少灾难性遗忘，同时保持旧任务的性能稳定。



