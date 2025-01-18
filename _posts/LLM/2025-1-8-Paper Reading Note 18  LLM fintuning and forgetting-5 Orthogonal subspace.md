---
layout: post
title:  Paper Reading 16 LLM finetuning and forgetting - 5 Orthogonal subspace
categories: [Paper Reading,  LLM, Continual Learning,] 
description:  [Learn or Recall? Revisiting Incremental Learning with Pre-trained Language Models]
keywords: [Paper Reading,  LLM, Continual Learning,  Orthogonal subspace] 
---



# Paper Reading 18  LLM finetuning and forgetting - 5 Orthogonal subspace





这里介绍一篇论文

Orthogonal Subspace Learning for Language Model Continual Learning

其主要使用了两个主要的概念 Orthogonal Gradient Descent  和 LoRA  vector space 

在介绍这篇论文之前，我首先介绍一下 Orthogonal Gradient Descent 的论文。



下面是论文列表

# 论文列表

- **Orthogonal Gradient Descent for Continual Learning**

  [`semanticscholar`](https://www.semanticscholar.org/paper/841c970f7ef35e28dbbe054d0a7c5df252533a4e)  [`Paper`](https://www.semanticscholar.org/paper/841c970f7ef35e28dbbe054d0a7c5df252533a4e)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F841c970f7ef35e28dbbe054d0a7c5df252533a4e%3Ffields%3DcitationCount)

  ​     

- **Orthogonal Subspace Learning for Language Model Continual Learning**

  [`semanticscholar`](https://www.semanticscholar.org/paper/28fde851680a40fbbc5c6a44bd3ac6f5ca4ad284)  [`Paper`](https://www.semanticscholar.org/paper/28fde851680a40fbbc5c6a44bd3ac6f5ca4ad284)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F28fde851680a40fbbc5c6a44bd3ac6f5ca4ad284%3Ffields%3DcitationCount)

  2023    Conference on Empirical Methods in Natural Language Processing 







- **Is Parameter Collision Hindering Continual Learning in LLMs?**

  [`semanticscholar`](https://www.semanticscholar.org/paper/f852e60dd32dc1f3f1f53ba8f76862f77c5cd8d2)  [`Paper`](https://www.semanticscholar.org/paper/f852e60dd32dc1f3f1f53ba8f76862f77c5cd8d2)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Ff852e60dd32dc1f3f1f53ba8f76862f77c5cd8d2%3Ffields%3DcitationCount)

  ​     





# 论文前序1  -- QGD

- **Orthogonal Gradient Descent for Continual Learning**

  [`semanticscholar`](https://www.semanticscholar.org/paper/841c970f7ef35e28dbbe054d0a7c5df252533a4e)  [`Paper`](https://www.semanticscholar.org/paper/841c970f7ef35e28dbbe054d0a7c5df252533a4e)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F841c970f7ef35e28dbbe054d0a7c5df252533a4e%3Ffields%3DcitationCount)

  ​     

![image-20250118191302143](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250118191302209.png)

## AI QA ---Summary the paper

Here is a comprehensive summary and analysis of the paper titled *Orthogonal Gradient Descent for Continual Learning* by Mehrdad Farajtabar, Navid Azizan, Alex Mott, and Ang Li:

### Main Content

#### **Scope and Setting**

The paper addresses the challenge of catastrophic forgetting in neural networks during continual learning (CL), where models trained sequentially on tasks forget previously learned knowledge.

#### **Purpose**

The aim is to propose a method that allows neural networks to retain prior knowledge effectively while learning new tasks, without needing to revisit or store old data.

#### **Key Idea**

The authors introduce the Orthogonal Gradient Descent (OGD) method. This approach modifies the direction of gradient updates to minimize interference with gradients associated with previous tasks.

#### **Method**

1. **Gradient Orthogonalization**: The gradients of new tasks are projected orthogonally to the subspace spanned by gradients of previous tasks, ensuring minimal disruption to prior knowledge.
2. **Memory Efficiency**: Only a subset of gradients from previous tasks is stored, mitigating storage requirements.
3. **Variants**:
   - **OGD-ALL**: Stores gradients for all logits.
   - **OGD-AVE**: Stores average gradients of logits.
   - **OGD-GTL**: Stores gradients for the ground truth logits (used in experiments).

#### **Contributions**

1. A novel method (OGD) to mitigate catastrophic forgetting.
2. Empirical evaluation on benchmarks (Permuted MNIST, Rotated MNIST, Split MNIST) showing state-of-the-art performance.
3. A method that is privacy-friendly by not requiring storage of raw data.

#### **Difference and Innovation**

OGD differs from other methods by working with model prediction gradients rather than loss gradients. This approach ensures better preservation of prior task knowledge and avoids problems like loss gradients becoming zero for well-fitted examples.

#### **Results and Conclusion**

- OGD performs competitively with and often exceeds state-of-the-art methods like A-GEM and EWC on various benchmarks.
- The method is robust in settings where tasks are similar but struggles when tasks are highly dissimilar (e.g., large rotations in Rotated MNIST).
- OGD requires careful tuning of the learning rate but provides a promising avenue for scalable and memory-efficient continual learning.

### Comparison with Other Methods

| Method | Storage Needs          | Gradient Basis       | Performance on CL Benchmarks              |
| ------ | ---------------------- | -------------------- | ----------------------------------------- |
| EWC    | Moderate               | Fisher Information   | Sensitive to overparameterization         |
| A-GEM  | High (stores raw data) | Loss gradients       | Competitive but requires data storage     |
| OGD    | Low (stores gradients) | Prediction gradients | Strong overall, especially on Split MNIST |

### Discussion

- **Advantages**: OGD does not require raw data storage, preserving user privacy and making it suitable for sensitive applications.
- **Limitations**: Performance degrades when tasks are highly dissimilar, and it is sensitive to large learning rates.
- **Future Directions**: Exploring high-order derivatives, better gradient storage prioritization, and extending the method to adaptive optimizers like Adam.

------

### Summary Table of Base Models and Datasets

| Model Architecture | Model Size          | Dataset        | Tasks                   |
| ------------------ | ------------------- | -------------- | ----------------------- |
| 3-layer MLP        | 100 units per layer | Permuted MNIST | Sequential Permutations |
| 3-layer MLP        | 100 units per layer | Rotated MNIST  | Rotation by angle       |
| Multi-head MLP     | 100 units per layer | Split MNIST    | Disjoint label sets     |

Let me know if you need further details or analysis.



## AI QA---Introduce the 3 variants 

In the context of the paper and neural networks, **logits** are the raw, unnormalized outputs of the last layer of the model, typically a fully connected layer. For a classification problem, logits represent the model's prediction scores for each class before applying an activation function like softmax.

### Key Definitions of Variants

The paper proposes three variants of the Orthogonal Gradient Descent (OGD) method, differing in which gradients are stored and how they are used:

1. **OGD-ALL**
   - **What it stores**: The gradients for all logits (scores for all classes).
   - **Key characteristic**: Maintains maximum information about the model’s prediction changes for all possible outputs.
   - **Memory Usage**: High, as it stores gradient vectors for all classes.
2. **OGD-AVE**
   - **What it stores**: The averaged gradients across all logits.
   - **Key characteristic**: Reduces memory requirements by summarizing the gradients, effectively losing fine-grained information about individual class scores.
   - **Memory Usage**: Lower than OGD-ALL.
3. **OGD-GTL (Ground Truth Logit)**
   - **What it stores**: The gradient for the logit corresponding to the ground truth class only (the class label of the given sample).
   - **Key characteristic**: Focuses exclusively on the gradient for the correct class, discarding information about other classes.
   - **Memory Usage**: Lowest among the three, as it stores only one gradient per data point.

### Differences Between the Variants and the Proposed Method

- **Storage Efficiency**:
  - OGD-GTL is the most memory-efficient as it focuses solely on the ground truth class gradient.
  - OGD-ALL is memory-intensive, storing gradients for all logits, which scales with the number of classes.
  - OGD-AVE balances memory usage by summarizing information but sacrifices specificity.
- **Performance**:
  - **OGD-ALL**: Best at preserving comprehensive model knowledge but requires significant storage, which may not be practical for many tasks.
  - **OGD-AVE**: May lose performance compared to the other variants due to the reduction in stored information, especially for more complex tasks.
  - **OGD-GTL**: Offers a good trade-off, showing comparable or slightly better performance than OGD-ALL in many experiments while requiring significantly less memory.
- **Applicability**:
  - **OGD-GTL** is favored in the paper for practical experiments because it maintains strong performance while being memory-efficient.

### Comparison with the Original OGD Proposal

The **original OGD method** does not inherently specify which gradients (logits, averages, or ground truth) to store; it focuses on the orthogonalization of the gradient updates. These three variants implement OGD differently based on the choice of gradients to store, optimizing for various trade-offs in memory and performance.

Let me know if you need a deeper explanation of logits, orthogonalization, or other aspects!



## AI---QA Clarity the Figure 1 design

![](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250118191307789.png)

To address your question based on the paper's content and Figure 1:

### Ensuring Subspace Orthogonality

The method ensures the subspace of Task A remains orthogonal to Task B by projecting the gradients for Task B onto the orthogonal complement of the gradient subspace of Task A. The process is as follows:

1. **Gradient Storage from Task A**:

   - During training on Task A, the gradients of the model predictions (`rf(x; w)`) are computed and stored for all or a subset of samples from Task A.
   - This forms a gradient subspace that represents the directions in parameter space critical for retaining knowledge of Task A.

2. **Orthogonal Projection for Task B**:

   - When Task B arrives, the loss gradients (`g`) are computed for the current task.

   - Before applying these gradients to update the model, they are projected onto the orthogonal complement of Task A's gradient subspace: 

     

   - $$
     \tilde{g} = g - \sum_{v \in S} \text{proj}_v(g),
     $$

     

     where SS is the set of stored gradients from Task A, and $\text{proj}_v(g)$ projects gg onto the direction $v$.

   - This ensures that updates for Task B minimally interfere with the gradients crucial for retaining Task A's knowledge.

### Can the Model Still Learn to Classify if the Gradient Changes?

Yes, the model can still learn to classify Task B even with the modified gradients. Here’s why:

1. **Preservation of Task A Knowledge**:

   - By projecting Task B's gradients orthogonally, the model avoids altering directions in parameter space that are critical for Task A. This minimizes forgetting while leaving enough capacity for Task B.

2. **Learning New Information**:

   - The projected gradient g~\tilde{g} is still a valid descent direction for Task B's loss function. This is because the projection only removes components that interfere with Task A, retaining components that contribute to learning Task B: 

   - 

   - $$
     \langle -\tilde{g}, g \rangle \leq 0, 
     $$

     

     which ensures that moving in the direction of g~\tilde{g} decreases Task B’s loss.

3. **Capacity Utilization**:

   - Neural networks are typically overparameterized, meaning there are sufficient degrees of freedom in the parameter space to satisfy the orthogonality constraint while still learning new tasks.

### Intuition from Figure 1

Figure 1 illustrates this concept: the original gradient gg from Task B is modified to g~\tilde{g}, which lies in the blue subspace orthogonal to Task A's gradient directions. Moving within this orthogonal subspace allows the model to learn Task B while ensuring predictions for Task A remain stable.

### Potential Challenges

- If the tasks are highly dissimilar (e.g., rotations >90° in Rotated MNIST), the orthogonal subspace may contain limited useful directions for Task B. In such cases, learning new tasks while retaining old ones becomes more challenging.

This approach balances learning and retention effectively in most scenarios while exploiting the high capacity of neural networks.


In the context of the paper, **orthogonal** refers specifically to the gradients and not the feature embeddings. Here's a breakdown:

### What Does Orthogonal Mean Here?

1. **Orthogonal Gradients**:

   - The concept of orthogonality in this paper applies to the **gradients** of the model's predictions with respect to its parameters.
   - When the paper says Task A and Task B are "orthogonal," it means that the gradient updates for Task B are projected onto a subspace that is orthogonal to the subspace spanned by the gradients of Task A.
   - This ensures that the updates for Task B do not interfere with the directions in the parameter space that are critical for retaining knowledge about Task A.

2. **Key Equation**: The paper formalizes this by requiring:

   
   $$
   \tilde{g} \perp \text{span}\left(\{\nabla f_j(x; w^*_A)\}\right), \forall x \in T_A, j \in [1, c],
   $$
   

   where $\nabla f_j(x; w^*_A)$ represents the gradient of the logit jj for input xx in Task A.

------

### Is It About Gradients or Feature Embeddings?

It is about **gradients**, not feature embeddings. Specifically:

- **Gradients**: Orthogonality refers to the mathematical property that the gradient directions for Task B are made perpendicular (orthogonal) to the gradient subspace of Task A.
- **Feature Embeddings**: The paper does not explicitly enforce orthogonality between the feature embeddings of Task A and Task B. Instead, the method ensures that the updates to the model parameters for Task B do not interfere with those critical for Task A.

------

### Why Gradients, Not Feature Embeddings?

1. **Gradients Represent Learning Directions**:
   - Gradients directly influence how the model's parameters are updated during backpropagation.
   - By projecting gradients orthogonally, the method directly controls the parameter changes, ensuring minimal interference with Task A.
2. **Feature Embeddings Are Indirect**:
   - While feature embeddings (intermediate representations of input data) might change as the model learns new tasks, the paper does not explicitly manipulate them. Instead, the method works at the gradient level to indirectly preserve task-relevant features.

------

### Intuitive Summary

- **Orthogonality of Gradients**: Ensures that the parameter updates for Task B do not "erase" knowledge relevant to Task A.
- **Feature Embeddings**: May or may not be orthogonal depending on the tasks, but the method does not enforce this explicitly.

If you'd like more details on why gradients are the focus or how embeddings might interact indirectly, feel free to ask!





论文前序2 LoRA

LoRA 的论文这里就不展开介绍，可以看另一篇论文



# 论文介绍 -- O-LoRA



Orthogonal Subspace Learning for Language Model Continual Learning

- **Orthogonal Subspace Learning for Language Model Continual Learning**

  [`semanticscholar`](https://www.semanticscholar.org/paper/28fde851680a40fbbc5c6a44bd3ac6f5ca4ad284)  [`Paper`](https://www.semanticscholar.org/paper/28fde851680a40fbbc5c6a44bd3ac6f5ca4ad284)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F28fde851680a40fbbc5c6a44bd3ac6f5ca4ad284%3Ffields%3DcitationCount)

  2023    Conference on Empirical Methods in Natural Language Processing 



![image-20250118193547333](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250118193547459.png)



![image-20250118193524331](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250118193524461.png)

## AI QA --- Summary 

### Summary of the Paper: Orthogonal Gradient Descent for Continual Learning

1. **Scope/Setting**:
   - Focuses on continual learning, specifically addressing catastrophic forgetting in neural networks, where performance on earlier tasks degrades after learning new tasks.
2. **Purpose**:
   - Proposes the Orthogonal Gradient Descent (OGD) method to enable neural networks to learn new tasks without forgetting previously learned tasks.
3. **Key Idea**:
   - Gradient updates for new tasks are projected onto a subspace orthogonal to the gradients of previously learned tasks. This approach minimizes interference with past knowledge while still optimizing for the current task.
4. **Method**:
   - OGD computes and stores gradients related to previous tasks. For new tasks, the gradients are modified to ensure orthogonality with stored gradients using techniques like Gram-Schmidt orthogonalization.
   - Variants like OGD-GTL (gradient on ground truth logit) and OGD-AVE (average gradient) manage memory efficiency.
5. **Contributions**:
   - Avoids catastrophic forgetting without storing past task data, addressing privacy concerns.
   - Efficiently utilizes neural network capacity to balance performance across multiple tasks.
   - Demonstrates competitive performance across standard continual learning benchmarks.
6. **Differences/Innovation**:
   - Unlike existing methods, OGD operates directly on model gradients rather than loss gradients, preserving critical knowledge more effectively.
   - Does not require storing raw past data, making it advantageous in privacy-sensitive applications.
7. **Results**:
   - Outperforms or matches state-of-the-art methods like Elastic Weight Consolidation (EWC) and Averaged Gradient Episodic Memory (A-GEM) on benchmarks such as Permuted MNIST, Rotated MNIST, and Split MNIST.
   - Shows robustness across tasks with minimal forgetting.
8. **Discussion**:
   - Highlights sensitivity to learning rates and the challenges with highly dissimilar tasks.
   - Suggests potential extensions for better memory management and adaptation to higher learning rates.
9. **Comparison to Other Methods**:
   - Compared with SGD, EWC, A-GEM, and Multi-Task Learning (MTL):
     - OGD achieves better balance in retaining past knowledge while learning new tasks.
     - Less reliant on storing or replaying past data, unlike A-GEM or generative replay methods.

------

### Base Model and Dataset Details:

| **Aspect**            | **Details**                                                  |
| --------------------- | ------------------------------------------------------------ |
| **Base Model**        | A three-layer Multilayer Perceptron (MLP) with 100 hidden units per layer, ReLU activation, and 10 output logits for classification. |
| **Dataset(s)**        | Benchmarks for continual learning:                           |
|                       | - **Permuted MNIST**: MNIST digits permuted by different shuffling schemes. |
|                       | - **Rotated MNIST**: MNIST digits rotated by fixed angles.   |
|                       | - **Split MNIST**: Disjoint subsets of MNIST labels for different tasks. |
| **Variants Explored** | OGD-ALL (all logits), OGD-AVE (average logits), and OGD-GTL (ground truth logit). |
| **Optimization**      | Stochastic Gradient Descent (SGD) with a learning rate of 10−310^{-3}. |
| **Storage**           | Gradient storage limited to 200 vectors per task.            |
| **Training**          | Each task trained for 5 epochs, batch size of 10.            |

Let me know if you would like a deeper analysis or specific sections rewritten!



AI QA --- Relation with 

### Summary of the Paper: Orthogonal Subspace Learning for Language Model Continual Learning

#### **Main Content Analysis**

1. **Scope/Setting**:
   - This paper addresses continual learning challenges in large language models (LLMs), focusing on mitigating catastrophic forgetting, which arises when tasks are learned sequentially.
2. **Purpose**:
   - Introduces Orthogonal Low-Rank Adaptation (O-LoRA), a method for efficient and privacy-preserving continual learning in LLMs. It ensures that the knowledge of previous tasks is preserved while learning new ones.
3. **Key Idea**:
   - The method leverages the low-rank adaptation (LoRA) technique to store gradients in orthogonal subspaces for each task, minimizing interference and catastrophic forgetting.
4. **Method**:
   - O-LoRA fixes LoRA parameters for past tasks and enforces orthogonality between gradient subspaces of new and past tasks.
   - Orthogonality is achieved through a constraint in the training objective that penalizes overlapping gradient directions.
5. **Contributions**:
   - Proposes a rehearsal-free and parameter-efficient method for continual learning.
   - Demonstrates the generalization capability of O-LoRA to unseen tasks without requiring task-specific inference.
6. **Differences/Innovation**:
   - Unlike methods that require storing past task data (rehearsal-based) or expanding model architecture, O-LoRA works within fixed model capacities and avoids data storage, ensuring data privacy.
   - Focuses on LoRA subspaces rather than storing raw or gradient data from previous tasks.
7. **Results**:
   - Outperforms existing state-of-the-art continual learning methods, such as LFPT5 and A-GEM, on standard benchmarks (e.g., text classification tasks).
   - Achieves superior generalization ability on unseen tasks compared to previous methods, verified through the MMLU zero-shot benchmark.
8. **Conclusion**:
   - O-LoRA is a novel and efficient approach for continual learning in LLMs, addressing catastrophic forgetting while maintaining generalization to unseen tasks.
9. **Discussion**:
   - The paper discusses limitations, including the need for task identification during training and scalability to hundreds of tasks.
   - Suggestions for future research include task-agnostic training and optimization for broader task sets.

#### **Comparison with Other Methods**:

| **Method**        | **Features**                                                 |
| ----------------- | ------------------------------------------------------------ |
| EWC               | Regularizes important weights; less effective with long task sequences. |
| A-GEM             | Uses memory buffers for gradient updates; incurs high storage costs. |
| LFPT5             | Prompt-tuning approach; requires experience replay for good performance. |
| O-LoRA (Proposed) | Avoids data storage, relies on LoRA subspaces for task-specific adaptations, and achieves strong results. |

------

### Base Model and Dataset Summary

| **Aspect**             | **Details**                                                  |
| ---------------------- | ------------------------------------------------------------ |
| **Base Model**         | - Encoder-decoder: T5 (e.g., T5-Base, T5-Large, T5-XL).      |
|                        | - Decoder-only: LLaMA-7B.                                    |
| **Datasets (Tasks)**   | - **Standard CL Benchmark**: AG News, Amazon reviews, Yelp reviews, DBpedia, Yahoo Answers. |
|                        | - **Extended Benchmark**: Includes tasks from GLUE (MNLI, QQP, RTE, SST-2) and SuperGLUE (WiC, CB, COPA, MultiRC, BoolQA) along with the IMDB dataset. |
| **Evaluation Metrics** | - Average Accuracy (AA): Mean accuracy across all tasks after the final task. |
| **Training Setup**     | - O-LoRA applied to attention weights (queries and values) in transformer blocks. |
|                        | - Rank r≤16r \leq 16 for LoRA, achieving efficient representation. |

Let me know if you need further analysis or tailored insights!



## AI QA --- Relation with QGD

### Relationship and Connection Between O-LoRA and OGD

O-LoRA (Orthogonal Low-Rank Adaptation) builds upon the foundational ideas of OGD (Orthogonal Gradient Descent) by adopting and extending the concept of **orthogonalizing gradient updates** to mitigate catastrophic forgetting in continual learning. Both methods share a fundamental approach but are tailored to different contexts and scales. Below is a detailed comparison of their connections, similarities, and differences:

------

### **Shared Characteristics (Common Ground)**

1. **Core Concept**:
   - Both O-LoRA and OGD utilize the **orthogonality constraint** to prevent interference between tasks. They ensure that updates for new tasks do not compromise knowledge of previously learned tasks by constraining updates to directions orthogonal to the gradient subspace of past tasks.
2. **Addressing Catastrophic Forgetting**:
   - Both methods aim to mitigate catastrophic forgetting in neural networks, a problem where learning new tasks degrades performance on older tasks.
3. **Rehearsal-Free Approaches**:
   - Both avoid reliance on replaying raw data from past tasks, addressing privacy concerns and storage limitations.
4. **Parameter Efficiency**:
   - By leveraging the existing structure of the model (e.g., parameter subspaces or gradient directions), both methods minimize memory and computational overhead compared to methods that store past data or gradients explicitly.
5. **Applicability to Continual Learning**:
   - Both are designed for sequential task learning and operate under the constraint of limited access to past task data during training.

------

### **Differences (Key Distinctions)**

| **Aspect**                  | **OGD**                                                      | **O-LoRA**                                                   |
| --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Target Domain**           | General neural networks, especially small to medium-scale continual learning benchmarks (e.g., MNIST variants). | Large Language Models (LLMs), such as T5 and LLaMA, with billions of parameters. |
| **Gradient Representation** | Stores gradient directions from past tasks directly and computes orthogonal updates. | Approximates gradient directions of past tasks using LoRA parameters (low-rank subspaces) instead of storing gradients explicitly. |
| **Computational Cost**      | Scales with the number of stored gradients, limiting feasibility for large-scale models. | Efficient for large-scale models due to LoRA’s low-rank parameterization. |
| **Storage Requirements**    | Requires explicit storage of gradients (e.g., gradients for 200 data points per task in experiments). | Requires only LoRA parameters (low-rank matrices), which are compact and scalable. |
| **Implementation**          | Directly modifies the gradient vectors during training using orthogonality constraints. | Operates on specific layers (e.g., attention weights in transformers) and applies orthogonality constraints on low-rank adaptations (LoRA). |
| **Task Representation**     | Treats tasks as gradient subspaces with no explicit connection to task-specific features. | Leverages task-specific LoRA parameters for better task modularity and compatibility with instruction tuning. |
| **Scalability**             | Limited to smaller-scale neural networks due to gradient storage and computational overhead. | Designed to scale efficiently to large models and long sequences of tasks, leveraging LoRA's compact parameterization. |
| **Generalization**          | Focused on preventing forgetting of prior tasks; generalization to unseen tasks is not explicitly addressed. | Explicitly aims to maintain generalization to unseen tasks via instruction tuning and careful parameter isolation. |
| **Primary Innovations**     | Orthogonalizes task gradients in raw parameter space.        | Orthogonalizes task-specific low-rank subspaces (LoRA), enabling efficient continual learning in transformers. |

------

### **How O-LoRA Extends OGD**

- **Scalability**: O-LoRA addresses the limitations of OGD in large-scale models by using low-rank approximations (LoRA) to capture task subspaces instead of storing high-dimensional gradient vectors.
- **Focus on LLMs**: O-LoRA is specifically tailored for transformer-based language models, integrating seamlessly with their architecture (e.g., by applying LoRA to attention layers).
- **Instruction Tuning**: O-LoRA incorporates instruction tuning for better generalization to unseen tasks, which is not addressed in OGD.
- **Resource Efficiency**: By leveraging the inherent low-rank structure of LLM adaptations, O-LoRA significantly reduces memory and computational demands, making it feasible for models with billions of parameters.

------

### **Summary**

- **OGD** provided a foundation for mitigating catastrophic forgetting through gradient orthogonalization but was limited in scalability and applicability to large-scale models.
- **O-LoRA** adapts and extends this idea to large language models, using LoRA for efficient gradient subspace representation, making it practical for modern, large-scale continual learning tasks.
- Both methods share a core philosophy but differ significantly in their implementation, scalability, and the types of models and tasks they target.



## AI QA --- Clarity the equation

In Section 3.2 of the paper, the authors introduce **Orthogonal Low-Rank Adaptation (O-LoRA)** for continual learning. This method ensures that the learning of new tasks minimally interferes with previously learned tasks by enforcing **orthogonality constraints** between the gradient subspaces of new and past tasks. Below is a detailed explanation of the key equations, their symbols, and their designed purpose.

---

### **Key Equations and Their Symbols**

1. **Gradient Subspace Representation**:
   $$
   A_t = [a_1^t, a_2^t, \ldots, a_r^t]
   $$
   - **$A_t$**: The LoRA low-rank adaptation matrix for the current task $t$, where $A_t \in \mathbb{R}^{d \times r}$.
   - **$r$**: The rank of the LoRA subspace, where $r \ll \min(d, k)$.
   - **$d$**: The dimensionality of the input space for the weight matrix.
   - **$a_i^t$**: The column vectors of $A_t$, representing basis vectors of the subspace.

   **Purpose**: $A_t$ spans the low-rank subspace $U_t$ that approximates the gradient space of task $t$.

2. **Subspace Definition**:
   $$
   U_t = \text{span}\{a_1^t, a_2^t, \ldots, a_r^t\}
   $$
   - **$U_t$**: The subspace spanned by the columns of $A_t$, representing the gradient directions for task $t$.

   **Purpose**: Defines the gradient subspace associated with task $t$, which will be orthogonalized with respect to subspaces of previous tasks.

3. **Orthogonality Constraint**:
   $$
   O_{i,t} = A_i^\top A_t = 0
   $$
   - **$O_{i,t}$**: Measures the overlap between the subspaces $U_i$ (for task $i$) and $U_t$ (for task $t$).
   - **$A_i$**: The LoRA parameter matrix for a previous task $i$.
   - **$A_t$**: The LoRA parameter matrix for the current task $t$.

   **Purpose**: Ensures that the gradient subspaces of task $t$ and task $i$ are orthogonal, thereby preventing interference between tasks.

4. **Training Objective with Orthogonality Loss**:
   $$
   \sum_{x, y \in D_t} \log p_\Theta(y | x) + \lambda_1 \sum_{i=1}^{t-1} L_{\text{orth}}(A_i, A_t)
   $$
   - **$\log p_\Theta(y | x)$**: The primary task loss for task $t$, where $\Theta$ denotes the model parameters.
   - **$L_{\text{orth}}(A_i, A_t)$**: Orthogonality loss that penalizes overlaps between subspaces $U_i$ and $U_t$.
   - **$\lambda_1$**: A hyperparameter that controls the weight of the orthogonality loss.
   - **$x, y \in D_t$**: Training samples from the dataset $D_t$ for task $t$.

   **Purpose**: Balances the objectives of learning the current task and minimizing interference with previous tasks.

5. **Orthogonality Loss Definition**:
   $$
   L_{\text{orth}}(A_i, A_t) = \sum_{j,k} \| O_{i,t}[j, k] \|^2
   $$
   - **$O_{i,t}[j, k]$**: The $j$-th row and $k$-th column element of the overlap matrix $O_{i,t}$.

   **Purpose**: Measures the degree of overlap between subspaces $U_i$ and $U_t$. Squaring ensures that any deviation from orthogonality contributes to the loss.

---

### **Designed Purpose of the Equations**

1. **Preserve Knowledge of Previous Tasks**:
   - By enforcing orthogonality ($O_{i,t} = 0$), the method ensures that updates for task $t$ do not disrupt the knowledge encapsulated in the subspaces of previous tasks.

2. **Efficient Learning of New Tasks**:
   - The use of low-rank LoRA matrices ($A_t$) ensures that the model can efficiently represent task-specific adaptations without significantly increasing computational costs.

3. **Avoid Interference Across Tasks**:
   - Orthogonality constraints minimize interference between gradient directions of different tasks, thus reducing catastrophic forgetting.

4. **Scalability**:
   - By representing gradients in compact low-rank subspaces, O-LoRA ensures that the method scales well to large language models with billions of parameters.

5. **Parameter Isolation**:
   - The separation of task-specific parameters into orthogonal subspaces facilitates modular learning, enabling the model to maintain generalization across tasks.



# 论文介绍 - (N-LoRA

**Is Parameter Collision Hindering Continual Learning in LLMs?**

[`semanticscholar`](https://www.semanticscholar.org/paper/f852e60dd32dc1f3f1f53ba8f76862f77c5cd8d2)  [`Paper`](https://www.semanticscholar.org/paper/f852e60dd32dc1f3f1f53ba8f76862f77c5cd8d2)    ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Ff852e60dd32dc1f3f1f53ba8f76862f77c5cd8d2%3Ffields%3DcitationCount)

 

## AI QA 1 Summary of the Paper: **Is Parameter Collision Hindering Continual Learning in LLMs?**

![image-20250118195146171](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250118195146313.png)





#### **Analysis of Main Content**

1. **Scope/Setting**:

   - The paper focuses on addressing **catastrophic forgetting** in large language models (LLMs) during continual learning (CL). It highlights the limitations of parameter collision in CL and proposes a new method, Non-collision Low-Rank Adaptation (N-LoRA).

2. **Purpose**:

   - To explore the role of parameter collision in degrading performance in CL and propose a method that minimizes collision to enhance task orthogonality and mitigate forgetting.

3. **Key Idea**:

   - While **orthogonality** of parameters helps in decoupling task interdependence, **non-collision parameters** are a more critical factor for knowledge retention and better task separation. Non-collision inherently ensures orthogonality and minimizes interference.

4. **Method**:

   - **N-LoRA**: Introduces sparsity constraints on LoRA parameters using $\ell_1$ regularization to reduce parameter collision rates.

   - Formulates the **collision rate** and its relationship to sparsity, showing that sparsity decreases collisions quadratically.

   - Updates LoRA parameters for new tasks while freezing those of previous tasks. The model minimizes: 

     
     $$
     L = L_{\text{task}} + \lambda \|\Delta W_i\|_1
     $$
     

     where $L_{\text{task}}$ is the task loss and $|\Delta W_i|_1$ enforces sparsity.

5. **Contributions**:

   - Reveals that parameter collisions degrade performance in CL more than a lack of orthogonality.
   - Proposes **N-LoRA** as a simple and effective solution, demonstrating better performance, higher orthogonality, and lower collision rates than existing methods.
   - Provides theoretical proof that non-collision is a sufficient but not necessary condition for orthogonality.

6. **Differences and Innovation**:

   - Compared to O-LoRA:
     - N-LoRA focuses on minimizing collisions rather than enforcing orthogonality explicitly.
     - Achieves better orthogonality as a byproduct of reduced collisions.
     - Demonstrates significant improvement in handling longer sequences and larger numbers of tasks.
   - **Novelty**: Introduces the concept of **parameter collisions** as a core metric for evaluating CL performance.

7. **Results**:

   - Outperforms O-LoRA and other state-of-the-art (SOTA) methods on standard CL benchmarks (e.g., T5-large model).
   - Demonstrates a **+2.9% accuracy improvement**, **4.1x better task orthogonality**, and **58.1x lower collision rates** than O-LoRA.

8. **Conclusion**:

   - Reducing parameter collision via sparsity constraints is critical for improving continual learning performance. N-LoRA provides a lightweight, plug-and-play solution for improving LLMs in CL scenarios.

9. **Discussion**:

   - **Limitations**: Challenges remain in scenarios with a significantly larger number of tasks, where parameter space saturation can still occur.
   - Proposes future exploration into task-agnostic training and dynamic parameter allocation for scalability.

------

### **Comparison with Other Methods**

| **Aspect**                | **O-LoRA**                                          | **N-LoRA (Proposed)**                                        |
| ------------------------- | --------------------------------------------------- | ------------------------------------------------------------ |
| **Focus**                 | Enforces orthogonality between tasks.               | Minimizes parameter collisions, inherently achieving orthogonality. |
| **Parameter Interaction** | Collisions still occur within orthogonal subspaces. | Minimizes collision using sparsity constraints.              |
| **Key Metric**            | Orthogonality (via overlap constraints).            | Collision Rate (explicitly addressed via $\ell_1$ sparsity regularization). |
| **Efficiency**            | Requires orthogonality loss computation.            | Lightweight due to sparsity-induced collision minimization.  |
| **Performance**           | State-of-the-art on some benchmarks.                | Outperforms O-LoRA across all benchmarks, with higher accuracy and better retention. |

------

### **Base Model and Dataset Summary**

| **Aspect**           | **Details**                                                  |
| -------------------- | ------------------------------------------------------------ |
| **Base Model**       | - **T5-Large**: Pre-trained transformer model with LoRA for task-specific adaptations. |
|                      | - **LLaMA-7B**: Larger language model tested for scalability. |
| **Datasets**         | - **Standard CL Benchmark**: AG News, Amazon reviews, Yelp reviews, DBpedia, Yahoo Answers. |
|                      | - **Extended CL Benchmark**: Includes datasets from GLUE (MNLI, QQP, RTE, SST-2) and SuperGLUE (WiC, CB, COPA, MultiRC, BoolQA) along with the IMDB dataset. |
| **Metrics**          | - **Accuracy (AA)**: Average accuracy after the final task.  |
|                      | - **Collision Rate (CR)**: Measures overlap in task-specific parameters. |
|                      | - **Orthogonality Metrics (OO, AWOM)**: Evaluate task separation quality. |
| **Training Details** | - **Sparsity Regularization**: $\ell_1$ regularization with task-specific $\lambda$ values. |
|                      | - **Sparsity Constraint**: Applied to LoRA matrices for reducing parameter interactions. |

Let me know if further clarification or additional insights are needed!    



## AI QA2  ---The relatio in these these papers

### **Comparative Analysis and Relationship Between the Three Papers**

The three papers—**Orthogonal Gradient Descent (OGD)**, **Orthogonal Low-Rank Adaptation (O-LoRA)**, and **Non-Collision Low-Rank Adaptation (N-LoRA)**—focus on mitigating catastrophic forgetting in continual learning (CL) but tackle the problem from progressively refined perspectives. Below is a detailed explanation of their relationships, shared goals, and key differences.

------

### **1. Shared Foundations**

All three papers share the following commonalities:

- Focus on Continual Learning

  :

  - Address catastrophic forgetting in sequential learning tasks where new task training interferes with past knowledge.

- Parameter Efficient Approaches

  :

  - Utilize methods that optimize parameter usage to adapt to new tasks without storing past data, thus preserving privacy and efficiency.

- Gradient/Subspace Management

  :

  - Introduce mechanisms to prevent interference between tasks by managing the parameter space effectively.

------

### **2. Key Evolutionary Differences**

| **Aspect**                   | **OGD**                                                      | **O-LoRA**                                                   | **N-LoRA**                                                   |
| ---------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Core Focus**               | Orthogonal gradient updates to avoid task interference.      | Orthogonal subspace learning using low-rank parameterization. | Minimizing parameter collisions to enhance task separation.  |
| **Parameter Representation** | Operates directly on the raw gradient space of neural networks. | Uses LoRA to represent task-specific subspaces orthogonally. | Introduces sparsity in LoRA to reduce collision and overlap. |
| **Orthogonality Mechanism**  | Projects gradients of new tasks orthogonal to those of past tasks. | Enforces orthogonality between LoRA parameter subspaces via loss constraints. | Achieves orthogonality as a result of sparse parameter allocation, rather than explicit constraints. |
| **Key Limitation Addressed** | Gradient interference between tasks in small to medium models. | Orthogonality enforcement but residual task interaction due to parameter overlap. | Resolves parameter collisions inherent in O-LoRA, improving task isolation and scalability. |
| **Scalability**              | Effective for small to medium models (e.g., MNIST).          | Scales to large language models (LLMs) using LoRA.           | Further improves scalability by reducing collision rates and parameter overlap. |
| **Efficiency**               | Computationally expensive due to gradient projections.       | Computationally efficient but incurs parameter overlap.      | Lightweight due to sparsity-induced task separation.         |
| **Task Orthogonality**       | Ensures strict gradient orthogonality.                       | Enforces subspace orthogonality, but collisions still occur. | Naturally achieves orthogonality by avoiding collisions entirely. |
| **Innovation**               | Introduced orthogonal gradient updates.                      | Extended orthogonality to LLM subspaces with LoRA.           | Redefined task separation by focusing on collision reduction. |

------

### **3. Relationships and Key Advances**

1. **OGD → O-LoRA**:

   - Relationship

     :

     - O-LoRA extends OGD's core idea of orthogonality to a scalable parameter-efficient framework suitable for LLMs by introducing LoRA. Instead of working on raw gradient spaces, it uses low-rank parameter subspaces for efficiency and scalability.

   - Advances

     :

     - Scales to large-scale tasks and models like T5 and LLaMA.
     - Incorporates instruction tuning for better generalization to unseen tasks.
     - Reduces memory overhead compared to OGD's explicit gradient storage.

2. **O-LoRA → N-LoRA**:

   - Relationship

     :

     - N-LoRA builds on O-LoRA by addressing its limitation of **parameter collision**—overlapping task-specific parameters even within orthogonal subspaces.

   - Advances

     :

     - Introduces sparsity in task-specific parameters to minimize collision rates, inherently achieving better task orthogonality.
     - Demonstrates that collision reduction is a **sufficient but not necessary condition** for achieving orthogonality.
     - Improves performance on both standard benchmarks and long-sequence tasks compared to O-LoRA.

3. **OGD → N-LoRA**:

   - Relationship

     :

     - N-LoRA retains the foundational idea of orthogonality but evolves it into a **collision-driven perspective**, emphasizing sparsity over explicit gradient or subspace constraints.

   - Advances

     :

     - Removes the need for direct orthogonality enforcement, relying instead on sparsity to naturally separate tasks.
     - Simplifies computation while delivering superior scalability and performance.

------

### **4. Conceptual Shifts Across the Three Papers**

| **Concept**                        | **OGD**                           | **O-LoRA**                                      | **N-LoRA**                                       |
| ---------------------------------- | --------------------------------- | ----------------------------------------------- | ------------------------------------------------ |
| **Orthogonality**                  | Enforced via gradient projection. | Imposed via subspace constraints.               | Achieved indirectly through sparsity.            |
| **Focus on Parameter Interaction** | Not explicitly addressed.         | Partially mitigated through subspace isolation. | Completely avoids interaction via non-collision. |
| **Primary Metric**                 | Gradient orthogonality.           | Subspace orthogonality.                         | Collision rate and sparsity.                     |

------

### **5. Practical Implications**

- Scalability

  :

  - OGD is limited to small models, while O-LoRA and N-LoRA scale effectively to LLMs like T5 and LLaMA.

- Parameter Management

  :

  - OGD focuses on raw gradients, O-LoRA optimizes subspaces, and N-LoRA minimizes collisions through sparsity.

- Performance

  :

  - N-LoRA consistently outperforms the other two in accuracy, task orthogonality, and generalization.

------

### **Conclusion**

The evolution from **OGD** to **O-LoRA** and then to **N-LoRA** represents a progression from basic orthogonality enforcement to a refined understanding of parameter interactions in CL. Each method builds upon its predecessor:

1. **OGD** introduced orthogonality-based continual learning.
2. **O-LoRA** adapted this idea to LLMs with low-rank subspace parameterization.
3. **N-LoRA** advanced the field by targeting parameter collisions, achieving higher efficiency and better scalability.

This progression highlights a deeper understanding of task interference, culminating in N-LoRA as the most effective and scalable solution for LLM-based continual learning.



## AI QA 3 -- Collision

### **Understanding the Idea of Collision in N-LoRA**

The concept of **collision** in the N-LoRA paper refers to **parameter collisions**, which occur when **task-specific parameters** (e.g., LoRA parameters) overlap or interact in the same regions of the parameter space. This interaction can degrade performance by causing interference between tasks, leading to **catastrophic forgetting**.

---

### **What Does Collision Mean in N-LoRA?**
- **Definition**:
  - Collision happens when two task-specific parameter matrices, $\Delta W_1$ and $\Delta W_2$, share **non-zero values at the same positions**. Mathematically:
    $$
    \text{Collision at position } (a, b): \Delta W_1[a, b] \neq 0 \ \text{and} \ \Delta W_2[a, b] \neq 0
    $$
  - A high collision rate indicates significant overlap between parameters used for different tasks, leading to interference and forgetting.

- **Contrast with Orthogonality**:
  - Orthogonality ensures that parameter updates for different tasks do not align in the same direction. However, even if tasks are orthogonal in a subspace, **parameter collisions can still occur within overlapping regions** of the parameter matrix.

- **Relation to Embedding Space**:
  - Collision is not explicitly tied to feature overlap in the **embedding space** but rather reflects how **parameter spaces interact** in the learned representation. It indirectly affects feature embeddings because parameter collisions can cause conflicting updates, resulting in suboptimal feature separation or reuse across tasks.

---

### **Why Does Collision Matter?**
1. **Task Interference**:
   - When task-specific parameters collide, gradients from one task can overwrite or interfere with those of another task, leading to poor task performance or catastrophic forgetting.
   
2. **Limits of Orthogonality**:
   - While orthogonality reduces task overlap at the subspace level, it does not prevent two tasks from **interacting at the same parameter positions**, causing local interference.

3. **Knowledge Retention**:
   - Minimizing collision ensures that parameters for each task are used independently, preserving knowledge of previously learned tasks while accommodating new tasks.

---

### **How N-LoRA Addresses Collisions**
N-LoRA introduces **sparsity constraints** to minimize collisions:
- **Sparsity-Driven Approach**:
  - By enforcing $\ell_1$ regularization on LoRA parameters, N-LoRA encourages **most parameter values to be zero**, creating sparse matrices for each task. Sparse parameters naturally reduce collisions because the probability of overlapping non-zero entries decreases quadratically with sparsity:
    $$
    \text{Collision Rate} \propto s_1 \times s_2
    $$
    where $s_1$ and $s_2$ are the sparsity rates of two tasks.

- **Decoupled Parameter Subspaces**:
  - Sparsity ensures that task-specific parameters occupy **non-overlapping regions** of the parameter space, minimizing interference.

- **Orthogonality as a Byproduct**:
  - With reduced collisions, tasks naturally become more orthogonal because their parameter usage does not overlap.

---

### **Does Collision Mean Features Do Not Overlap in the Embedding Space?**
- **Not Directly**:
  - Collision is primarily about **parameter-level overlap**, not directly about feature overlap in the embedding space.
  - However, parameter collisions can lead to **feature interference**, where task-specific features are not well-separated due to conflicting updates during training.

- **Indirect Effect on Embedding Space**:
  - Reducing collisions can improve the separation of features in the embedding space by ensuring independent parameter updates for each task. This improves task-specific feature representations and reduces cross-task interference.

---

### **Visualizing Collision in Parameter Space**
Figure 2 in the paper illustrates collision:
1. **O-LoRA**:
   - Despite orthogonality constraints, significant regions of the parameter matrices overlap (red regions in the visualization), leading to task interference.
2. **N-LoRA**:
   - By introducing sparsity, N-LoRA minimizes overlap, reducing red regions significantly and resulting in improved task separation and knowledge retention.

---

### **Conclusion**
In N-LoRA, **collision** refers to the overlap of non-zero task-specific parameters within the same positions of the parameter matrix, not directly to feature overlap in the embedding space. However, minimizing parameter collisions indirectly improves feature separation by ensuring better task-specific parameter isolation, leading to more robust continual learning.



## AI  QA 4 metric in thsi paper

### **Comparison of Metrics Across the Papers**

Each paper introduces different metrics to evaluate the effectiveness of their continual learning (CL) approach. Below is a detailed explanation of these metrics:

---

### **1. Metrics in OGD**
OGD focuses on **gradient orthogonality** to mitigate catastrophic forgetting. The metrics used are:

#### **Gradient Orthogonality**
- Measures the alignment between gradient updates for different tasks.
- Mathematically, the angle between gradients $g_i$ and $g_j$ for tasks $i$ and $j$ should ideally satisfy:
  $$
  g_i^\top g_j = 0
  $$
- **Purpose**:
  - Ensures that gradient updates for new tasks do not interfere with the parameter space of previously learned tasks.

#### **Accuracy Across Tasks**
- Evaluates the average accuracy on all tasks after training on the final task:
  $$
  \text{Accuracy} = \frac{1}{T} \sum_{i=1}^T a_{i, T}
  $$
  where $a_{i, T}$ is the accuracy on task $i$ after training on task $T$.

---

### **2. Metrics in O-LoRA**
O-LoRA evaluates **orthogonality** at the subspace level using LoRA parameters, focusing on **parameter efficiency** and **knowledge retention**. The key metrics include:

#### **Orthogonal Overlap (OO)**
- Measures the overlap between LoRA subspaces for tasks $i$ and $j$:
  $$
  \text{OO} = \sum_{i=1}^{t-1} \|A_i^\top A_t\|^2
  $$
  where $A_i$ is the LoRA parameter matrix for task $i$.
- **Purpose**:
  - Quantifies the degree of interference between subspaces; lower values indicate better orthogonality.

#### **Average Accuracy (AA)**
- Same as in OGD, measures the average accuracy across all tasks after training on the final task:
  $$
  \text{AA} = \frac{1}{T} \sum_{i=1}^T a_{i, T}
  $$

#### **Forgetting Rate (F.Ra)**
- Quantifies how much knowledge is forgotten across tasks:
  $$
  F_R = \frac{1}{T-1} \sum_{t=1}^{T-1} \left( \max_{k \geq t} a_{k, t} - a_{T, t} \right)
  $$
- **Purpose**:
  - Captures performance drops on earlier tasks due to new task training.

---

### **3. Metrics in N-LoRA**
N-LoRA expands on O-LoRA by introducing **collision-related metrics** and emphasizes **sparsity** as a key factor for minimizing interference. The metrics include:

#### **Average Collision Rate (ACR)**
- Measures the fraction of overlapping non-zero elements between task-specific parameter matrices:
  $$
  \text{ACR} = \frac{2}{T(T-1)} \sum_{1 \leq i < j \leq T} \frac{\sum_{a,b} 1(\Delta W_i[a, b] \neq 0 \wedge \Delta W_j[a, b] \neq 0)}{n \times m}
  $$
  - $\Delta W_i$ and $\Delta W_j$ are parameter matrices for tasks $i$ and $j$.
- **Purpose**:
  - Quantifies parameter collisions; lower values indicate better separation.

#### **Adapter Weight Orthogonality Magnitude (AWOM)**
- Measures the orthogonality between task-specific parameter matrices:
  $$
  \text{AWOM} = \sum_{i=1}^{T-1} \|\Delta W_T^\top \Delta W_i\|^2
  $$
- **Purpose**:
  - Directly evaluates parameter interference; smaller values indicate better orthogonality.

#### **Generalized Sparsity Rate (GSR)**
- Evaluates the sparsity of task-specific parameter matrices:
  $$
  \text{GSR} = \frac{\sqrt{mn} \cdot \|\Delta W\|_1}{\|\Delta W\|_2}
  $$
  where $\|\Delta W\|_1$ is the $\ell_1$ norm, and $\|\Delta W\|_2$ is the Frobenius norm.
- **Purpose**:
  - Higher sparsity leads to fewer parameter collisions, improving task separation.

#### **Nuclear Norm**
- Measures the effective dimensionality of task subspaces:
  - Calculated as the sum of singular values of the LoRA parameter matrix $\Delta W$.
- **Purpose**:
  - Indicates the subspace size used by each task; lower norms suggest efficient use of subspaces.

#### **Average Accuracy (AA)** and **Forgetting Rate (F.Ra)**
- Retains the same definitions as in O-LoRA.

---

### **Summary of Metric Evolution**

| **Metric**                              | **OGD** | **O-LoRA** | **N-LoRA**                           |
| --------------------------------------- | ------- | ---------- | ------------------------------------ |
| **Gradient Orthogonality**              | ✅       | ✖          | ✖                                    |
| **Orthogonal Overlap (OO)**             | ✖       | ✅          | ✅ (improved by reducing collisions). |
| **Accuracy (AA)**                       | ✅       | ✅          | ✅                                    |
| **Forgetting Rate (F.Ra)**              | ✅       | ✅          | ✅                                    |
| **Collision Rate (ACR)**                | ✖       | ✖          | ✅                                    |
| **Sparsity (GSR)**                      | ✖       | ✖          | ✅                                    |
| **Adapter Weight Orthogonality (AWOM)** | ✖       | ✖          | ✅                                    |
| **Nuclear Norm**                        | ✖       | ✖          | ✅                                    |

---

### **Conclusion**
The metrics evolve from focusing on **gradient orthogonality** in OGD to **subspace orthogonality** in O-LoRA, and finally to **collision minimization and sparsity** in N-LoRA. Each paper adds new metrics to address specific challenges in continual learning, leading to progressively refined evaluations of task interference and parameter efficiency.





# **Continual Task Settings in the  Papers**



## 1 Task , dataset , other methods

### O-LoRA 

#### task and dataset

![image-20250118202405373](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250118202405567.png)

#### task sequence



![image-20250118202429975](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250118202430156.png)

#### prompt for different task



![image-20250118202450097](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250118202450306.png)



#### compare with different methods

![image-20250118203058858](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250118203059051.png)



### **Baseline Methods in O-LoRA**

The O-LoRA paper compares its performance against 10 baseline methods, which represent a mix of continual learning approaches. Below is a summary of these methods:

| **Method**     | **Type**                | **Description**                                              | **Key Characteristics**                                      |
| -------------- | ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **SeqFT**      | Non-Continual Learning  | Fine-tunes all model parameters on a sequence of tasks without any replay or regularization. | Serves as a lower bound; prone to catastrophic forgetting.   |
| **SeqLoRA**    | Non-Continual Learning  | Fine-tunes a single LoRA adapter across all tasks, freezing the pre-trained model. | Sequential training without regularization or replay, leading to high forgetting. |
| **IncLoRA**    | Non-Continual Learning  | Trains a new LoRA module for each task, keeping previous LoRA parameters fixed. | Task-specific LoRA parameters; no mechanisms to prevent interference between modules. |
| **Replay**     | Continual Learning      | Fine-tunes all model parameters while replaying samples from previous tasks using a memory buffer. | Memory buffer mitigates forgetting but requires access to prior task data. |
| **EWC**        | Regularization-Based CL | Applies a regularization loss to prevent large changes to important parameters identified via Fisher Information Matrix. | Protects key parameters to reduce forgetting but struggles with longer task sequences. |
| **LwF**        | Regularization-Based CL | Preserves knowledge of previous tasks by regularizing outputs (or logits) on new task data to match the model's responses on prior tasks. | Avoids storing data but may underperform when task distributions are highly dissimilar. |
| **L2P**        | Prompt-Based CL         | Dynamically selects and updates prompts from a prompt pool in an instance-wise manner for each task. | Efficient and flexible for LLMs; relies on prompt selection to adapt to new tasks. |
| **LFPT5**      | Prompt-Based CL         | A unified prompt-tuning framework for continual learning that uses soft prompts to simultaneously solve tasks and generate replay samples. | Combines prompt-tuning with experience replay for better task retention and adaptation. |
| **ProgPrompt** | Architecture-Based CL   | Trains task-specific soft prompts sequentially, appending them to previously learned prompts. Task ID is used to select the appropriate prompt during inference. | Creates independent models for each task; lacks generalization to unseen tasks. |
| **PerTaskFT**  | Independent Training    | Trains a separate model for each task, storing and utilizing task-specific parameters independently. | Represents an upper bound for continual learning; computationally expensive and not scalable for a large number of tasks. |
| **MTL**        | Multi-Task Learning     | Trains a single model on all tasks simultaneously, with access to all task data during training. | Serves as the **upper bound** for performance; assumes access to all tasks and data, which is infeasible in most CL scenarios. |

---

### **Additional Notes**
- **Continual Learning Methods**:
  - Replay, EWC, LwF, and L2P are traditional CL methods focusing on memory, regularization, or prompt strategies.
- **Prompt-Based CL**:
  - L2P and LFPT5 leverage the flexibility of LLMs with prompt tuning to improve task adaptation.
- **Upper Bound Comparisons**:
  - PerTaskFT and MTL represent idealized scenarios for CL, with MTL assuming access to all data simultaneously and PerTaskFT storing independent models.

This variety of baselines ensures that O-LoRA is compared against a diverse set of approaches, spanning non-continual, regularization-based, replay-based, and prompt-based methods.



###   N-LoRA

#### task and dataset



![image-20250118202536011](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250118202536195.png)



####  prompt for different task

![image-20250118202551548](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250118202551743.png)



####  task sequence

![image-20250118202624282](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250118202624466.png)



#### compare with different methods

![image-20250118203131940](https://zuti.oss-cn-qingdao.aliyuncs.com/img/20250118203132141.png)



## Question 1 Just a text classifcaion task?

Yes, you are absolutely correct!

In essence, the approach used in the **O-LoRA** paper (and similar continual learning frameworks) treats tasks like **QA (Question Answering)**, **NLI (Natural Language Inference)**, and others as **text classification problems**. The key idea is to transform these tasks into a **unified format** that fits the model’s capabilities, effectively turning them into classification tasks. Here’s why and how this is done:

------

### **Why Are QA and NLI Treated as Classification Tasks?**

1. **Unified Task Framework**:
   - By reformatting diverse tasks into a text classification framework, the model can handle them using a consistent input-output format.
   - This simplifies training and evaluation across multiple task types in a continual learning setting.
2. **Reduced Complexity**:
   - Treating QA or NLI as classification reduces the need for specialized architectures or task-specific outputs. Instead, the model generates answers from a fixed set of options (classes).
3. **Efficiency**:
   - Classification tasks are computationally simpler compared to generating free-form text, making this approach more efficient in terms of memory and computation.
4. **Alignment with Pre-trained Language Models**:
   - Pre-trained language models (e.g., T5, LLaMA) are highly effective at processing text when provided with explicit options. Using predefined classes leverages their ability to "select" the best output based on the prompt.

------

### **How Are QA and NLI Reformulated as Classification Tasks?**

1. **QA as Classification**:
   - For tasks like Boolean Question Answering (BoolQA), the question is rephrased into a binary classification problem:
     - Input: "According to the following passage, is the question true or false?"
     - Options: ["true", "false"]
     - The model selects one of the two classes.
2. **NLI as Classification**:
   - For NLI tasks, the goal is to classify the relationship between two sentences:
     - Input: "What is the logical relationship between 'sentence 1' and 'sentence 2'?"
     - Options: ["entailment", "neutral", "contradiction"]
     - The model determines the correct relationship.
3. **Other Tasks**:
   - Sentiment analysis (SC), topic classification (TC), and other tasks follow a similar pattern:
     - Provide a clear prompt.
     - Define a fixed set of options (classes).
     - Let the model classify the input into one of the options.

------

### **Does This Mean the Model Only Performs Text Classification?**

- **Yes, Functionally**:
  - The model essentially acts as a **text classifier** by selecting one of the predefined options based on the input and task instructions.
- **However, With a Broader Perspective**:
  - The model’s **underlying mechanism** involves encoding rich semantic representations of the input text, which goes beyond simple classification. It leverages its pre-trained capabilities to interpret complex relationships (e.g., in NLI) or validate logical reasoning (e.g., in QA).

------

### **Conclusion**

Yes, the O-LoRA framework essentially **reformats all tasks into a classification problem**, even for inherently more complex tasks like QA and NLI. This approach capitalizes on the strengths of pre-trained language models while simplifying task adaptation and evaluation in continual learning settings.
