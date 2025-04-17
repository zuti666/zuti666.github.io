# Experiment Record



# Experience , Some Useful Tips or tools :

## Available Methods and Code

### **🎯 Evaluation Metrics**

‣ Loss Landscape Sharpness:  

1. Hessian value:  

   Theory Source, How to calculate to get Hessian value:

   1. Block Lanczos  [The Block Lanczos Method for Computing Eigenvalues - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/B9780125872607500182)  . Apply in  Paper

      **I implement it by myself .**

   2. The Lanczos algorithm of Ghorbani  [An Investigation into Neural Net Optimization via Hessian Eigenvalue Density](https://proceedings.mlr.press/v97/ghorbani19b). 

      Apply in Paper [[2010.01412\] Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412) .

      Code implementation:  [GitHub - google/spectral-density: Hessian spectral density estimation in TF and Jax](https://github.com/google/spectral-density).

      **I implement it by myself .**

   3. The Power Iteration algorithm  [Praktische Verfahren der Gleichungsauflösung . - Mises - 1929 - ZAMM](https://onlinelibrary.wiley.com/doi/abs/10.1002/zamm.19290090105), Apply in Paper  [When Do Flat Minima Optimizers Work?](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69b5534586d6c035a96b49c86dbeece8-Abstract-Conference.html) 

2. Median of the dominant Hessian eigenvalue, Apply in Paper  [When Do Flat Minima Optimizers Work?](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69b5534586d6c035a96b49c86dbeece8-Abstract-Conference.html) 

3. Hessian dominate eigenvalue, Apply in Paper  [[2106.01548\] When Vision Transformers Outperform ResNets without Pre-training or Strong Data Augmentations](https://arxiv.org/abs/2106.01548)

4. Based on the Hessian eigenvalue, we can do some statistical analysis and summary

5. **2-D  Loss Landscape**: Visual Loss Landscape in two random direction with different noise [Visualizing the Loss Landscape of Neural Nets](https://proceedings.neurips.cc/paper/2018/hash/a41b3bb3e6b050b6c9067c67f663b915-Abstract.html). Code implementation: [tomgoldstein/loss-landscape: Code for visualizing the loss landscape of neural nets](https://github.com/tomgoldstein/loss-landscape) 

   Apply in the paper: [[2106.01548\] When Vision Transformers Outperform ResNets without Pre-training or Strong Data Augmentations](https://arxiv.org/abs/2106.01548)

   **I implement it by myself .**

   

6. Based on the 2-D Loss Landscape, we can do some statistical analysis and summary

7. **Sharpness**:  [[1609.04836\] On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836)  Available Code [keskarnitish/large-batch-training: Code to reproduce some of the figures in the paper "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"](https://github.com/keskarnitish/large-batch-training)

   **I implement it by myself .**

   

8. **Visual the mineral point relationship between the three models in the loss land**

   describe in the paper   [Visualizing the Loss Landscape of Neural Nets](https://proceedings.neurips.cc/paper/2018/hash/a41b3bb3e6b050b6c9067c67f663b915-Abstract.html).

    Apply in the paper   [SWAD: Domain Generalization by Seeking Flat Minima](https://proceedings.neurips.cc/paper_files/paper/2021/hash/bcb41ccdc4363c6848a1d760f26c28a0-Abstract.html)

   

### SAM 

 Paper [[2010.01412\] Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412) 

Code Implement : 

[davda54/sam: SAM: Sharpness-Aware Minimization (PyTorch)](https://github.com/davda54/sam)

[moskomule/sam.pytorch: A PyTorch implementation of Sharpness-Aware Minimization for Efficiently Improving Generalization](https://github.com/moskomule/sam.pytorch)



### LoRA 

lora peft 0.3.0  :  [peft · PyPI](https://pypi.org/project/peft/0.3.0/#files)

The LoRA method replace the linear Layer into the LoRA



## Hyper parameters 

### Learning  Rate set :

Change from Adamw to  SGD , set a large learning rate. 



## Zero-shot 

When continual  train the Vit ,  we need to replace the final classifier,  so the result are not zero-shot  in Language Model or CLip.



# Q1 : Adding a LoRA changes the entire model's loss landscape flatness?



## **🧪 Experiment 1: Effect of LoRA on Loss Landscape Flatness**

1. **🆔 Experiment ID**: Exp-1

2. **📅 Time**: [Add date]

3. **🎯 Purpose**: To investigate how the integration of LoRA modules affects the flatness of the loss landscape.

4. **Tags**: `📚 NLP` `💡 LoRA` `⚡ AdamW`

5. **📚 Scope**:  **NLP**

6. **🧠 Base Model**: `T5-small`

7. **🗂️ Training Set**:  

8. **🗃️ Dataset**: `DBpedia`  **amazon ** **agnews**   **yahoo**

9. **⚙️ Optimizer**:  `AdamW (HuggingFace)`

10. **🧩 Task Setting**:  Multi-task sequential continual training on a LoRA part

11. **🔧 Experiment Setup**:

   - Train T5-small with LoRA on  `DBpedia`  **amazon ** **agnews**   **yahoo**
   - Visualize loss landscape (`w + ε`) before and after training.

12. **🎯 Evaluation Metrics**:
      ‣ Loss Landscape Sharpness:  

   1. Hessian value:  

      Theory Source, How to calculate to get Hessian value:

      1. **Block Lanczos**  [The Block Lanczos Method for Computing Eigenvalues - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/B9780125872607500182)  . Apply in  Paper

      2. The Lanczos algorithm of Ghorbani  [An Investigation into Neural Net Optimization via Hessian Eigenvalue Density](https://proceedings.mlr.press/v97/ghorbani19b). 

         Apply in Paper [[2010.01412\] Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412) .

         Code implementation:

      3. The Power Iteration algorithm  [Praktische Verfahren der Gleichungsauflösung . - Mises - 1929 - ZAMM](https://onlinelibrary.wiley.com/doi/abs/10.1002/zamm.19290090105), Apply in Paper  [When Do Flat Minima Optimizers Work?](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69b5534586d6c035a96b49c86dbeece8-Abstract-Conference.html) 

   2.   Median of the dominant Hessian eigenvalue, Apply in Paper  [When Do Flat Minima Optimizers Work?](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69b5534586d6c035a96b49c86dbeece8-Abstract-Conference.html) 

   3. Based on the Hessian eigenvalue, we can do some statistical analysis and summary

   4. 2-D  Loss Landscape: Visual Loss Landscape in two random direction with different noise [Visualizing the Loss Landscape of Neural Nets](https://proceedings.neurips.cc/paper/2018/hash/a41b3bb3e6b050b6c9067c67f663b915-Abstract.html). Code implementation:

   

13. **🧱 Base Code**: 

   - N lora : [PKU-YuanGroup/N-LoRA: 【COLING 2025🔥】Code for the paper "Is Parameter Collision Hindering Continual Learning in LLMs?".](https://github.com/PKU-YuanGroup/N-LoRA)
   - lora peft 0.3.0  :  [peft · PyPI](https://pypi.org/project/peft/0.3.0/#files)

14. **📖 Reference Papers/Methods**: 

   - LoRA:  [[2106.09685\] LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
   - NLoRA :  [[2410.10179\] Is Parameter Collision Hindering Continual Learning in LLMs?](https://arxiv.org/abs/2410.10179)

15. **📊 Results**:

   1. Loss landscape became sharper after training.

   2. Loss landscape also rely on Dataset, Yahoo performance  obviously drop

   3. The hessian value has some **outlier eigenvalues**.(which is aligned with [An Investigation into Neural Net Optimization via Hessian Eigenvalue Density](https://proceedings.mlr.press/v97/ghorbani19b.html))

      

16. 📊 **Results Visualization**:

    

17. **🧾 Conclusion**:

   - Sharpness increased around minima compare with the origin Model.
   - Dataset characteristics influence flatness (e.g., performance dropped on Yahoo due to complexity).
   - The hessian value are rely on dataset, because the Lanczos algorithm calculate the HV.

------



## **🧪 Experiment 1.2: Effect of LoRA on Loss Landscape Flatness**

1. **🆔 Experiment ID**: Exp-1.2

2. **📅 Time**: [Add date]

3. **🎯 Purpose**: To investigate how the integration of LoRA modules affects the flatness of the loss landscape in a single dataset.

4. **Tags**: `📚 NLP` `💡 LoRA` `⚡ AdamW`

5. **📚 Scope**:  **NLP**

6. **🧠 Base Model**: `T5-small`

7. **🗂️ Training Set**:    Single-task training

8. **🗃️ Dataset**: `DBpedia`  

9. **⚙️ Optimizer**:  `AdamW (HuggingFace)`

10. **🧩 Task Setting**:  Single-task training on a LoRA part

11. **🔧 Experiment Setup**:

    - Train T5-small with LoRA on  `DBpedia`  
    - Visualize loss landscape (`w + ε`) before and after training.

12. **🎯 Evaluation Metrics**:
    ‣ Loss Landscape Sharpness:  

    1. 2-D  Loss Landscape: Visual Loss Landscape in two random direction with different noise [Visualizing the Loss Landscape of Neural Nets](https://proceedings.neurips.cc/paper/2018/hash/a41b3bb3e6b050b6c9067c67f663b915-Abstract.html). Code implementation:
    2. Based on the Hessian eigenvalue, we can do some statistical analysis and summary

    

13. **🧱 Base Code**: 

    - N lora : [PKU-YuanGroup/N-LoRA: 【COLING 2025🔥】Code for the paper "Is Parameter Collision Hindering Continual Learning in LLMs?".](https://github.com/PKU-YuanGroup/N-LoRA)
    - lora peft 0.3.0  :  [peft · PyPI](https://pypi.org/project/peft/0.3.0/#files)

14. **📖 Reference Papers/Methods**: 

    - LoRA:  [[2106.09685\] LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
    - NLoRA :  [[2410.10179\] Is Parameter Collision Hindering Continual Learning in LLMs?](https://arxiv.org/abs/2410.10179)

15. **📊 Results**:

    1. Loss landscape became sharper after training.
    2. **2.2 → 3.2**: The **LoRA** component, after training, improves from an initial state to a well-performing state.
    3. **1 → 2.3 (r=8)→ 3.3 (r=8)**:  After training the **LoRA** component, the model's loss landscape becomes sharper.
    4. **3.2 (r=8) → 3.2 (r=4)**: A smaller rank **LoRA** results in a sharper model. The reduced model size leads to increased sharpness.
    5. **3.3 (r=8) → 3.3 (r=4)**: The overall model becomes flatter. A smaller **LoRA** has less impact on the entire structure.

16. 📊 **Results Visualization**:

    

17. **🧾 Conclusion**:

    - Sharpness increased around minima compare with the origin Model.
    - A smaller **LoRA** has less impact on the entire structure compared with a big rank LoRA.

------

# Q2 Does SAM help Model get a flatter landscape? 



## **🧪 Experiment 2: Comparing SGD and SAM for Improving Loss Flatness**

1. **🆔 Experiment ID**: Exp 1.3 

2. **📅 Time**: 

3. **🎯 Purpose**: To compare whether SAM can improve model flatness and overall performance versus SGD.

4. **📚 Scope**: 🖼️ **CV**

5. **🧠 Base Model**: `ViT-base-patch16-224-in21k`

6. **🗂️ Training Set**: Task 0 of Split CIFAR-100

7. **🗃️ Dataset**: `CIFAR-100 (Split 20)`

8. **⚙️ Optimizer**: 🔁 `SGD` vs 🌊 `SAM`

9. **🧩 Task Setting**:  classification . / Homogeneous tasks (1 dataset split into task)

10. **🔧 Experiment Setup**:

  - Replace original classifier.
  - Train new classifier with SGD and SAM.
  - Check compare test performance.

11. **📖 Reference**: 

    - SAM from paper [[2010.01412\] Sharpness-Aware Minimization for Efficiently Improoving Generalization](https://arxiv.org/abs/2010.01412) 

      Code Implement : 

      [davda54/sam: SAM: Sharpness-Aware Minimization (PyTorch)](https://github.com/davda54/sam)

    - Experiment reference Paper [An Empirical Investigation of the Role of Pre-training in Lifelong Learning](https://www.jmlr.org/papers/v24/22-0496.html)

      Code Implement : [sanketvmehta/lifelong-learning-pretraining-and-sam: Code for the paper "Mehta, S. V., Patil, D., Chandar, S., & Strubell, E. (2023). An Empirical Investigation of the Role of Pre-training in Lifelong Learning. The Journal of Machine Learning Research 24 (2023)"](https://github.com/sanketvmehta/lifelong-learning-pretraining-and-sam)

       

12. **🎯 Evaluation Metrics**: 

    ‣ Model's performance: 

    - accuracy on test dataset

    ‣ Loss Landscape Sharpness:  

    1. 2-D  Loss Landscape: Visual Loss Landscape in two random direction with different noise [Visualizing the Loss Landscape of Neural Nets](https://proceedings.neurips.cc/paper/2018/hash/a41b3bb3e6b050b6c9067c67f663b915-Abstract.html). 

    2. Based on the Hessian eigenvalue, we can do some statistical analysis and summary

    3. **Sharpness**:  [[1609.04836\] On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836)  Available Code [keskarnitish/large-batch-training: Code to reproduce some of the figures in the paper "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"](https://github.com/keskarnitish/large-batch-training)

       **I implement it by myself .**

13. **📊 Results**:

   1. The SAM  shows better performance in accuracy and sharpness.
   2. In the Loss land, the SAM result get a more flat Minimum cone.

   

14. 📊 **Results Visualization**:


15. **🧾 Conclusion**:
   - SAM improves performance and flattens the loss surface under small Homo dataset.



# Q3 Dose SAM help Model get a better performance under CL?

## **🧪 Experiment 3.1: Continual training a LoRA with SAM vs SGD in 4 task**

1. **🆔 Experiment ID**: Exp-3.1

2. **📅 Time**: 

3. **🎯 Purpose**: Continual training a LoRA part on four tasks and compare the performances of SGD and SAM

4. **📚 Scope**: 🖼️ **CV**

5. **🧠 Base Model**: `CLIP (ViT-B/16)`

6. **🗂️ Training Set**:  Multi-domain  

7. **🗃️ Dataset**: Aircraft, Caltech101, CIFAR100, DTD

8. **⚙️ Optimizer**: 🔁 `SGD` vs 🌊 `SAM`

   **⚙️ Hyperparameters**:
   ‣ Learning Rate:  (5.5e-1 3e-2 3e-2 1.2e-1)
   ‣ SAM ρ:  0.05 0.10 0.20

9. **🧩 Task Setting**: Multi-domain Task-Incremental

10. **🔧 Experiment Setup**:

    -  Only one LoRA adapter.
    - When training a new task , re-train the old one .

11. **📖 Reference**: 

12. **🎯 Evaluation Metrics**: 

    ‣ Model's performance: 

    - accuracy on test dataset

13. **📊 Results**: 

    1. SAM with big noise get a better performance both in anti-forgetting and generalization.

14. 📊 **Results Visualization**:

15. **🧾 Conclusion**:  SAM helps.

------

## **🧪 Experiment 3.2: Train a LoRA for each task separately and compare the generalization of SGD and SAM on other tasks**

1. **🆔 Experiment ID**: Exp-3.2

2. **📅 Time**:  4,7 

3. **🎯 Purpose**: Train a LoRA for each task separately and compare the generalization of SGD and SAM on other tasks

4. **📚 Scope**: 🖼️ **CV**

5. **🧠 Base Model**: `CLIP (ViT-B/16)`

6. **🗂️ Training Set**:  Multi-domain  

7. **🗃️ Dataset**: Aircraft, Caltech101, CIFAR100, DTD

8. **⚙️ Optimizer**: 🔁 `SGD` vs 🌊 `SAM`

   **⚙️ Hyperparameters**:
   ‣ Learning Rate:  (5.5e-1 3e-2 3e-2 1.2e-1)
   ‣ SAM ρ:   0.05 0.10 0.20

9. **🧩 Task Setting**: Multi-domain Task-Incremental

10. **🔧 Experiment Setup**:

  - Assign each task a separately LoRA adapter.
  - Train incrementally using InfLoRA design.

11. **📖 Reference**: 

12. **🎯 Evaluation Metrics**: 

    ‣ Model's performance: 

    - accuracy on test dataset

13. **📊 Results**: 

    1. SAM get a better performance  in   generalization on the other task.
    2. SAM may get a Suboptimal results on the training dataset.

14. 📊 **Results Visualization**:

15. **🧾 Conclusion**:  



## **🧪 Experiment 3.3: Multi-LoRA with SAM vs SGD in Multi-domain Learning**

1. **🆔 Experiment ID**: Exp-004

2. **📅 Time**: [Add date]

3. **🎯 Purpose**: To evaluate the use of SAM and SGD in training multi-LoRA models in a multi-domain continual setting.

4. **📚 Scope**: 🖼️ **CV**

5. **🧠 Base Model**: `CLIP (ViT-B/16)`

6. **🗂️ Training Set**: Multi-domain

7. **🗃️ Dataset**: Aircraft, Caltech101, CIFAR100, DTD

8. **⚙️ Optimizer**: 🔁 `SGD` vs 🌊 `SAM`

   **⚙️ Hyperparameters**:
   ‣ Learning Rate:  (5.5e-1 3e-2 3e-2 1.2e-1)
   ‣ SAM ρ:   0.05 0.10 0.20

9. **🧩 Task Setting**: Multi-domain Task-Incremental

10. **🔧 Experiment Setup**:

    - 
    - 

11. **📖 Reference**: 

     InfLoRA   [InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning | IEEE Conference Publication | IEEE Xplore ](https://ieeexplore.ieee.org/document/10658274) 

    Available Code

    [liangyanshuo/InfLoRA: The official implementation of the CVPR'2024 work Interference-Free Low-Rank Adaptation for Continual Learning](https://github.com/liangyanshuo/InfLoRA)

    

12. **📊 Results**: 

13. **🧾 Conclusion**: 









# Q4 Dose SAM also helps in NLP or finetune  Model ?

We find some  related Paper :

[[2110.08529\] Sharpness-Aware Minimization Improves Language Model Generalization](https://arxiv.org/abs/2110.08529)

1. **📚 Scope**: 🖼️  NLP
2. **🧠 Base Model**:  GPT-3, T5

Conclusion:

 Sharpness-Aware Minimization (SAM), a recently proposed optimization procedure that encourages convergence to flatter minima, can substantially improve the generalization of language models without much computational overhead. We show that SAM is able to boost performance on SuperGLUE, GLUE, Web Questions, Natural Questions, Trivia QA, and TyDiQA, with particularly large gains when training data for these tasks is limited.



[[2106.01548\] When Vision Transformers Outperform ResNets without Pre-training or Strong Data Augmentations](https://arxiv.org/abs/2106.01548)



Conclusion :

By promoting smoothness with a recently proposed sharpnessaware optimizer, we substantially improve the accuracy and robustness of ViTs and MLP-Mixers on various tasks spanning supervised, adversarial, contrastive, and transfer learning.

We show that the improved smoothness attributes to sparser active neurons in the first few layers.











# Q5  How SAM influence the CL-Lora's performance?

Several studies have shown that Sharpness-Aware Minimization (SAM) can improve the flatness of the loss landscape, leading to better performance both in classical machine learning tasks and in fine-tuning large pre-trained models. 

[[2110.08529\] Sharpness-Aware Minimization Improves Language Model Generalization](https://arxiv.org/abs/2110.08529) are actually focus on  in-task generalization rather than task-transfer performance.





[[2106.01548\] When Vision Transformers Outperform ResNets without Pre-training or Strong Data Augmentations](https://arxiv.org/abs/2106.01548)  evaluates the generalization ability of Sharpness-Aware Minimization (SAM) across various model architectures, including ResNet, Vision Transformers (ViT), and MLP-Mixers, under diverse settings such as standard image classification, out-of-distribution robustness, low-data regimes, and different model capacities. Empirical results demonstrate that SAM substantially improves test performance and robustness, particularly for architectures lacking strong inductive biases. However, the study primarily focuses on generalization with respect to input distribution shifts and does not investigate task-level transfer generalization in a systematic manner. 



However, these works do not focus on the continual learning (CL) setting with LoRA-based adaptation. Therefore, we aim to investigate how SAM affects the performance of CL-LoRA. This raises several related questions.







### Q5.1 Does a larger rank $r$ lead to smaller sharpness?

- Our answer is: **the less you change the model, the flatter the loss landscape tends to be**.

In our experiment 1.2 (results shown in slides 55 and 56), we can see that small-rank LoRA has less influence on the overall model compared to larger-rank LoRA. This suggests that when the LoRA rank is smaller, the whole model tends to remain flatter.

If we consider the extreme case, it raises the question: which model is flatter—the original pre-trained model or the fine-tuned one?

And as shown in slide 64, full fine-tuning results in the sharpest model among the three. 

 

**Maybe the result is not very clear. We should further investi .**



## Q5.2  Does SAM help CL-LoRA setting ? 



We can reuse the paper's setting , and only try to change Optimizer to SAM with different noise to check the result.  

There are some available setting from paper  which are under setting CL-LoRA





1 O-LoRA/ N-LoRA

Paper: 

[[2410.10179\] Is Parameter Collision Hindering Continual Learning in LLMs?](https://arxiv.org/abs/2410.10179)

Code :  

[PKU-YuanGroup/N-LoRA: 【COLING 2025🔥】Code for the paper "Is Parameter Collision Hindering Continual Learning in LLMs?".](https://github.com/PKU-YuanGroup/N-LoRA)

1. **📚 Scope**:  **NLP**

2. **🧠 Base Model**: `T5-small`

3. **🗂️ Training Set**:   Multi-task continual training  follow paper's setting

4. **🗃️ Dataset**: `DBpedia`  **amazon ** **agnews**   **yahoo**

5. **⚙️ Optimizer**:  `AdamW (HuggingFace)`

I will first check if we can change the optimizer to SAM. 



2 InfLoRA 

Paper : 

[InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/document/10658274)

Code : 

[liangyanshuo/InfLoRA: The official implementation of the CVPR'2024 work Interference-Free Low-Rank Adaptation for Continual Learning](https://github.com/liangyanshuo/InfLoRA)

Note Here the model is ViT , we need add a classifier for each task.

1. **📚 Scope**: 🖼️ **CV**

2. **🧠 Base Model**: `ViT-`  change the final one-layer classifier.

3. **🗂️ Training Set**:   Multi-task continual training  follow paper's setting

4. **🗃️ Dataset**: 

5. **⚙️ Optimizer**: 



