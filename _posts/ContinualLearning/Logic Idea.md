# Logic Idea





# Q1  Flatness helps in generalization/Performance ?



## Q 1.1 What the Generalization means?



### **Train→Test** 



We consider a standard supervised learning setting, where:

- $\mathcal{X}$ is the input space and $\mathcal{Y}$ the output space;
- $\mathcal{D}$ is an unknown data distribution over $\mathcal{X} \times \mathcal{Y}$;
- A training dataset $S = {(x_1, y_1), \dots, (x_n, y_n)}$ is sampled i.i.d. from $\mathcal{D}$.

Let $f \in \mathcal{F}$ denote a hypothesis from the hypothesis space, and $\ell: \mathcal{F} \times \mathcal{X} \times \mathcal{Y} \to \mathbb{R}_{\geq 0}$ be a loss function measuring prediction error.

#### Empirical Risk

The empirical risk (training error) of $f$ on $S$ is defined as:

$\hat{L}_S(f) := \frac{1}{n} \sum_{i=1}^n \ell(f, x_i, y_i)$

It quantifies how well the model fits the training data and is directly computable.

#### Generalization Error

The generalization error (expected risk) of $f$ is its expected loss over the true data distribution:

$L_{\mathcal{D}}(f) := \mathbb{E}_{(x, y) \sim \mathcal{D}}[\ell(f, x, y)]$

While training aims to minimize $\hat{L}*S(f)$, the ultimate goal is to find a model that performs well on unseen data, i.e., minimizes $L*{\mathcal{D}}(f)$. Ideally, we want:

$\hat{f} := \arg\min_{f \in \mathcal{F}} \hat{L}_S(f) \quad \text{such that} \quad L_{\mathcal{D}}(\hat{f}) \text{ is also small}$

**However, minimizing training error does not guarantee low test error** due to potential overfitting, especially when the model memorizes noise in $S$ rather than learning generalizable patterns.

#### Convergence via Law of Large Numbers

According to the Law of Large Numbers, for a fixed $f$ and **i.i.d. samples** $(x_i, y_i) \sim \mathcal{D}$, the empirical risk is an unbiased estimator of the generalization error and converges almost surely:

$\hat{L}_n(f) = \frac{1}{n} \sum_{i=1}^n \ell(f, x_i, y_i) \xrightarrow{\text{a.s.}} L_{\mathcal{D}}(f)$

As $n \to \infty$, the empirical risk approaches the true risk.

#### Generalization Bound

In practice, we desire a *quantitative* and finite-sample characterization of the gap between empirical and true risk. A typical generalization bound takes the form:

$L_{\mathcal{D}}(f) \leq \hat{L}_S(f) + \text{complexity penalty}$

where the penalty term depends on the hypothesis space $\mathcal{F}$, sample size $n$, and confidence level $\delta$.

This leads to the refined learning objective:

$\hat{f} := \arg\min_{f \in \mathcal{F}} \left[ \hat{L}_S(f) + \text{complexity penalty} \right]$

Optimizing this upper bound provides a more robust approach to achieving good generalization, especially under limited data.



#### related Paper

**PAC - Bayesian Th







### **Big Pretrained Model →Downstream Task**











### **Cross Task Generalization** 











## Q1.2 What flatness means?









### Q 1.3  The relizationship between Generalization  with  Flat  Loss Land



[[1706.10239\] Towards Understanding Generalization of Deep Learning: Perspective of Loss Landscapes](https://arxiv.org/abs/1706.10239)







## Q 1.4  Evidence from experiment 







## Q1.5 Possible Explanation









# Q2  Noise contributes to flatness（loss land）？

