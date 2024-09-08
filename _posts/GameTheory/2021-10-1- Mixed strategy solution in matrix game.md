---
layout: post
title:  矩阵博弈中的混合策略求解 mixed strategy solution in matrix game
categories: GameTheory
description:  介绍矩阵博弈的混合策略求解，最大最小值以及最小最大值，线性规划求解方法以及求解石头剪刀步的一个实例。
keywords: GameTheory
---

介绍矩阵博弈的混合策略求解，最大最小值以及最小最大值，线性规划求解方法以及求解石头剪刀步的一个实例。

## 1 混合策略

我们已经看到, 矩阵博弈可能没有鞍点或纯策略纳什均衡。然而, 当我们允许混合策略时, 均衡必定存在。令 $x=\left(x_1, \cdots, x_m\right)$ 表示行参与人的混合策略, $y=\left(y_1, \cdots, y_n\right)$ 为列参与人的混合策略。注意, $a_{i j}$ 是当行参与人以概率 1 选择第 $i$ 行且列参与人以概率 1 选择第 $j$ 列时行参与人的收益。此时, 列参与人的收益为 $-a_{i j}$ 。在伴随上述混合策略 $x$ 和 $y$ 的情形下, 行参与人的期望收益:
$$
=u_1(x, y)=\sum_{i=1}^m \sum_{j=1}^n x_i y_j a_{i j}=x A y
$$
其中 $x=\left(x_1, \cdots, x_m\right) ; y=\left(y_1, \cdots, y_n\right) ; A=\left[a_{i j}\right]$ 。在上面的表达式中, 我们稍微 滥用了符号, 因为我们本来应该用向量 $y$ 的转置 (即 $y^{\mathrm{T}}$ ) 但用了 $y$ 本身 (出于简单目 的, 我们在本章都这么做, 因为这不会导致混淆)。此时, 列参与人的期望收益 为 $-x A y$ 。当行参与人选择 $x$ 时, 他保证自己的期望收益为
$$
\min _{y \in \Delta\left(S_2\right)} x A y
$$
因此, 行参与人应该选择使得上述收益最大的混合策略 $x$ 。也就是说, 他应该选择 $x$ 使得
$$
\max _{x \in \Delta\left(S_1\right) y \in \Delta\left(S_2\right)} \min _x x A y
$$
换句话说, 行参与人的最优策略是**最大最小化** (maxminimization) 策略。注意, 这里 隐含地假设不管行参与人怎么选择, 列参与交与都会选择对他（行参与人）最不利的策略。行参与人在这样的背景下选择最优策略。行参与人选择的这种策略也称为行参与人的**安全策略**。

类似地, 当列参与人选择 $y$ 时，他保证自己的收益

$$
\begin{aligned}
& =\min _{x \in \Delta\left(S_{1}\right)}-x A y \\
& =-\max _{x \in \Delta\left(S_{1}\right)} x A y
\end{aligned}
$$

也就是说, 列参与人保证自己的损失不超过

$$
\max _{x \in \Delta\left(S_{1}\right)} x A y
$$

列参与人的最优策略应该使这个损失最小， 即

$$
\min _{y \in \Delta\left(S_{2}\right) x \in \Delta\left(S_{1}\right)} \max x A y
$$

这称为最小最大化 (minmaximization)。列参与人的这种策略也称为列参与人的安全 策略。

我们现在陈述和证明一个重要引理, 它断言若行参与人选择 $x$, 则在列参与人的最 优反应策略 $y$ 中至少存在一个纯策略。

### **引理** $9.1$

给定矩阵博恋 $A$ 以及混合策略 $x=\left(x_{1}, \cdots, x_{m}\right)$ 和 $y=\left(y_{1}, \cdots, y_{n}\right)$,
$$
\min _{y \in \Delta\left(S_{2}\right)} x A y=\min _{j} \sum_{i=1}^{m} a_{i j} x_{i}
$$

**证明**：对于给定的 $j$,
$$
 \sum_{i=1}^{m} a_{i j} x_{i}
$$
这个加和给出了当行参与人选择 $x=\left(x_{1}, \cdots, x_{m}\right)$ 且列参与人选择纯策略 $j$ 时, 行参与人的收益。因此,

$$
\min _{j} \sum_{i=1}^{m} a_{i j} x_{i}
$$

给出了当行参与人选择 $x$ 但列参与人能自由选择任何纯策略时行参与人的最小收益。由于纯策略是混合策略的一种特殊情形, 我们有

$$
\min _{j} \sum_{i=1}^{m} a_{i j} x_{i} \geqslant \min _{y \in \Delta\left(S_{2}\right)} x A y
$$

另一方面,

$$
x A y=\sum_{j=1}^{n} y_{j}\left(\sum_{i=1}^{m} a_{i j} x_{i}\right) \geqslant \sum_{j=1}^{n} y_{j}\left(\min _{j} \sum_{i=1}^{m} a_{i j} x_{i}\right)=\min _{j} \sum_{i=1}^{m} a_{i j} x_{i}\left(\text { 因为 } \sum_{j=1}^{n} y_{j}=1\right. \text { ) }
$$

因此, 我们有:

$$
x A y \geqslant \min _{j} \sum_{i=1}^{m} a_{i j} x_{i} \quad \forall x \in \Delta\left(S_{1}\right), \forall y \in \Delta\left(S_{2}\right)
$$

这意味着

$$
\min _{y \in \Delta\left(S_{2}\right)} x A y \geqslant \min _{j} \sum_{i=1}^{m} a_{i j} x_{i}
$$

根据式 $(9.1)$ 和式 $(9.2)$, 我们有

$$
\min _{y \in \Delta\left(S_{2}\right)} x A y=\min _{j} \sum_{i=1}^{m} a_{i j} x_{i}
$$

这样, 我们就完成了这个引理的证明。

作为上面引理的一个直接推论, 可以证明

$$
\max _{x \in \Delta\left(S_{1}\right)} x A y=\max _{i} \sum_{j=1}^{n} a_{i j} y_{j}
$$

使用上面的结果, 我们可以将行参与人以及列参与人的最优化问题描述如下。

## 行参与人的最优化问题 (最大最小化)

行参与人面对的最优化问题可以表示为

$$
\begin{aligned}
&\max \min _{j} \sum_{i=1}^{m} a_{i j} x_{i} \\
& \text { s. t.}  \\
& \sum_{i=1}^{m} x_{i}=1 \\
& x_{i} \geqslant 0, i=1, \cdots, m
\end{aligned}
$$

将上面这个问题称为问题 $P_{1}$ 。注意, 这个问题可以简练地表示为

$$
\max _{x \in \Delta\left(S_{1}\right)} \min _{y \in \Delta\left(S_{2}\right)} x A y_{\circ}
$$

## 列参与人的最优化问题 (最小最大化)

列参与人面对的最优化问题可以表示为

$$
\begin{aligned}
& \min \max _{i} \sum_{j=1}^{n} a_{i j} y_{j} \\
& \text { s. t. } \\
& \qquad \sum_{j=1}^{n} y_{j}=1 \\
& y_{j} \geqslant 0, j=1, \cdots, n
\end{aligned}
$$

将上面的问题称为问题 $P_{2}$ 。注意, 这个问题可以简练地写为

$$
\min _{y \in \Delta\left(S_{2}\right) x \in \Delta\left(S_{1}\right)} \max x A y
$$

下列命题说明问题 $P_{1}$ 和 $P_{2}$ 分别等价于适当的**线性规划 (linear program, LP)**。

### **命题** $9.3$ 

问题 $P_{1}$ 等价于下列线性规划 (我们将其称为线性规划 $L P_{1}$ ): 
$$
\begin{aligned}
& \max z \\
& \text { s. t.}  \\
& z-\sum_{i=1}^{m} a_{i j} x_{i} \leqslant 0, j=1, \cdots, n \\
& \sum_{i=1}^{m} x_{i}=1 \\
& x_{i} \geqslant 0, i=1, \cdots, m
\end{aligned}
$$

**证明**： 注意, $P_{1}$ 是一个最大化问题, 因此, 我们考察约束条件
$$
z-\sum_{i=1}^{m} a_{i j} x_{i} \leqslant 0, j=1, \cdots, n
$$

任何最优解 $\left(z^{*}, x^{*}\right)$ 将满足上述 $n$ 个不等式中的一个。也就是,

$$
z^{*}=\sum_{i=1}^{m} a_{i j} x_{i}^{*} \quad \text { 对于某个 } j \in\{1, \cdots, n\}
$$

令 $j^{*}$ 就是满足上式的 $j$ 值。于是

$$
z^{*}=\sum_{i=1}^{m} a_{i j}{ }^{*} x_{i}^{*}
$$

由于 $z^{*}$ 是线性规划 $L P_{1}$ 的一个可行解, 我们有

$$
\sum_{i=1}^{m} a_{i j}{ }^{*} x_{i}^{*} \leqslant \sum_{i=1}^{m} a_{i j} x_{i}^{*} \quad \forall j=1, \cdots, n
$$

这意味着

$$
\sum_{i=1}^{m} a_{i j}{ }^{*} x_{i}^{*}=\min _{j} \sum_{i=1}^{m} a_{i j} x_{i}^{*}
$$

如若不然, 我们有

$$
z^{*}<\sum_{i=1}^{m} a_{i j} x_{i} \quad \forall j=1, \cdots, n
$$

因此, 下列两个线性规划分别描述了行参与人与列参与人面对的最优化问题。

## 行参与人的线性规划 $\left(L P_{1}\right)$



$$
\begin{aligned}
\max z \\
& \text { s. t. }  \\
& z-\sum_{i=1}^{m} a_{i j} x_{i} \leqslant 0, j=1, \cdots, n \\
& \sum_{i=1}^{m} x_{i} \doteq 1 \\
& x_{i} \geqslant 0 \quad \forall i=1, \cdots, m
\end{aligned}
$$

## 列参与人的线性规划 $\left(\boldsymbol{L P _ { 2 }}\right)$

$$
\begin{aligned}
\min w  \\
& \text { s. t. }  \\
& w-\sum_{j=1}^{n} a_{i j} y_{j} \geqslant 0, i=1, \cdots, m \\
& \sum_{j=1}^{n} y_{j}=1 \\
& y_{j} \geqslant 0 \quad \forall j=1, \cdots, n
\end{aligned}
$$

## 例 $9.8$ (石头剪刀布博弈)

 对于石头剪刀布博弈, 回忆一下, 行参与人的收益矩 阵为
$$
A=\left[\begin{array}{rrr}
0 & -1 & 1 \\
1 & 0 & -1 \\
-1 & 1 & 0
\end{array}\right]
$$

行参与人的最优化问题 $P_{1}$ 为:

$$

$$



$$
\begin{aligned}
 & \max \min \left\{x_{2}-x_{3},-x_{1}+x_{3}, x_{1}-x_{2}\right\} \\
& \text { s. t. } \\
& x_{1}+x_{2}+x_{3}=1 \\
& x_{1} \geqslant 0 ; x_{2} \geqslant 0 ; x_{3} \geqslant 0
\end{aligned}
$$

上面的这个问题等价于线性规划 $L P_{1}$ :

$$
\begin{aligned}
& \max z \\
& \text { s. t. } \\
& z \leqslant x_{2}-x_{3} ; z \leqslant-x_{1}+x_{3} ; z \leqslant x_{1}-x_{2} \\
& x_{1}+x_{2}+x_{3}=1 ; x_{1} \geqslant 0 ; x_{2} \geqslant 0 ; x_{3} \geqslant 0
\end{aligned}
$$

列参与人的最优化问题 $P_{2}$ 为

$$

$$

$$
\begin{aligned}
&\min \max \left\{-y_{2}+y_{3}, y_{1}-y_{3},-y_{1}+y_{2}\right\} \\
& \text { s. t. } \\ 
& y_{1}+y_{2}+y_{3}=1 \\
& y_{1} \geqslant 0 ; y_{2} \geqslant 0 ; y_{3} \geqslant 0
\end{aligned}
$$

上面这个问题等价于线性规划 $L P_{2}$ :

$$
\begin{aligned}
\text { min } w \\
& \text { s. t. } \\
& w \geqslant-y_{2}+y_{3} ; w \geqslant y_{1}-y_{3} ; w \geqslant-y_{1}+y_{2} \\
& y_{1}+y_{2}+y_{3}=1 ; y_{1} \geqslant 0 ; y_{2} \geqslant 0 ; y_{3} \geqslant 0
\end{aligned}
$$

上面的线性规划问题使我们能够计算混合策略均衡。

## 参考文献

博弈论与机制设计

Game Theory and Mechanism Design Y.Narahari

时间有限，本博客就暂时写到这里，有时间会再补充更新一些细节。如果有问题或者对其中的一部分细节不理解，可以扫描下方二维码关注我的微信公众号或者联系我的邮箱 1978192989@qq.com，我会第一时间解答。

<center>
    <img src="/assets/images/qrcode.jpg" alt="picture not found" style="zoom:80%;" />
    <br>
</center>
