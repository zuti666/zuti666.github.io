# -*- coding: utf-8 -*-

"""
@author     : zuti
@software   : PyCharm
@file       : gan_torch_gpt.py.py
@time       : 03/10/2023 10:51
@desc       ：

"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torch.autograd import Variable
from torchvision import datasets, transforms

# Define the transformations for the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# Load the MNIST dataset
mnist = datasets.MNIST(root='D:\LocalGit\generative-models-master\data\MNIST',train=True,transform=transform ,download=True)
data_loader = torch.utils.data.DataLoader(mnist, batch_size=64, shuffle=True, drop_last=True)

mb_size = 64
Z_dim = 100
X_dim = 784  # MNIST images are 28x28, so the flattened dimension is 784
h_dim = 128
c = 0
lr = 1e-3

# Generator model
class Generator(torch.nn.Module):
    def __init__(self):
        # 定义生成器的权重与偏置项。
        # 输入层为100个神经元且接受随机噪声，输出层为784个神经元，并输出手写字体图片。
        # 生成网络根据原论文为三层全连接网络
        super(Generator, self).__init__()
        self.fc1 = torch.nn.Linear(Z_dim, h_dim)
        self.fc2 = torch.nn.Linear(h_dim, X_dim)


    def forward(self, z):
        h = F.relu(self.fc1(z))
        X = torch.sigmoid(self.fc2(h))
        return X

# Discriminator model
class Discriminator(torch.nn.Module):
    def __init__(self):
        # 定义判别器的权重矩阵和偏置项向量，由此可知判别网络为三层全连接网络
        super(Discriminator, self).__init__()
        self.fc1 = torch.nn.Linear(X_dim, h_dim)
        self.fc2 = torch.nn.Linear(h_dim, 1)

    # 定义判别器的前向传播函数D(X),其中X是输入的手写数字图片，
    # 经过一系列全连接层和激活函数， 输出一个值，表示输入图片的真实性。
    def forward(self, X):
        h = F.relu(self.fc1(X))
        y = torch.sigmoid(self.fc2(h))
        return y

G = Generator()
D = Discriminator()

# Optimizers
G_solver = optim.Adam(G.parameters(), lr=1e-3)
D_solver = optim.Adam(D.parameters(), lr=1e-3)

ones_label = Variable(torch.ones(mb_size, 1))
zeros_label = Variable(torch.zeros(mb_size, 1))

def reset_grad():
    G.zero_grad()
    D.zero_grad()

# Training loop
for it in range(100000):
    for X, _ in data_loader:
        X = X.view(-1, X_dim)
        X = Variable(X)

        # Discriminator forward-loss-backward-update
        z = Variable(torch.randn(mb_size, Z_dim))
        G_sample = G(z)
        D_real = D(X)
        D_fake = D(G_sample)

        """计算判别器的损失，包括真实图片的损失和生成图片的损失，并更新判别器的参数。"""
        D_loss_real = F.binary_cross_entropy(D_real, ones_label)  # 真实图片的损失
        D_loss_fake = F.binary_cross_entropy(D_fake, zeros_label) # 生成图片的损失
        D_loss = D_loss_real + D_loss_fake

        # 更新判别器的参数
        D_loss.backward()
        D_solver.step()

        # Housekeeping - reset gradient
        reset_grad()

        # Generator forward-loss-backward-update
        """重新生成一批噪声数据z，用生成器生成图片，并计算生成器的损失，然后更新生成器的参数。"""
        z = Variable(torch.randn(mb_size, Z_dim))
        G_sample = G(z)
        D_fake = D(G_sample)

        G_loss = F.binary_cross_entropy(D_fake, ones_label)

        G_loss.backward()
        G_solver.step()

        # Housekeeping - reset gradient
        reset_grad()

    # Print and plot every now and then
    if it % 1000 == 0:
        # 在每1000次迭代后，打印当前的判别器损失和生成器损失。
        print('Iter-{}; D_loss: {}; G_loss: {}'.format(it, D_loss.data.numpy(), G_loss.data.numpy()))

        """可视化生成器生成的一批样本图片，将其保存在输出文件夹中。"""
        samples = G(z).data.numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
        c += 1
        plt.close(fig)
