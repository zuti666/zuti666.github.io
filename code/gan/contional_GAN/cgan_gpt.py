# -*- coding: utf-8 -*-

"""
@author     : zuti
@software   : PyCharm
@file       : cgan_gpt.py
@time       : 03/10/2023 11:50
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
mnist = datasets.MNIST('D:\LocalGit\generative-models-master\data\MNIST', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(mnist, batch_size=64, shuffle=True, drop_last=True)

mb_size = 64
Z_dim = 100
X_dim = 784  # MNIST images are 28x28, so the flattened dimension is 784
y_dim = 10   # There are 10 classes in MNIST
h_dim = 128
cnt = 0  # 展示生成过程中的图片的编号
lr = 1e-3

# Generator model
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = torch.nn.Linear(Z_dim + y_dim, h_dim)
        self.fc2 = torch.nn.Linear(h_dim, X_dim)

    def forward(self, z, c):
        inputs = torch.cat([z, c], 1)
        h = F.relu(self.fc1(inputs))
        X = torch.sigmoid(self.fc2(h))
        return X

# Discriminator model
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = torch.nn.Linear(X_dim + y_dim, h_dim)
        self.fc2 = torch.nn.Linear(h_dim, 1)

    def forward(self, X, c):
        inputs = torch.cat([X, c], 1)
        h = F.relu(self.fc1(inputs))
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
    for X, y in data_loader:
        X = X.view(-1, X_dim)
        X = Variable(X)
        c = Variable(torch.zeros(mb_size, y_dim))
        c.scatter_(1, y.view(-1, 1), 1)  # One-hot encode the labels

        # Discriminator forward-loss-backward-update
        z = Variable(torch.randn(mb_size, Z_dim))
        G_sample = G(z, c)
        D_real = D(X, c)
        D_fake = D(G_sample, c)




        D_loss_real = F.binary_cross_entropy(D_real, ones_label)
        D_loss_fake = F.binary_cross_entropy(D_fake, zeros_label)
        D_loss = D_loss_real + D_loss_fake

        D_loss.backward()
        D_solver.step()

        # Housekeeping - reset gradient
        reset_grad()

        # Generator forward-loss-backward-update
        z = Variable(torch.randn(mb_size, Z_dim))
        G_sample = G(z, c)
        D_fake = D(G_sample, c)

        G_loss = F.binary_cross_entropy(D_fake, ones_label)

        G_loss.backward()
        G_solver.step()

        # Housekeeping - reset gradient
        reset_grad()

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D_loss: {}; G_loss: {}'.format(it, D_loss.data.numpy(), G_loss.data.numpy()))

        c = Variable(torch.zeros(mb_size, y_dim))
        c.scatter_(1, torch.randint(0, y_dim, (mb_size, 1)), 1)  # Randomly select a class
        samples = G(z, c).data.numpy()[:16]

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

        plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)
