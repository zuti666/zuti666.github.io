---
layout: wiki
title: Diffusion Model 
categories: Diffusion Model 
description: Diffusion Model  相关算法
keywords: Diffusion Model 
---



# Diffusion Model 







[GitHub - huggingface/diffusers: 🤗 Diffusers: State-of-the-art diffusion models for image, video, and audio generation in PyTorch and FLAX.](https://github.com/huggingface/diffusers)





[使用Diffusers来实现Stable Diffusion 🧨](https://huggingface.co/blog/zh/stable_diffusion)



[“Snippets: Importing libraries”的副本 - Colab](https://colab.research.google.com/drive/1HT5ZXXRJOcPJSnxu00mlP9KkitAw4-Lx)



[“diffusers_training_example.ipynb”的副本 - Colab](https://colab.research.google.com/drive/1U-5oadm1cqCWYtg5qqECEYA1fkDNtgzv)



wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh





```mermaid
graph TD;
    subgraph "Encoder (Downsampling)"
        A["Input Image"] -->|"DownBlock2D (128)"| B
        B -->|"DownBlock2D (128)"| C
        C -->|"DownBlock2D (256)"| D
        D -->|"DownBlock2D (256)"| E
        E -->|"AttnDownBlock2D (512)"| F
        F -->|"DownBlock2D (512)"| Bottleneck["Bottleneck Layer"]
    end
    
    subgraph "Decoder (Upsampling)"
        Bottleneck -->|"UpBlock2D (512)"| G
        G -->|"AttnUpBlock2D (512)"| H
        H -->|"UpBlock2D (256)"| I
        I -->|"UpBlock2D (256)"| J
        J -->|"UpBlock2D (128)"| K
        K -->|"UpBlock2D (128)"| L
        L -->|"Output Image"| M
    end

   

```

