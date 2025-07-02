---
title: "GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians"
date: 2025-07-02
categories:
  - paper review
tags:
  - Scene representation
  - View synthesis
  - Neural Rendering
  - Gaussian Splatting
  - 3D Vision
use_math: true
classes: wide
---

![GSAV]({{ site.baseurl }}/assets/GSAV/GSAV_main.png)

## Introduction

In this post, I review **GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians**, a recent paper presented at CVPR 2024 by Shenhan Qian et al. This work builds on the success of **3D Gaussian Splatting** and extends it to model **photorealistic dynamic head avatars** using a **rigged and animatable 3D Gaussian-based representation**. Unlike static scene models, GaussianAvatars focuses on **realistic facial expressions, head motion**, and **view-dependent appearance**, making it a powerful tool for virtual telepresence, AR/VR, and character animation.

---

## Paper Info

- **Title**: *GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians*
- **Authors**: Shenhan Qian, Tobias Kirschstein, Liam Schoneveld, Davide Davoli, Simon Giebenhain, Matthias Nießner  
- **Conference**: CVPR 2024  
- **Paper Link**: [ArXiv:2312.02069](https://arxiv.org/abs/2312.02069)

---

## Motivation

Previous works like Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) excel at rendering static scenes, but they struggle with:
- Real-time performance
- Animatable human models
- High-fidelity rendering under dynamic conditions (e.g., expressions, head turns)

GaussianAvatars address these limitations by introducing:
- **A dynamic, rigged 3D Gaussian representation**
- **Blendshape-based expression modeling**
- **An efficient rasterization pipeline** for real-time rendering

---

## Method Overview

![GSAV]({{ site.baseurl }}/assets/GSAV/GSAV_explain.png)

### 1. **Rigged 3D Gaussian Representation**

Each **head avatar** is modeled as a collection of **anisotropic 3D Gaussians**:
- Each Gaussian has a **mean position**, **scale**, **rotation (via quaternion)**, **opacity**, and **view-dependent color**.
- Gaussians are **attached to a blendshape rig**, which allows dynamic expression modeling using blendshape weights.
  
This enables **pose-dependent deformation** of Gaussians, allowing the model to simulate muscle movements and facial expressions.

---

### 2. **View-Dependent Appearance Modeling**

To achieve photorealism, each Gaussian stores a **view-dependent color appearance** using **learned basis functions**, similar to **spherical harmonics (SH)** or neural MLPs:
- Appearance varies smoothly with camera viewpoint
- Lighting and specular effects are preserved

---

### 3. **Training Pipeline**

GaussianAvatars extend the static 3D Gaussian Splatting (3DGS) method to handle **dynamic, expression-driven facial motion** by introducing a **rigged deformation model** and adapting the optimization pipeline accordingly.

#### 🔄 Extension over 3DGS

Unlike 3DGS, which optimizes a static set of Gaussians \( \{ \mathcal{G}_j \} \), GaussianAvatars define **rigged Gaussians** that deform according to **blendshape-based facial rigs**. Each Gaussian is linked to blendshape bones through skinning weights and is updated based on current expression parameters \( \mathbf{w} \in \mathbb{R}^K \).

The updated position of each Gaussian becomes:

$$
\mathbf{x}_j(\mathbf{w}) = \sum_{k=1}^{K} w_k \cdot \Delta \mathbf{x}_{j}^{(k)} + \mathbf{x}_j^0
$$

Where:
- $$\( \mathbf{x}_j^0 \)$$: rest position of Gaussian j
- $$\( \Delta \mathbf{x}_j^{(k)} \)$$: displacement under blendshape component k

The output color and opacity are also view-dependent:

$$
F_j(x_j(\mathbf{w}), v) \rightarrow (r_j, g_j, b_j, \alpha_j)
$$

---

#### 🎯 Training Objective (Extended 3DGS Loss)

GaussianAvatars retain the photometric supervision of 3DGS but augment it for dynamic expressions. The total loss is:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{photo}} + \lambda_{\text{expr}} \| \mathbf{w} \|_2^2 + \lambda_{\text{sparse}} \sum_j \alpha_j^2
$$

This enables the system to learn how each Gaussian should deform and appear under various expressions and views.

---

### 4. **Real-Time Rendering via Splatting**

GaussianAvatars preserve the rasterization and splatting pipeline from 3DGS, but extend it to support **expression-dependent Gaussian deformation**.

#### 🧠 Deformation-aware Splatting

The projection of each deformed Gaussian becomes:

$$
\mathbf{u}_j = \mathbf{P} \cdot \mathbf{x}_j(\mathbf{w})
$$

and its 2D projected covariance is:

$$
\Sigma_j^{(2D)}(\mathbf{w}) = J_j(\mathbf{w}) \Sigma_j J_j(\mathbf{w})^\top
$$

Rendering is done using the over-operator:

$$
\hat{I}(\mathbf{u}) = \sum_{j} T_j(\mathbf{u}) \cdot \alpha_j \cdot F_j(x_j(\mathbf{w}), v)
$$

Where transmittance is defined as:

$$
T_j(\mathbf{u}) = \prod_{k < j} (1 - \alpha_k)
$$

---


## Key Contributions

- **Dynamic Gaussian Avatar Model**: First method to rig 3D Gaussians for animatable facial avatars.
- **Blendshape Rig Integration**: Enables realistic expression and motion modeling via parametric blendshape control.
- **Real-Time Rendering**: Achieves interactive speeds using GPU rasterization.
- **Photorealistic Output**: High fidelity results comparable or superior to NeRF-based methods but with significantly faster inference.

---

## Results


![GSAV]({{ site.baseurl }}/assets/GSAV/GSAV_result.png)

- Outperforms NeRF and other neural rendering methods in **speed**, **visual quality**, and **animatability**
- Produces **photorealistic facial renderings** with **subtle expression dynamics**
- Can generalize across expressions not seen during training

![Results from the paper](https://arxiv.org/abs/2312.02069)

---

## Limitations and Future Work

- Limited to **head-only avatars**; extending to full body or upper body remains an open challenge
- Requires **multi-view data for training**
- Generalization to **novel subjects** is not addressed (subject-specific training)

Future directions may include:
- **Full-body avatars** using similar Gaussian rigs
- **Expression transfer** between identities
- **Few-shot personalization** with monocular input

---

## Conclusion

GaussianAvatars demonstrate that **rigged 3D Gaussians** can be used not just for static scenes, but also for **high-quality dynamic facial avatars**. This work significantly pushes forward the frontier in real-time neural rendering and avatar modeling, offering a promising foundation for virtual humans, telepresence, and digital actors.

---
