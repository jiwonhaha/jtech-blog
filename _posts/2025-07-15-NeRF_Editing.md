---
title: "NeRF-Editing: Geometry Editing of Neural Radiance Fields"
date: 2025-07-15
categories:
  - paper review
tags:
  - Scene representation
  - View synthesis
  - Neural Rendering
  - NeRF
  - 3D Vision
  - 3D Scene Editing
use_math: true
classes: wide
---


![N-edit]({{ site.baseurl }}/assets/NeRF_Editing/main.png)

## Introduction


In this post, I review **NeRF-Editing: Geometry Editing of Neural Radiance Fields**, a paper presented at CVPR 2022 by Yu-Jie Yuan et al. The work introduces a novel method to deform geometry in static NeRF scenes without retraining, which is a significant advancement over previous NeRF-editing efforts that were limited to appearance or rigid transformation edits.


---

## Paper Info

- **Title**: *NeRF-Editing: Geometry Editing of Neural Radiance Fields*
- **Authors**: Yu-Jie Yuan, Yang-Tian Sun, Yu-Kun Lai,Yuewen Ma, Rongfei Jia, Lin Gao
- **Conference**: CVPR 2022  
- **Paper Link**: [ArXiv](https://arxiv.org/abs/2205.04978)

---
## NeRF: A Brief Overview

Neural Radiance Fields (NeRF) model a scene by learning a continuous volumetric representation through an MLP. Given a camera ray, NeRF samples multiple 3D points along it and predicts the color and density of each point. These predictions are integrated using volume rendering to synthesize the pixel color.

The function NeRF learns is:

$$
F_\Theta(\zeta(x), \zeta(d)) \rightarrow (\sigma, c)
$$


Where:
- $$x \in \mathbb{R}^3$$ is a 3D position
- $$d \in \mathbb{S}^2$$ is the viewing direction
- $$\zeta$$ is positional encoding
- $$\sigma$$ is volume density, $$c$$ is RGB color

For a full technical review of NeRF, refer to my earlier post:

👉 [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://jiwonhaha.github.io/jtech-blog/paper%20review/NeRF/)

## Key Idea of NeRF-Editing

NeRF-Editing addresses the lack of editable geometry in standard NeRF. It introduces a hybrid **explicit-implicit deformation pipeline**:

1. Extract a **triangular mesh** from trained NeRF (using NeuS)
2. Allow user-controlled **deformation** of the mesh
3. Construct a **tetrahedral mesh proxy** wrapping the triangle mesh
4. Transfer mesh deformation to the tetrahedral mesh
5. Use the deformed mesh to **bend rays** during rendering

This enables **photorealistic and view-consistent shape deformation** without re-training the NeRF model.

---

## NeRF-Editing

![N-edit_explain]({{ site.baseurl }}/assets/NeRF_Editing/explain.png)

### 1. NeRF Training & Mesh Extraction

- A NeRF is trained from multi-view input images using standard supervision.
- Instead of extracting mesh via marching cubes (which gives noisy surfaces), the method uses **NeuS** to extract a high-quality mesh as a **signed distance function (SDF)**.

### 2. Mesh-Based Editing (ARAP)

- The extracted triangle mesh is deformed using **As-Rigid-As-Possible (ARAP)** editing.
- ARAP minimizes local distortion:

$$
E(S') = \sum_{i} \sum_{j \in N(i)} w_{ij} \| (v'_i - v'_j) - R_i(v_i - v_j) \|^2
$$

- Users can use traditional 3D editing tools to create intuitive deformations.

### 3. Building Tetrahedral Proxy

- A **tetrahedral mesh** is constructed around the triangle mesh using **TetWild**.
- The triangle mesh's deformation is transferred to the tetrahedral mesh using **barycentric coordinate constraints**.

Constraint-based energy minimization:

$$
\min E(T') \quad \text{s.t. } A t' = v'
$$

Where:
- $$A$$: barycentric matrix
- $$t'$$: deformed tetrahedral vertices
- $$v'$$: deformed triangle mesh vertices

### 4. Ray Bending and Rendering

- During rendering, for each ray sample point:
  1. Find the corresponding tetrahedron
  2. Use barycentric interpolation to compute displacement $$\Delta p$$
  3. Apply displacement to query NeRF:

$$
F_\Theta(\zeta(p + \Delta p), \zeta(d)) \rightarrow (\sigma, c)
$$

- Use volume rendering to synthesize the final pixel color.

This effectively "bends" rays through the deformed geometry.

---

## Results

![N-edit_result]({{ site.baseurl }}/assets/NeRF_Editing/result.png)

### Datasets
- **Synthetic**: Mixamo, Lego, Chair (NeRF dataset)
- **Real**: Giraffe plush toy, horse statue, chair, laptop

### Metrics
- **SSIM**, **PSNR**, **LPIPS** for synthetic
- **FID** for real scenes (since GT does not exist)

### Performance

| Method         | SSIM ↑ | LPIPS ↓ | PSNR ↑ | FID (Real) ↓ |
|----------------|--------|----------|---------|---------------|
| Closest Point | 0.928  | 0.055    | 22.38   | 300.8         |
| 3NN           | 0.941  | 0.042    | 24.30   | 291.7         |
| **Ours**      | **0.975**  | **0.024**    | **29.62**   | **253.7**         |

The Closest Point method transfers mesh deformation by assigning to each sampled ray point the displacement of its closest mesh vertex. While simple, this approach leads to artifacts near edges and thin parts.

The 3NN (Three Nearest Neighbors) baseline improves on this by interpolating displacements from the three closest mesh vertices, weighted by inverse distance. However, it still suffers from discontinuities.

Their method uses a tetrahedral mesh to propagate deformations via barycentric interpolation in 3D space, resulting in smoother, more accurate, and view-consistent edits without retraining the NeRF.

---

## Limitations

- **Appearance is static**: Lighting/shadow do not change after deformation.
- **Non-realtime**: Rendering remains slow; interaction is not instantaneous.

---

## Conclusion

NeRF-Editing is a powerful step toward making neural scene representations **editable and interactive**. It successfully combines the intuitive control of mesh editing with the photorealism of NeRF.

Future directions include:
- Adding **relighting** to fix appearance artifacts after deformation
- Integrating **real-time NeRF acceleration** frameworks for interactive feedback

---
