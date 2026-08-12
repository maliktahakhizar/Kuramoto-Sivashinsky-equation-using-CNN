# Spatiotemporal Chaos Classification in Kuramoto–Sivashinsky Dynamics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)

> **Academic Context & Attribution**  
> This project is based on **Exercise 6.4** from the textbook *Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control* by **Steven L. Brunton** and **J. Nathan Kutz** (Cambridge University Press).

---

## Overview

This repository provides an end-to-end framework combining scientific computing and computational fluid dynamics / machine learning:
1. **FFT Pseudo-Spectral Solver:** Numerically solves the non-linear 1D Kuramoto–Sivashinsky (KS) partial differential equation using an implicit BDF method in Fourier space.
2. **Patch-Based Data Pipeline:** Slices the resulting spatio-temporal scalar field $u(t, x)$ into uniform $64 \times 64$ 2D snapshots across three domain regimes.
3. **2D CNN Classifier & Length Scale Extraction:** Sweeps the spatial width $k_x$ of the initial convolutional kernel to determine the minimum physical receptive field required to classify chaotic dynamics and isolate the system's characteristic wavelength.

---

## Key Results & Physical Insights

* **Characteristic Wavelength Identification:** Sweeping spatial kernel widths across $k_x \in \{3, 7, 15, 31, 63, 127\}$ shows a distinct jump in performance to **95.6% test accuracy** at $k_x = 63$.
* **Physics–CNN Receptive Field Alignment:** With spatial discretization $dx = 0.1$, a kernel width of $k_x = 63$ corresponds to a physical spatial receptive field of:
  $$\text{Length Scale} = k_x \cdot dx = 63 \times 0.1 = 6.3 \text{ spatial units}$$
  This matches the fundamental spatial instability wavelength of the Kuramoto–Sivashinsky system ($\lambda = 2\pi \approx 6.28$).
* **Multi-Wavelength Saturation:** Expanding $k_x$ to 127 ($12.7$ units $\approx 2\lambda$) achieves **98.8% classification accuracy** by capturing long-range spatial defect interactions across wave periods.

### Accuracy vs. Spatial Kernel Width ($k_x$)

| Kernel Shape $(k_t, k_x)$ | Physical Width ($k_x \cdot dx$) | Wavelength Ratio ($\lambda \approx 6.28$) | Test Accuracy |
| :---: | :---: | :---: | :---: |
| $(3, 3)$ | $0.3$ units | $0.05 \lambda$ | **80.0%** |
| $(3, 7)$ | $0.7$ units | $0.11 \lambda$ | **86.9%** |
| $(3, 15)$ | $1.5$ units | $0.24 \lambda$ | **90.6%** |
| $(3, 31)$ | $3.1$ units | $0.50 \lambda$ ($\frac{1}{2}\lambda$) | **90.6%** |
| $(3, 63)$ | $6.3$ units | $1.00 \lambda$ ($1\lambda$) | **95.6%** |
| $(3, 127)$ | $12.7$ units | $2.00 \lambda$ ($2\lambda$) | **98.8%** |
`CNN_accuracy.png`
---

## Mathematical Background

The 1D Kuramoto–Sivashinsky equation is given by:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + \frac{\partial^2 u}{\partial x^2} + \frac{\partial^4 u}{\partial x^4} = 0$$

Using the conservative non-linear identity $u u_x = \frac{1}{2}(u^2)_x$, transforming into Fourier space yields:

$$\frac{\partial \hat{u}}{\partial t} = \left( k^2 - k^4 \right) \hat{u} - \frac{i k}{2} \mathcal{F}\left\{ u^2 \right\}$$

### Regimes Generated
By varying domain length $L$, three distinct attractor states are synthesized:
* **Class 0 (Periodic / Steady):** $L = 16.0$
* **Class 1 (Weak Chaos / Intermittent):** $L = 28.0$
* **Class 2 (Fully Developed Chaos):** $L = 100.0$

---

