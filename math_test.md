---
layout: default
title: "Math Rendering Test"
---

# Math Rendering Test Page

This page tests various mathematical expressions to ensure MathJax is working correctly.

## Inline Math

Here is some inline math: $E = mc^2$ and $\alpha + \beta = \gamma$.

The quadratic formula is $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$.

## Display Math

The Bayesian formula:

$$P(\text{diagnosis} | \text{symptoms}, \text{history}) = \frac{P(\text{symptoms} | \text{diagnosis}) \cdot P(\text{diagnosis} | \text{history})}{P(\text{symptoms} | \text{history})}$$

Attention mechanism:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

Matrix operations:

$$\mathbf{A} = \begin{pmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{pmatrix}$$

## Complex Expressions

$$\sum_{i=1}^{n} x_i = x_1 + x_2 + \cdots + x_n$$

$$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$$

If all equations above render properly, MathJax is working correctly!
