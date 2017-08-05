# ConstrainedLasso

**ConstrainedLasso** estimates the following constrained lasso problem

```math
\begin{split}
& \text{minimize} \hspace{1em} \frac 12||\boldsymbol{y}-\boldsymbol{X\beta}||^2_2 + \rho||\beta||_1 \\
& \text{ subject to} \hspace{0.5em} \boldsymbol{A\beta}=\boldsymbol{b} \text{ and } \boldsymbol{C\beta} \leq \boldsymbol{d}
\end{split}
```

where $\boldsymbol{y} \in \mathbb{R}^n$ is the response vector, $\boldsymbol{X}\in \mathbb{R}^{n\times p}$ is the design matrix of predictor or covariates, $\boldsymbol{\beta} \in \mathbb{R}^p$ is the vector of unknown regression coefficients, and $\rho \geq 0$ is a tuning parameter that controls the amount of regularization.

## Installation

Within Julia, use the package manager to install **ConstainedLasso**:

```{julia}
Pkg.clone("https://github.com/Hua-Zhou/ConstrainedLasso.git")
```

This package supports Julia v0.6.

## Citation

If you use ConstrainedLasso package in your research, please cite the following paper on the algorithms:

*B.R. Gaines, H. Zhou. (2016) Algorithms for Fitting the Constrained Lasso. <https://arxiv.org/abs/1611.01511>*

Original method paper on the constrained lasso is

*G.M. James, C. Paulson and P. Rusmevichientong. (2013) Penalized and constrained regression. <http://www-bcf.usc.edu/~gareth/research/PAC.pdf>*
