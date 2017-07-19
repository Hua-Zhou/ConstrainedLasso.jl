# ConstrainedLasso

**ConstrainedLasso** estimates the following constrained lasso problem, using the approach of Gaines and Zhou (2016).

<center><a href="https://www.codecogs.com/eqnedit.php?latex=\text{minimize}&space;\hspace{1em}&space;\frac&space;12&space;||\boldsymbol{y}-\boldsymbol{X\beta}||^2_2&space;&plus;&space;\rho||\boldsymbol{\beta}||_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{minimize}&space;\hspace{1em}&space;\frac&space;12&space;||\boldsymbol{y}-\boldsymbol{X\beta}||^2_2&space;&plus;&space;\rho||\boldsymbol{\beta}||_1" title="\text{minimize} \hspace{1em} \frac 12 ||\boldsymbol{y}-\boldsymbol{X\beta}||^2_2 + \rho||\boldsymbol{\beta}||_1" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{&space;subject&space;to}&space;\hspace{0.5em}&space;\boldsymbol{A\beta}=\boldsymbol{b}&space;\text{&space;and&space;}&space;\boldsymbol{C\beta}&space;\leq&space;\boldsymbol{d}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{&space;subject&space;to}&space;\hspace{0.5em}&space;\boldsymbol{A\beta}=\boldsymbol{b}&space;\text{&space;and&space;}&space;\boldsymbol{C\beta}&space;\leq&space;\boldsymbol{d}" title="\text{ subject to} \hspace{0.5em} \boldsymbol{A\beta}=\boldsymbol{b} \text{ and } \boldsymbol{C\beta} \leq \boldsymbol{d}" /></a></center>

where <a href="http://www.codecogs.com/eqnedit.php?latex=\inline&space;\boldsymbol{y}&space;\in&space;\mathbb{R}^n" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\inline&space;\boldsymbol{y}&space;\in&space;\mathbb{R}^n" title="\boldsymbol{y} \in \mathbb{R}^n" /></a> is the response vector, <a href="http://www.codecogs.com/eqnedit.php?latex=\inline&space;\boldsymbol{X}\in&space;\mathbb{R}^{n\times&space;p}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\inline&space;\boldsymbol{X}\in&space;\mathbb{R}^{n\times&space;p}" title="\boldsymbol{X}\in \mathbb{R}^{n\times p}" /></a> is the design matrix of predictor or covariates, <a href="http://www.codecogs.com/eqnedit.php?latex=\inline&space;\boldsymbol{\beta}&space;\in&space;\mathbb{R}^p" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\inline&space;\boldsymbol{\beta}&space;\in&space;\mathbb{R}^p" title="\boldsymbol{\beta} \in \mathbb{R}^p" /></a> is the vector of unknown regression coefficients, and <a href="http://www.codecogs.com/eqnedit.php?latex=\inline&space;\rho&space;\geq&space;0" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\inline&space;\rho&space;\geq&space;0" title="\rho \geq 0" /></a> is a tuning parameter that controls the amount of regularization.

## Installation 

Within Julia, use the package manager to install **ConstainedLasso**:

```{julia}
Pkg.clone("https://github.com/Hua-Zhou/ConstrainedLasso.git")
```

This package supports Julia v0.6.

## Citation 

If you use ConstrainedLasso package in your research, please cite the following reference in the resulting publications:

*Gaines BR, Zhou H (2016) Algorithms for Fitting the Constrained Lasso. arXiv preprint arXiv:1611.01511.*
