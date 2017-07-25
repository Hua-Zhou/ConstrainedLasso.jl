# ConstrainedLasso

| **Documentation**                                                                           | **Build Status**                                                              | **Code Coverage**                                                                            |
|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| [![Stable][docs-stable-img]][docs-stable-url] [![Latest][docs-latest-img]][docs-latest-url] | [![Travis][travis-img]][travis-url] [![Appveyor][appveyor-img]][appveyor-url] | [![Coverage Status][coveralls-img]][coveralls-url] [![codecov.io][codecov-img]][codecov-url] |






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

## Examples

Examples are found here in this [documentation] (http://github.com/Hua-Zhou/ConstrainedLasso.jl.git). 


## Citation 

If you use ConstrainedLasso package in your research, please cite the following reference in the resulting publications:

*Gaines BR, Zhou H (2016) Algorithms for Fitting the Constrained Lasso. arXiv preprint arXiv:1611.01511.*



[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://Hua-Zhou.github.io/ConstrainedLasso.jl/latest

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://Hua-Zhou.github.io/ConstrainedLasso.jl/stable

[travis-img]: https://travis-ci.org/Hua-Zhou/ConstrainedLasso.jl.svg?branch=master
[travis-url]: https://travis-ci.org/Hua-Zhou/ConstrainedLasso.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/wvaqu7i3ty2gk377/branch/master?svg=true
[appveyor-url]: https://ci.appveyor.com/project/Hua-Zhou/constrainedlasso-jl

[coveralls-img]: https://coveralls.io/repos/github/Hua-Zhou/ConstrainedLasso.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/github/https://travis-ci.org/ConstrainedLasso.jl?branch=master

[codecov-img]: https://codecov.io/gh/Hua-Zhou/ConstrainedLasso.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/Hua-Zhou/ConstrainedLasso.jl