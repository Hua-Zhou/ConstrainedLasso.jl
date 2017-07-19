# ConstrainedLasso


**ConstrainedLasso** package estimates the following constrained lasso problem:


<center><a href="https://www.codecogs.com/eqnedit.php?latex=\text{minimize}&space;\hspace{1em}&space;\frac&space;12&space;||\boldsymbol{y}-\boldsymbol{X\beta}||^2_2&space;&plus;&space;\rho||\boldsymbol{\beta}||_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{minimize}&space;\hspace{1em}&space;\frac&space;12&space;||\boldsymbol{y}-\boldsymbol{X\beta}||^2_2&space;&plus;&space;\rho||\boldsymbol{\beta}||_1" title="\text{minimize} \hspace{1em} \frac 12 ||\boldsymbol{y}-\boldsymbol{X\beta}||^2_2 + \rho||\boldsymbol{\beta}||_1" /></a>
<a href="https://www.codecogs.com/eqnedit.php?latex=\text{&space;subject&space;to}&space;\hspace{0.5em}&space;\boldsymbol{A\beta}=\boldsymbol{b}&space;\text{&space;and&space;}&space;\boldsymbol{C\beta}&space;\leq&space;\boldsymbol{d}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{&space;subject&space;to}&space;\hspace{0.5em}&space;\boldsymbol{A\beta}=\boldsymbol{b}&space;\text{&space;and&space;}&space;\boldsymbol{C\beta}&space;\leq&space;\boldsymbol{d}" title="\text{ subject to} \hspace{0.5em} \boldsymbol{A\beta}=\boldsymbol{b} \text{ and } \boldsymbol{C\beta} \leq \boldsymbol{d}" /></a></center>


## Installation 

Within Julia, use the package manager to install **ConstainedLasso**:

```{julia}
Pkg.clone("https://github.com/Hua-Zhou/ConstrainedLasso.git")
```

This package supports Julia v0.6.