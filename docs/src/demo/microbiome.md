# Microbiome Data
## Section 6.3

   Our last real data application with the constrained lasso uses microbiome data. 
   Here the problem is to 

```math 
\begin{align}
& \text{minimize} \hspace{1em} \frac 12||\boldsymbol{y}-\boldsymbol{X\beta}||^2_2 + \rho\Big(||\boldsymbol{\beta}||_1 + \frac{1-\alpha}{2}||\boldsymbol{\beta}||_2^2\Big) \\
& \text{subject to} \hspace{1em} \sum_j \beta_j = 0
\end{align}
```

where ``\alpha = 1``. Hence this problem is reduced to the constrained lasso. 

```@setup micro
using ConstrainedLasso
using Mosek 
```

```@example micro
## load & organize data 
zerosum = readcsv("data/zerosum.csv", header=true)[1]
# extract data 
y = zerosum[:, 1]
X = zerosum[:, 2:end]
# extract dimensions 
n, p = size(X)

## model set-up
solver = MosekSolver(MSK_IPAR_BI_MAX_ITERATIONS=10e8)
# set up equality constraints
Aeq = ones(1, p)
beq = [0]
m1 = size(Aeq, 1)

## constrained Lasso solution path
# estimate solution path
β̂path, ρpath, = lsq_classopath(X, y; Aeq = Aeq, beq = beq, solver = solver)
# scale the tuning parameter to match the zeroSum formulation (which
#	   divides the loss fuction by 2n instead of just 2
newρpath = ρpath ./ n

# @show β̂path[:, end]
# @show ρpath

# calculate L1 norm along path
norm1path = zeros(size(β̂path, 2))
for i in eachindex(norm1path)
    norm1path[i] = norm(β̂path[:, i], 1)
end

nothing # hide 
```
Now, let's plot the solution path. 

```@example micro
using Plots; pyplot(); # hide
plot(norm1path, β̂path', xaxis = ("||β̂||₁"), yaxis=("β̂"), label="")
title!("Microbiome Data: Solution Path via Constrained Lasso")
savefig("micro.svg"); nothing # hide
```
The following figure plots the coefficient estimate solution paths, ``\widehat{\boldsymbol{\beta}}(\rho)``, as a function of ``||\widehat{\boldsymbol{\beta}}(\rho)||_1`` using both the zero-sum regression and the constrained lasso. 

![](micro.svg)

As can be seen in the graphs, the coeffcient estimates are nearly indistinguishable except for some very minor differences, which are a result of the slightly different formulations of the two problems.