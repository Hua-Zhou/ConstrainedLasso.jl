# Microbiome Data


   This real data application uses microbiome data [[7](../references.md#7)]. The dataset itself contains information on 160 bacteria genera from 37 patients. The bacteria counts were ``\log_2``-transformed and normalized to have a constant average across samples.

First, let's load and organize data.

```@example micro
zerosum = readcsv("data/zerosum.csv", header=true)[1]
y = zerosum[:, 1]
```
```@example micro 
X = zerosum[:, 2:end]
```

[Altenbuchinger et al.](../references.md#1) demonstrated that a sum-to-zero constraint is useful anytime the normalization of data relative to some reference point results in proportional data, as is often the case in biological applications, since the analysis using the constraint is insensitive to the choice of the reference. [Altenbuchinger et al.](../references.md#1) derived a coordinate descent algorithm for the elastic net with a zero-sum constraint,

```math
\begin{split}
& \text{minimize} \hspace{1em} \frac 12||\boldsymbol{y}-\boldsymbol{X\beta}||^2_2 + \rho\Big(||\boldsymbol{\beta}||_1 + \frac{1-\alpha}{2}||\boldsymbol{\beta}||_2^2\Big) \\
& \text{subject to} \hspace{1em} \sum_j \beta_j = 0
\end{split}
```
but the focus of their analysis corresponds to ``\alpha = 1``. Hence the problem is reduced to the constrained lasso.

We set up the zero-sum constraint.

```@example micro
n, p = size(X)
Aeq = ones(1, p)
beq = [0]
m1 = size(Aeq, 1)
nothing # hide 
```
Now we estimate the constrained lasso solution path using path algorithm.

```@example micro
using ConstrainedLasso
using Mosek
solver = MosekSolver(MSK_IPAR_BI_MAX_ITERATIONS=10e8);
β̂path, ρpath, = lsq_classopath(X, y; Aeq = Aeq, beq = beq, solver = solver)
β̂path
```
Then we calculate `L1` norm of coefficients at each ``\rho``.

```@example micro
norm1path = zeros(size(β̂path, 2))
for i in eachindex(norm1path)
    norm1path[i] = norm(β̂path[:, i], 1)
end
norm1path
```
Now, let's plot the solution path, ``\widehat{\boldsymbol{\beta}}(\rho)``, as a function of ``||\widehat{\boldsymbol{\beta}}(\rho)||_1`` using constrained lasso.

```@example micro
using Plots; pyplot();
plot(norm1path, β̂path', xaxis = ("||β̂||₁"), yaxis=("β̂"), label="")
title!("Microbiome Data: Solution Path via Constrained Lasso")
savefig("micro.svg"); nothing # hide
```

![](micro.svg)
