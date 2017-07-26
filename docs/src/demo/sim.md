# Path algorithm  


## Sum-to-zero constraint 

In this example, we will solve a problem defined by 

```math 
\begin{split}
& \text{minimize} \hspace{1em} \frac 12||\boldsymbol{y}-\boldsymbol{X\beta}||^2_2 + \rho||\boldsymbol{\beta}||_1  \\
& \text{subject to} \hspace{1em} \sum_j \beta_j = 0
\end{split}
```
Note that we can re-write the constraint as 
$\boldsymbol{A\beta} = \boldsymbol{b}$

where 

```math
\boldsymbol{A} = \begin{pmatrix} 1 & 1 & \cdots & 1 \end{pmatrix} \text{ and } \boldsymbol{b} = 0.
```

First let's generate the predictor matrix `X` and response vector `y`. To do so, we need a true parameter vector `β` whose sum equals to 0. Note `n` is the number of observations `n` and `p` is the number of predictors. 

```@example sim1
n, p = 50, 100  
β = zeros(p)
β[1:round(Int, p / 4)] = 0
β[(round(Int, p / 4) + 1):round(Int, p / 2)] = 1
β[(round(Int, p / 2) + 1):round(Int, 3p / 4)] = 0
β[(round(Int, 3p / 4) + 1):p] = -1
srand(41)
X = randn(n, p)
```
```@example sim1
y = X * β + randn(n)
```
Since the problem has equality constraints only, we define the constraints as below. 

```@example sim1
beq = [0]
Aeq = ones(1, p)
```
Now we are ready to obtain the solution path using the path algorithm. By default, we use the solver SCS. 

```@example sim1
using ConstrainedLasso
β̂path1, ρpath1, objpath, = lsq_classopath(X, y; Aeq = Aeq, beq = beq);
β̂path1
```
Let's see if sums of coefficients at all ``\rho`` values are approximately 0. 

```@example sim1
all(abs.(sum(β̂path1, 1)) .< 1e-6)
```
We plot the solution path below. 

```@example sim1
using Plots; pyplot();
plot(ρpath1, β̂path1', label="", xaxis = ("ρ", (minimum(ρpath1),
      maximum(ρpath1))), yaxis = ("β̂(ρ)"), width=0.5) 
title!("Simulation 1: Solution Path via Constrained Lasso") 
savefig("sumtozero.svg"); nothing # hide 
```

![](sumtozero.svg)


## Non-negativity constraint 

In this example, the problem is defined by 

```math 
\begin{split}
& \text{minimize} \hspace{1em} \frac 12||\boldsymbol{y}-\boldsymbol{X\beta}||^2_2 + \rho||\boldsymbol{\beta}||_1  \\
& \text{subject to} \hspace{1em} \beta_j \geq 0 \forall j
\end{split}
```

We can re-write the inequality constraint as
$\boldsymbol{C\beta} \leq \boldsymbol{d}$ where 

```math
\boldsymbol{C} = \begin{pmatrix} 
-1 & & & \\
	& -1 & & \\
	&   & \ddots & \\
	& 	& 	& -1
\end{pmatrix}
\text{ and } \boldsymbol{d} = \begin{pmatrix} 0 \\ 0 \\ \vdots \\ 0 \end{pmatrix}
```

First we define a true parameter vector `β` that is sparse with a few non-zero coefficients. Let `n` and `p` be the number of observations and predictors, respectively. 

```@example sim1
n, p = 50, 100   
β = zeros(p)
β[1:10] = 1:10
srand(41)
X = randn(n, p)
```
```@example sim1
y = X * β + randn(n)
```
Now set up the inequality constraint for the problem. 

```@example sim1
bineq = zeros(p)
Aineq = - eye(p)
```
Now we are ready to obtain the solution path using the path algorithm. Here, let's try using different solver `Mosek` for `Convex.jl`. 

```@example sim1
using Mosek; solver = MosekSolver(MSK_IPAR_BI_MAX_ITERATIONS=10e8);
β̂path2, ρpath2, = lsq_classopath(X, y; Aineq = Aineq, bineq = bineq, solver = solver) 
β̂path2
```
We plot the solution path below. 

```@example sim1
plot(ρpath2, β̂path2', label="", xaxis = ("ρ", (minimum(ρpath2),
      maximum(ρpath2))), yaxis = ("β̂(ρ)"), width=0.5) 
title!("Simulation 2: Solution Path via Constrained Lasso") 
savefig("nonneg.svg"); nothing # hide
```

![](nonneg.svg)