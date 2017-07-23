# Simulation Examples 

```@setup sim1
using ConstrainedLasso
using Mosek 
using Plots; pyplot(); # hide
```

## Sum-to-zero constraint 

```@example sim1
n = 50    # no. of observations
p = 100   # no. of predictors

## define true parameter vector
# sum(β) = 0
β = zeros(p)
β[1:round(Int, p / 4)] = 0
β[(round(Int, p / 4) + 1):round(Int, p / 2)] = 1
β[(round(Int, p / 2) + 1):round(Int, 3p / 4)] = 0
β[(round(Int, 3p / 4) + 1):p] = -1

## constraints housekeeping
# equality constraints
Aeq = ones(1, p)
beq = [0]
# no inequality constraints

## generate data
srand(41)
X = randn(n, p)
y = X * β + randn(n)

## estimate path with classopath
solver = MosekSolver(MSK_IPAR_BI_MAX_ITERATIONS=10e8)
β̂path1, ρpath1, objpath, = lsq_classopath(X, y; Aeq = Aeq, beq = beq, solver = solver)
nothing # hide 
```
```@example sim1
all(abs.(sum(β̂path1, 1)) .< 1e-6)
```

```@exmaple sim1
plot(ρpath1, β̂path1', label="", xaxis = ("ρ", (minimum(ρpath1),
      maximum(ρpath1))), yaxis = ("β̂(ρ)"), width=0.5)
title!("Simulation 1: Solution Path via Constrained Lasso")
savefig("sumtozero.svg"); nothing # hide 
```

![](sumtozero.svg)


## Non-negativity constraint 


```@example sim1
n = 50    # no. of observations
p = 100   # no. of predictors

## define true parameter vector
# sparsity with a few non-zero coefficients
β = zeros(p)
β[1:10] = 1:10

## constraints housekeeping
# no equality constraints
# inequality constraints
Aineq = - eye(p)
bineq = zeros(p)

## generate data
srand(41)
X = randn(n, p)
y = X * β + randn(n)

## estimate path with classopath
solver = MosekSolver(MSK_IPAR_BI_MAX_ITERATIONS=10e8)
β̂path2, ρpath2, = lsq_classopath(X, y; Aineq = Aineq, bineq = bineq, solver = solver)
nothing # hide 
```

```@exmaple sim1
plot(ρpath2, β̂path2', label="", xaxis = ("ρ", (minimum(ρpath2),
      maximum(ρpath2))), yaxis = ("β̂(ρ)"), width=0.5)
title!("Simulation 2: Solution Path via Constrained Lasso")
savefig("nonneg.svg"); nothing # hide 
```

![](nonneg.svg)