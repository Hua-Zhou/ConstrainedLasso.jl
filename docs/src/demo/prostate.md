
# Prostate Data  

## Unconstrained lasso

This demonstration solves a regular, unconstrained lasso problem using
the constrained lasso solution path (`lsq_classopath.jl`) and compares to other method.

```@setup lasso
using ConstrainedLasso
using Mosek 
```

```@example lasso
## load data
prostate = readcsv("data/prostate.csv", header=true)
tmp = []

## organize data
# combine predictors into data matrix
labels = ["lcavol" "lweight" "age" "lbph" "svi" "lcp" "gleason" "pgg45"]
for i in labels
    push!(tmp, find(x -> x == i, prostate[2])[1])
end
Xz = Array{Float64}(prostate[1][:, tmp])
# demean predictors
for i in 1:size(Xz,2)
    Xz[:, i] -= mean(Xz[:, i])
    Xz[:, i] /= std(Xz[:, i])
end
# define response
y =	Array{Float64}(prostate[1][:,end-1])
# extract dimensions
n, p = size(Xz)

## solve using lasso solution path algorithm
logging(DevNull, ConstrainedLasso, :lsq_classopath, kind=:warn) # hide 
solver = MosekSolver(MSK_IPAR_BI_MAX_ITERATIONS=10e8)
βpath, ρpath, = lsq_classopath(Xz, y; solver=solver)
nothing # hide
```

```@example lasso 
## plot solution path 
using Plots; pyplot(); # hide 
colors = [:green :orange :black :purple :red :grey :brown :blue] 
plot(ρpath, βpath', xaxis = ("ρ", (minimum(ρpath),
      maximum(ρpath))), yaxis = ("β̂(ρ)"), label=labels, color=colors)
title!("Prostrate Data: Solution Path via Constrained Lasso")
savefig("prostate.svg"); nothing # hide
```
![](prostate.svg)

Following is the model fit using `GLMNet` package. 

```@example lasso
using GLMNet; 
logging(DevNull, GLMNet, :glmnet, kind=:warn) # hide 
path = glmnet(Xz, y)
plot(log.(path.lambda), flipdim(path.betas', 1), color=colors, label=labels, 
		xaxis=("log(λ)", :flip), yaxis= ("β̂(λ)"))
savefig("prostate2.svg"); nothing # hide
```
![](prostate2.svg)