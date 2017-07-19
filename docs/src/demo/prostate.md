
# Example 

## Unconstrained lasso using constrained lasso solution path

This demonstration solves a regular, unconstrained lasso problem using
the constrained lasso solution path (`lsq_classopath.jl`) and compares to two other methods.

```@setup lasso
using ConstrainedLasso
using DataFrames
```

```@example 1
## load data
prostate = readtable("data/prostate.csv")

## organize data
# combine predictors into data matrix
X= prostate[:, [:lcavol, :lweight, :age, :lbph, :svi, :lcp, :gleason, :pgg45]]
# demean predictors
Xz = Array{Float64}(X)
for i in 1:size(Xz,2)
    Xz[:, i] -= mean(Xz[:, i])
    Xz[:, i] /= std(Xz[:, i])
end
# define response
y = Vector(prostate[:lpsa])
# extract dimensions
n, p = size(Xz)

## solve using lasso solution path algorithm
βpath, ρpath, = lsq_classopath(Xz, y);
@show βpath
@show ρpath

## plot solution path 
using Plots; using LaTeXStrings; pyplot(); # hide 
labels = ["lcavol" "lweight" "age" "lbph" "svi" "lcp" "gleason" "pgg45"]
colors = [:green :orange :black :purple :red :grey :brown :blue] 
plot(ρpath, βpath', xaxis = (L"$\rho$", (minimum(ρpath),
      maximum(ρpath))), yaxis = (L"$\beta(\rho$)"), label=labels, color=colors)
savefig("betapath.svg") # hide
```
![](betapath.svg)
