# Real Data Example 1 
## Global Warming data - Section 6.1

Here we estimate isotonic regression and constrained lasso solution path.

```@setup warming
using ConstrainedLasso 
using DataFrames
using Base.Test
include("monreg.jl")
```



```@example 1
## load & organize data
# load data 
warming = readtable("warming.csv", header=true)[1]
# extract year & response
year = warming[:, 1]
y    = warming[:, 2]
# extract dimensions
n = p = size(y, 1)
X = eye(n)

## estimate models with monotonicity constraints
## isotonic regression
monoreg, = monreg(y)

## constrained lasso solution path 
# model set up: inequality constraints
A = [eye(p-1) zeros(p-1, 1)] - [zeros(p-1, 1) eye(p-1)]
m2 = size(A, 1)
b = zeros(m2)
# estimate constrained lasso solution path
β̂path, ρpath, = lsq_classopath(X, y; Aineq = A, bineq = b)

## compare estimates
@show maximum(abs.(monoreg - β̂path[:, end]))

```
```@example 1
## graph estimates 
using Plots; pyplot(); using LaTeXStrings; # hide
scatter(year, y, label="Observed Data", markerstrokecolor="darkblue", 
        markercolor="white")
scatter!(year, β̂path[:, end], label=L"Classopath $(\rho=0)$", 
        markerstrokecolor="black", marker=:rect, markercolor="white")
scatter!(year, monoreg, label="Isotonic Regression", marker=:x,
        markercolor="red", markersize=2)
xaxis!("Year") 
yaxis!("Temperature anomalies")
title!("Global Warming Data")
savefig("betapath.svg") # hide
```
![](warming.svg)