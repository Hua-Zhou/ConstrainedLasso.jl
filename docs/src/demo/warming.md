# Global Warming Data  
## Section 6.1

Here we consider the annual data on temperature anomalies. As has been previously noted in the literature on isotonic regression, in general temperature appears to increase monotonically over the time period of 1850 to 2015 (Wu et al., 2001; Tibshirani et al., 2011). This monotonicity can be imposed on the coeffcient estimates using the constrained lasso with the inequality constraint matrix:

```math
\boldsymbol{C} = \begin{pmatrix} 
1 & -1 &     &    	  &       & 	& \\
  & 1  & -1  &    	  &  		&	& \\
  &    &  1  & -1 	  & 		& 	& \\
  &		&		& \ddots & \ddots &  & \\
  &		&		&		 &			& 1 & -1 \\
\end{pmatrix}
```
and ``\boldsymbol{d} = \boldsymbol{0}.``


```@setup warming
using ConstrainedLasso 
using DataFrames
using Base.Test
using Mosek 
```

```@example warming
## load & organize data
# load data 
warming = readcsv("data/warming.csv", header=true)[1]
# extract year & response
year = warming[:, 1]
y    = warming[:, 2]
# extract dimensions
n = p = size(y, 1)
X = eye(n)

## constrained lasso solution path 
# model set up: inequality constraints
A = [eye(p-1) zeros(p-1, 1)] - [zeros(p-1, 1) eye(p-1)]
m2 = size(A, 1)
b = zeros(m2)
# estimate constrained lasso solution path
logging(DevNull, ConstrainedLasso, :lsq_classopath, kind=:warn) # hide 
solver = MosekSolver(MSK_IPAR_BI_MAX_ITERATIONS=10e8)
β̂path, ρpath, = lsq_classopath(X, y; Aineq = A, bineq = b, solver = solver) 
nothing # hide
```
In this formulation, isotonic regression is a special case of the constrained lasso with ``\rho=0.``
Below, `monoreg` is trend estimates using isotonic regression. 

```@example warming 
## estimate models with monotonicity constraints # hide
## isotonic regression # hide 
monoreg = readdlm("data/monoreg.txt")
```
Now let's compare estimates. 

```@example warming 
maximum(abs.(monoreg - β̂path[:, end]))
```
```@example warming
## graph estimates 
using Plots; pyplot(); # hide
scatter(year, y, label="Observed Data", markerstrokecolor="darkblue", 
        markercolor="white")
scatter!(year, β̂path[:, end], label="Classopath (ρ=0)", 
        markerstrokecolor="black", marker=:rect, markercolor="white")
scatter!(year, monoreg, label="Isotonic Regression", marker=:x,
        markercolor="red", markersize=2)
xaxis!("Year") 
yaxis!("Temperature anomalies")
title!("Global Warming Data")
savefig("warming.svg"); nothing # hide
```
![](warming.svg)

The above figure plots the constrained lasso fit at $\rho = 0$ with the estimates using isotonic regression. 