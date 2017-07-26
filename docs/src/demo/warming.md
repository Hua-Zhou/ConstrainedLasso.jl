# Global Warming Data  

Here we consider the annual data on temperature anomalies. As has been previously noted in the literature on isotonic regression, in general temperature appears to increase monotonically over the time period of 1850 to 2015 ([Wu et al., 2001](../references.md#8); [Tibshirani et al., 2011](../references.md#5)). This monotonicity can be imposed on the coeffcient estimates using the constrained lasso with the inequality constraint matrix:

```math
\begin{split}
& \text{minimize} \hspace{1em} \frac 12||\boldsymbol{y}-\boldsymbol{X\beta}||^2_2 + \rho||\beta||_1 \\
& \text{ subject to} \hspace{1em} \boldsymbol{C\beta} \leq \boldsymbol{d} 
\end{split}
```
where 

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
using Mosek 
```
First we load and organize the data. 

```@example warming
warming = readcsv("data/warming.csv", header=true)[1]
year = warming[:, 1]
y    = warming[:, 2]
```
```@example warming 
n = p = size(y, 1)
X = eye(n)
```
Now we define inequality constraints as specified earlier. 

```@example warming
C = [eye(p-1) zeros(p-1, 1)] - [zeros(p-1, 1) eye(p-1)]
```

```@example warming  
d = zeros(size(C, 1))
```
Then we estimate constrained lasso solution path. Here we use `Mosek` solver rather than the default `SCS` solver. 

```@example warming 
logging(DevNull, ConstrainedLasso, :lsq_classopath, kind=:warn) # hide 
β̂path, ρpath, = lsq_classopath(X, y; Aineq = C, bineq = d, solver = MosekSolver(MSK_IPAR_BI_MAX_ITERATIONS=10e8)); 
β̂path
```
In this formulation, isotonic regression is a special case of the constrained lasso with ``\rho=0.``
Below, `monoreg` is coefficient estimates obtained using isotonic regression. 

```@example warming 
monoreg = readdlm("data/monoreg.txt")
```
Now let's compare estimates by obtaining the largest absolute difference between isotonic regression constrained lasso fit. 

```@example warming 
maximum(abs.(monoreg - β̂path[:, end]))
```
Below is a figure that plots the constrained lasso fit at $\rho = 0$ with the estimates using isotonic regression.

```@example warming 
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
