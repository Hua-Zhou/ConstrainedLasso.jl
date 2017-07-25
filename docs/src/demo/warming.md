# Global Warming Data  

Here we consider the annual data on temperature anomalies. As has been previously noted in the literature on isotonic regression, in general temperature appears to increase monotonically over the time period of 1850 to 2015 ([Wu et al., 2001](../references.md); [Tibshirani et al., 2011](../references.md)). This monotonicity can be imposed on the coeffcient estimates using the constrained lasso with the inequality constraint matrix:

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
```
First we load and organize the data. 

```@example warming
warming = readcsv("data/warming.csv", header=true)[1]
year = warming[:, 1]
y    = warming[:, 2]
n = p = size(y, 1)
X = eye(n)
nothing # hide 
```
Now we define inequality constraints as specified earlier. 

```@example warming
A = [eye(p-1) zeros(p-1, 1)] - [zeros(p-1, 1) eye(p-1)]
m2 = size(A, 1)
b = zeros(m2)
nothing # hide 
```
Then we estimate constrained lasso solution path.

```@example warming 
logging(DevNull, ConstrainedLasso, :lsq_classopath, kind=:warn) # hide 
β̂path, ρpath, = lsq_classopath(X, y; Aineq = A, bineq = b) 
nothing # hide
```
In this formulation, isotonic regression is a special case of the constrained lasso with ``\rho=0.``
Below, `monoreg` is coefficient estimates obtained using isotonic regression. 

```@example warming 
monoreg = readdlm("data/monoreg.txt")
nothing # hide 
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
