
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


```julia
using ConstrainedLasso 
using ECOS
```

First we load and organize the data. 


```julia
warming = readcsv("misc/warming.csv", header=true)[1]
year = warming[:, 1]
y    = warming[:, 2]
```




    166-element Array{Float64,1}:
     -0.375
     -0.223
     -0.224
     -0.271
     -0.246
     -0.271
     -0.352
     -0.46 
     -0.466
     -0.286
     -0.346
     -0.409
     -0.522
      ⋮    
      0.45 
      0.544
      0.505
      0.493
      0.395
      0.506
      0.559
      0.422
      0.47 
      0.499
      0.567
      0.746




```julia
n = p = size(y, 1)
X = eye(n)
```




    166×166 Array{Float64,2}:
     1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  …  0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  …  0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  …  0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     ⋮                        ⋮              ⋱       ⋮                        ⋮  
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  …  0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     1.0  0.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  …  0.0  1.0  0.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  1.0  0.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  1.0  0.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  1.0  0.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     0.0  0.0  0.0  0.0  0.0  1.0  0.0
     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  …  0.0  0.0  0.0  0.0  0.0  0.0  1.0



Now we define inequality constraints as specified earlier. 


```julia
C = [eye(p-1) zeros(p-1, 1)] - [zeros(p-1, 1) eye(p-1)]
```




    165×166 Array{Float64,2}:
     1.0  -1.0   0.0   0.0   0.0   0.0  …   0.0   0.0   0.0   0.0   0.0   0.0
     0.0   1.0  -1.0   0.0   0.0   0.0      0.0   0.0   0.0   0.0   0.0   0.0
     0.0   0.0   1.0  -1.0   0.0   0.0      0.0   0.0   0.0   0.0   0.0   0.0
     0.0   0.0   0.0   1.0  -1.0   0.0      0.0   0.0   0.0   0.0   0.0   0.0
     0.0   0.0   0.0   0.0   1.0  -1.0      0.0   0.0   0.0   0.0   0.0   0.0
     0.0   0.0   0.0   0.0   0.0   1.0  …   0.0   0.0   0.0   0.0   0.0   0.0
     0.0   0.0   0.0   0.0   0.0   0.0      0.0   0.0   0.0   0.0   0.0   0.0
     0.0   0.0   0.0   0.0   0.0   0.0      0.0   0.0   0.0   0.0   0.0   0.0
     0.0   0.0   0.0   0.0   0.0   0.0      0.0   0.0   0.0   0.0   0.0   0.0
     0.0   0.0   0.0   0.0   0.0   0.0      0.0   0.0   0.0   0.0   0.0   0.0
     0.0   0.0   0.0   0.0   0.0   0.0  …   0.0   0.0   0.0   0.0   0.0   0.0
     0.0   0.0   0.0   0.0   0.0   0.0      0.0   0.0   0.0   0.0   0.0   0.0
     0.0   0.0   0.0   0.0   0.0   0.0      0.0   0.0   0.0   0.0   0.0   0.0
     ⋮                             ⋮    ⋱   ⋮                             ⋮  
     0.0   0.0   0.0   0.0   0.0   0.0      0.0   0.0   0.0   0.0   0.0   0.0
     0.0   0.0   0.0   0.0   0.0   0.0      0.0   0.0   0.0   0.0   0.0   0.0
     0.0   0.0   0.0   0.0   0.0   0.0  …   0.0   0.0   0.0   0.0   0.0   0.0
     0.0   0.0   0.0   0.0   0.0   0.0      0.0   0.0   0.0   0.0   0.0   0.0
     0.0   0.0   0.0   0.0   0.0   0.0      0.0   0.0   0.0   0.0   0.0   0.0
     0.0   0.0   0.0   0.0   0.0   0.0      0.0   0.0   0.0   0.0   0.0   0.0
     0.0   0.0   0.0   0.0   0.0   0.0     -1.0   0.0   0.0   0.0   0.0   0.0
     0.0   0.0   0.0   0.0   0.0   0.0  …   1.0  -1.0   0.0   0.0   0.0   0.0
     0.0   0.0   0.0   0.0   0.0   0.0      0.0   1.0  -1.0   0.0   0.0   0.0
     0.0   0.0   0.0   0.0   0.0   0.0      0.0   0.0   1.0  -1.0   0.0   0.0
     0.0   0.0   0.0   0.0   0.0   0.0      0.0   0.0   0.0   1.0  -1.0   0.0
     0.0   0.0   0.0   0.0   0.0   0.0      0.0   0.0   0.0   0.0   1.0  -1.0




```julia
d = zeros(size(C, 1))
```




    165-element Array{Float64,1}:
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     ⋮  
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0



Then we estimate constrained lasso solution path. Here we use `ECOS` solver rather than the default `SCS` solver. 


```julia
β̂path, ρpath, = lsq_classopath(X, y; Aineq = C, bineq = d, solver = ECOSSolver(verbose=0, maxit=1e8)); 
```


```julia
β̂path
```




    166×198 Array{Float64,2}:
     0.0  0.0  0.0  0.0  0.0  0.0       …  -0.323225  -0.366     -0.375   
     0.0  0.0  0.0  0.0  0.0  0.0          -0.294082  -0.336857  -0.345857
     0.0  0.0  0.0  0.0  0.0  0.0          -0.294082  -0.336857  -0.345857
     0.0  0.0  0.0  0.0  0.0  0.0          -0.294082  -0.336857  -0.345857
     0.0  0.0  0.0  0.0  0.0  0.0          -0.294082  -0.336857  -0.345857
     0.0  0.0  0.0  0.0  0.0  0.0       …  -0.294082  -0.336857  -0.345857
     0.0  0.0  0.0  0.0  0.0  0.0          -0.294082  -0.336857  -0.345857
     0.0  0.0  0.0  0.0  0.0  0.0          -0.294082  -0.336857  -0.345857
     0.0  0.0  0.0  0.0  0.0  0.0          -0.294082  -0.336857  -0.345857
     0.0  0.0  0.0  0.0  0.0  0.0          -0.294082  -0.336857  -0.345857
     0.0  0.0  0.0  0.0  0.0  0.0       …  -0.294082  -0.336857  -0.345857
     0.0  0.0  0.0  0.0  0.0  0.0          -0.294082  -0.336857  -0.345857
     0.0  0.0  0.0  0.0  0.0  0.0          -0.294082  -0.336857  -0.345857
     ⋮                        ⋮         ⋱   ⋮                             
     0.0  0.0  0.0  0.0  0.0  0.0           0.432796   0.475571   0.484571
     0.0  0.0  0.0  0.0  0.0  0.0       …   0.432796   0.475571   0.484571
     0.0  0.0  0.0  0.0  0.0  0.0           0.432796   0.475571   0.484571
     0.0  0.0  0.0  0.0  0.0  0.0           0.432796   0.475571   0.484571
     0.0  0.0  0.0  0.0  0.0  0.0           0.432796   0.475571   0.484571
     0.0  0.0  0.0  0.0  0.0  0.0           0.437475   0.48025    0.48925 
     0.0  0.0  0.0  0.0  0.0  0.0       …   0.437475   0.48025    0.48925 
     0.0  0.0  0.0  0.0  0.0  0.0           0.437475   0.48025    0.48925 
     0.0  0.0  0.0  0.0  0.0  0.0           0.437475   0.48025    0.48925 
     0.0  0.0  0.0  0.0  0.0  0.0           0.447225   0.49       0.499   
     0.0  0.0  0.0  0.0  0.0  0.0           0.515225   0.558      0.567   
     0.0  0.0  0.0  0.0  0.0  0.011639  …   0.699138   0.737854   0.746   



In this formulation, isotonic regression is a special case of the constrained lasso with ``\rho=0.``
Below, `monoreg` is coefficient estimates obtained using isotonic regression. 


```julia
monoreg = readdlm("misc/monoreg.txt")
```




    166×1 Array{Float64,2}:
     -0.375   
     -0.345857
     -0.345857
     -0.345857
     -0.345857
     -0.345857
     -0.345857
     -0.345857
     -0.345857
     -0.345857
     -0.345857
     -0.345857
     -0.345857
      ⋮       
      0.484571
      0.484571
      0.484571
      0.484571
      0.484571
      0.48925 
      0.48925 
      0.48925 
      0.48925 
      0.499   
      0.567   
      0.746   



Now let's compare estimates by obtaining the largest absolute difference between isotonic regression constrained lasso fit. 


```julia
maximum(abs.(monoreg - β̂path[:, end]))
```




    1.2212453270876722e-15



Below is a figure that plots the constrained lasso fit at $\rho = 0$ with the estimates using isotonic regression.


```julia
using Plots; pyplot(); 
scatter(year, y, label="Observed Data", markerstrokecolor="darkblue", 
        markercolor="white")
scatter!(year, β̂path[:, end], label="Classopath (ρ=0)", 
        markerstrokecolor="black", marker=:rect, markercolor="white")
scatter!(year, monoreg, label="Isotonic Regression", marker=:x,
        markercolor="red", markersize=2)
xaxis!("Year") 
yaxis!("Temperature anomalies")
title!("Global Warming Data")
```

![](misc/warming.svg)




*Follow this [link](https://github.com/Hua-Zhou/ConstrainedLasso.jl/blob/master/docs/src/demo/warming.ipynb) to access the .ipynb file of this page.*