
# ADMM 

In this section, we apply the alternating direction method of multipliers (ADMM) algorithm to the constrained lasso problem. Below is the ADMM algorithm for solving the constrained lasso. 

> **Algorithm:** ADMM for solving the constrained lasso 
> 
> 1. Initialize $\boldsymbol{\beta}^{(0)} = \boldsymbol{z}^{(0)} = \boldsymbol{\beta}^{0}, \boldsymbol{u}^{(0)} = \boldsymbol{0}, \tau > 0$     
> - *Repeat* the following until *convergence criterion is met*   
> 	 * $\boldsymbol{\beta}^{(t+1)} \leftarrow \text{argmin} \frac 12 ||\boldsymbol{y}-\boldsymbol{X\beta}||_2^2 + \frac{1}{2\tau}||\boldsymbol{\beta} + \boldsymbol{z}^{(t)} + \boldsymbol{u}^{(t)}||_2^2 + \rho||\boldsymbol{\beta}||_1$       
> 	 * $\boldsymbol{z}^{(t+1)} \leftarrow \text{proj}_{\mathcal{C}}(\boldsymbol{\beta}^{(t+1)}+\boldsymbol{u}^{(t)})$     
> 	 * $\boldsymbol{u}^{(t+1)} \leftarrow \boldsymbol{u}^{(t)} + \boldsymbol{\beta}^{(t+1)} + \boldsymbol{z}^{(t+1)}$     


where $\text{proj}_{\mathcal{C}}$ is a projection onto 

```math
\mathcal{C} = \{\boldsymbol{\beta}\in \mathbb{R}^p: \boldsymbol{A\beta}=\boldsymbol{b}, \boldsymbol{C\beta} \leq \boldsymbol{d} \}.
```

## sum-to-zero constraint 

We demonstrate using a sum-to-zero constraint example

```math
\begin{split}
& \text{minimize} \hspace{1em} \frac 12||\boldsymbol{y}-\boldsymbol{X\beta}||^2_2 + \rho||\beta||_1 \\
& \text{ subject to} \hspace{0.5em} \sum_j \beta_j = 0
\end{split}
```

First, let's define a true parameter `β` such that `sum(β) = 0`.


```julia
using ConstrainedLasso, Base.Test

n, p = 50, 100  
β = zeros(p);
β[1:round(Int, p / 4)] = 0
β[(round(Int, p / 4) + 1):round(Int, p / 2)] = 1
β[(round(Int, p / 2) + 1):round(Int, 3p / 4)] = 0
β[(round(Int, 3p / 4) + 1):p] = -1
β
```




    100-element Array{Float64,1}:
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
     -1.0
     -1.0
     -1.0
     -1.0
     -1.0
     -1.0
     -1.0
     -1.0
     -1.0
     -1.0
     -1.0
     -1.0



Next we generate data based on the true parameter β.


```julia
srand(41)
X = randn(n, p)
```




    50×100 Array{Float64,2}:
      1.21212    -0.153889    0.141533  …  -0.458125    0.0951976  -2.14019   
      0.345895    1.30676     1.60944      -0.409901    0.323719    0.989333  
     -1.27859    -1.18894     0.512064      1.80509     1.62606    -1.44251   
      0.230616    2.54741    -0.523533      2.73358     1.07999     0.432834  
     -1.17103    -0.39082     0.441921     -0.179239   -0.158189   -0.640611  
      1.67135     0.0829011   0.964089  …  -0.720038    1.99359    -0.671572  
     -0.614717    2.16204    -0.0602       -0.324456   -0.616887    1.11243   
     -0.810535    0.974719   -0.045405      0.881578    1.29611     0.696869  
     -1.10879    -1.32489    -1.18272       0.579381   -0.971269   -0.687591  
     -0.219752   -0.447897   -0.974186     -0.880804   -0.480702   -1.36887   
      0.0952544  -0.126203   -0.273737  …  -0.264421    0.565684   -0.798719  
      1.4126      0.295896   -0.213161     -1.46343    -1.27144    -0.0589753 
     -0.418407   -0.479389    0.324243      1.96976     0.867659   -1.2999    
      ⋮                                 ⋱                                     
      0.504861   -1.03911    -0.357771      0.815027    0.919037    1.07463   
     -0.820358   -0.955319    0.097768      0.553219    1.56424     0.10535   
      1.39684     1.93183     0.706641  …  -0.0222014   0.987281   -0.0646814 
     -1.55206     0.446778    1.48206      -1.42384    -1.04209     0.0460478 
      0.928527    0.933087   -0.641975     -1.16347    -0.313851   -1.20434   
      0.380879   -0.144713    1.54374      -0.605637    0.408246    0.632131  
     -1.30233    -2.31664     1.51324       0.765034   -0.515553    0.984551  
      1.36747     1.34059    -0.114778  …   0.846682   -0.565511   -0.539113  
     -2.82496    -0.0447351   0.426242     -0.353497   -0.14583    -0.00304009
     -0.847741    1.49306     1.15522       0.637659    1.70818     0.641035  
     -0.22286    -0.43932    -0.373259      0.788337    0.223785   -0.343495  
      1.32145     0.104516   -0.993017     -0.272744   -0.133748    0.968627  




```julia
y = X * β + randn(n)
```




    50-element Array{Float64,1}:
      -9.90585 
      -5.40562 
       5.24289 
      -6.29951 
      -4.9586  
      -6.1342  
      -7.90981 
       2.51009 
      -5.79548 
       1.61355 
      -0.722766
      10.4522  
       4.03935 
       ⋮       
       0.397781
      -2.6661  
       5.36896 
      -3.56537 
      -2.402   
       0.11478 
      -5.39248 
       4.38391 
       0.706801
     -10.1066  
      -1.12558 
      14.2473  



Now we estimate coefficients at fixed tuning parameter value using ADMM alogrithm. 


```julia
ρ = 2.0
β̂admm = lsq_constrsparsereg_admm(X, y, ρ; proj = x -> x - mean(x))
```


```julia
β̂admm
```




    100×1 GLMNet.CompressedPredictorMatrix:
      0.0     
      0.174344
      0.0     
     -0.421288
      0.0     
      0.0     
      0.0     
      0.0     
      0.324233
     -0.15384 
      0.0     
      0.0     
      0.0     
      ⋮       
     -0.397027
     -0.32079 
      0.0     
      0.0     
     -0.868508
     -0.992272
     -0.571755
      0.0     
     -1.16568 
      0.0     
      0.0     
      0.0     



Now let's compare the estimated coefficients with those obtained using quadratic programming. 


```julia
ρ = 2.0 
beq = [0]
Aeq = ones(1, p)
using ECOS; solver=ECOSSolver(verbose=0, maxit=1e8);
β̂, = lsq_constrsparsereg(X, y, ρ; Aeq = Aeq, beq = beq, solver = solver) 
```


```julia
hcat(β̂admm, β̂)
```




    100×2 Array{Float64,2}:
      0.0        1.51451e-8 
      0.174344   0.178717   
      0.0       -1.02944e-9 
     -0.421288  -0.414043   
      0.0       -3.33221e-10
      0.0       -4.14188e-10
      0.0        3.28018e-11
      0.0        8.38037e-11
      0.324233   0.335375   
     -0.15384   -0.157908   
      0.0       -2.44739e-9 
      0.0       -8.79003e-10
      0.0       -8.17787e-10
      ⋮                     
     -0.397027  -0.391635   
     -0.32079   -0.33352    
      0.0       -5.51459e-9 
      0.0        1.09637e-9 
     -0.868508  -0.867639   
     -0.992272  -0.999583   
     -0.571755  -0.577743   
      0.0       -8.93601e-10
     -1.16568   -1.16862    
      0.0        2.29367e-9 
      0.0       -1.52035e-9 
      0.0        2.71896e-9 



## Non-negativity constraint 

Here we look at the non-negativity constraint. First let's generate `X` and `y`.


```julia
n, p = 50, 100   
β = zeros(p)
β[1:10] = 1:10
srand(41)
X = randn(n, p)
```




    50×100 Array{Float64,2}:
      1.21212    -0.153889    0.141533  …  -0.458125    0.0951976  -2.14019   
      0.345895    1.30676     1.60944      -0.409901    0.323719    0.989333  
     -1.27859    -1.18894     0.512064      1.80509     1.62606    -1.44251   
      0.230616    2.54741    -0.523533      2.73358     1.07999     0.432834  
     -1.17103    -0.39082     0.441921     -0.179239   -0.158189   -0.640611  
      1.67135     0.0829011   0.964089  …  -0.720038    1.99359    -0.671572  
     -0.614717    2.16204    -0.0602       -0.324456   -0.616887    1.11243   
     -0.810535    0.974719   -0.045405      0.881578    1.29611     0.696869  
     -1.10879    -1.32489    -1.18272       0.579381   -0.971269   -0.687591  
     -0.219752   -0.447897   -0.974186     -0.880804   -0.480702   -1.36887   
      0.0952544  -0.126203   -0.273737  …  -0.264421    0.565684   -0.798719  
      1.4126      0.295896   -0.213161     -1.46343    -1.27144    -0.0589753 
     -0.418407   -0.479389    0.324243      1.96976     0.867659   -1.2999    
      ⋮                                 ⋱                                     
      0.504861   -1.03911    -0.357771      0.815027    0.919037    1.07463   
     -0.820358   -0.955319    0.097768      0.553219    1.56424     0.10535   
      1.39684     1.93183     0.706641  …  -0.0222014   0.987281   -0.0646814 
     -1.55206     0.446778    1.48206      -1.42384    -1.04209     0.0460478 
      0.928527    0.933087   -0.641975     -1.16347    -0.313851   -1.20434   
      0.380879   -0.144713    1.54374      -0.605637    0.408246    0.632131  
     -1.30233    -2.31664     1.51324       0.765034   -0.515553    0.984551  
      1.36747     1.34059    -0.114778  …   0.846682   -0.565511   -0.539113  
     -2.82496    -0.0447351   0.426242     -0.353497   -0.14583    -0.00304009
     -0.847741    1.49306     1.15522       0.637659    1.70818     0.641035  
     -0.22286    -0.43932    -0.373259      0.788337    0.223785   -0.343495  
      1.32145     0.104516   -0.993017     -0.272744   -0.133748    0.968627  




```julia
y = X * β + randn(n)
```




    50-element Array{Float64,1}:
      12.6173  
      40.3776  
       2.2169  
      27.4631  
      38.592   
       7.82023 
      22.7367  
       7.88475 
      -7.47037 
       0.621035
      -4.91899 
     -14.9363  
       8.26901 
       ⋮       
       7.83882 
      -9.30699 
     -29.7205  
      15.2482  
     -19.1784  
      14.9865  
       2.32728 
      -9.11988 
     -15.3472  
      22.9679  
      -0.997964
      42.6068  




```julia
ρ = 2.0
β̂admm = lsq_constrsparsereg_admm(X, y, ρ; proj = x -> clamp.(x, 0, Inf))
```


```julia
β̂admm
```




    100×1 GLMNet.CompressedPredictorMatrix:
     0.611673  
     2.17111   
     2.65667   
     4.05568   
     4.72435   
     5.87293   
     6.6957    
     8.36528   
     8.61945   
     9.80517   
     1.18896e-6
     0.0       
     0.0       
     ⋮         
     0.100613  
     0.0       
     0.0       
     0.0       
     0.0       
     0.0       
     0.0       
     0.0       
     0.0       
     0.0       
     6.84603e-7
     0.0593872 



Again we compare the estimates with those from quadratic programming. Here we use `ECOS` solver instead of the default `SCS`. 


```julia
ρ = 2.0 
bineq = zeros(p)
Aineq = - eye(p)
using ECOS; solver=ECOSSolver(verbose=0, maxit=1e8);
β̂, = lsq_constrsparsereg(X, y, ρ; Aineq = Aineq, bineq = bineq, solver = solver) 
```


```julia
β̂
```




    100×1 Array{Float64,2}:
      0.610587   
      2.17169    
      2.65765    
      4.05601    
      4.72551    
      5.87414    
      6.69414    
      8.36632    
      8.62049    
      9.80458    
     -1.18573e-10
     -8.86515e-11
     -5.48968e-11
      ⋮          
      0.10151    
      1.08588e-9 
      3.51552e-10
     -4.42556e-11
     -8.01753e-11
      1.40626e-10
     -1.58472e-11
      2.50567e-10
     -4.76544e-11
      3.28495e-10
     -7.49201e-11
      0.0602764  
      

*Follow this [link](https://github.com/Hua-Zhou/ConstrainedLasso.jl/blob/master/docs/src/demo/admm.ipynb) to access the .ipynb file of this page.*