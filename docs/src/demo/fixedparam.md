# Optimize at fixed tuning parameter value(s)

`lsq_constrsparsereg.jl` fits constrained lasso

```math
\begin{split}
& \text{minimize} \hspace{1em} \frac 12||\boldsymbol{y}-\boldsymbol{X\beta}||^2_2 + \rho||\beta||_1 \\
& \text{ subject to} \hspace{0.5em} \boldsymbol{A\beta}=\boldsymbol{b} \text{ and } \boldsymbol{C\beta} \leq \boldsymbol{d}
\end{split}
```

at a fixed tuning parameter value $\rho$ or several tuning parameter values provided by user.

### Single tuning parameter value

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

n, p = 100, 20
β = zeros(p)
β[1:round(Int, p / 4)] = 0
β[(round(Int, p / 4) + 1):round(Int, p / 2)] = 1
β[(round(Int, p / 2) + 1):round(Int, 3p / 4)] = 0
β[(round(Int, 3p / 4) + 1):p] = -1
β
```




    20-element Array{Float64,1}:
      0.0
      0.0
      0.0
      0.0
      0.0
      1.0
      1.0
      1.0
      1.0
      1.0
      0.0
      0.0
      0.0
      0.0
      0.0
     -1.0
     -1.0
     -1.0
     -1.0
     -1.0



Next we generate data based on the true parameter `β`.


```julia
srand(123)
X = randn(n, p)
```




    100×20 Array{Float64,2}:
      1.19027     0.376264    0.346589    0.458099   …   0.0523088   2.23365    
      2.04818    -0.405272    1.60431     0.139124      -0.168468    1.29252    
      1.14265     1.33585    -0.0246589  -0.230745      -0.247202   -0.822482   
      0.459416    1.60076    -0.106035    1.35195       -1.66701     0.0134896  
     -0.396679   -1.45789    -1.29118    -0.106316      -1.24891     0.466877   
     -0.664713    0.800589   -0.337985   -0.205883   …  -0.623667    1.32134    
      0.980968    0.895878   -0.177092   -0.612003       0.372048    0.581506   
     -0.0754831  -0.691934    0.57499    -1.39397       -0.633969    1.35917    
      0.273815   -1.50876    -1.37834     1.73135       -0.74986     1.27668    
     -0.194229   -0.754523   -0.867869    2.61556       -2.07352     1.59349    
     -0.339366    0.115622   -0.400076    1.76909    …   0.496242    0.806838   
     -0.843878    0.242595    0.295087    0.240332       0.5764      1.55243    
     -0.888936   -0.223211    0.817696   -1.42384       -0.859119    1.49178    
      ⋮                                              ⋱                          
     -0.733961    0.911747   -0.618047   -0.319891       1.44548    -1.1144     
      0.459398    0.0138789  -1.08527    -0.529198       0.395225    0.822061   
      1.70619     2.2959      0.4024      1.47241    …  -0.260034   -0.746822   
      0.678443    0.934982    0.425372    1.17431        0.780863    0.439673   
      0.28718     2.00606    -1.18929     1.35692       -0.545467   -0.40497    
      1.06816    -0.379291    0.11631     2.48089       -1.04331     1.24328    
     -0.306877    0.20646    -1.34497    -0.584326      -1.8609     -0.383338   
     -1.92021    -0.276028    0.426339    0.38792    …   2.16327    -1.02578    
      1.6696      1.19586    -0.783625    0.718697       1.13162    -1.31358    
     -0.213558   -1.2965      0.648433   -0.289336       0.263283    0.000636658
     -0.163711    0.575279   -0.176555   -0.0457259      0.152164    0.1559     
     -0.902986   -0.166001   -1.27924    -1.31238        0.49458    -0.171711   




```julia
y = X * β + randn(n)
```




    100-element Array{Float64,1}:
     -6.50888  
     -3.00423  
     -2.3809   
      2.50638  
     -2.24753  
     -2.51881  
      3.60092  
     -1.38597  
      0.0562454
      0.787719 
      3.17731  
     -2.1989   
     -1.86609  
      ⋮        
      0.87746  
     -3.68264  
     -3.19285  
     -0.961258 
      0.793834 
      0.140524 
      1.71841  
     -6.31781  
     -1.95472  
      1.4803   
     -2.15804  
     -6.66724  



Since the equality constraint can be written as

```math
\begin{split}
\begin{pmatrix} 1 & 1 & \cdots & 1 \end{pmatrix} \beta = 0,
\end{split}
```

we define the constraint as below.


```julia
beq   = [0.0]
Aeq   = ones(1, p)
```




    1×20 Array{Float64,2}:
     1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  …  1.0  1.0  1.0  1.0  1.0  1.0  1.0



Now we are ready to fit the constrained lasso problem, say at `ρ=10`.


```julia
ρ = 10.0
β̂, = lsq_constrsparsereg(X, y, ρ; Aeq = Aeq, beq = beq);
```


```julia
β̂
```




    20×1 Array{Float64,2}:
      1.06543e-7 
      1.16499e-7 
      0.184567   
     -1.07162e-8 
      6.88319e-8 
      0.891789   
      0.867016   
      0.89869    
      0.830042   
      0.740962   
      3.21933e-8 
     -7.74822e-8 
     -6.04844e-12
     -6.67961e-8 
      0.0155802  
     -0.918892   
     -0.927699   
     -0.874741   
     -1.00308    
     -0.704239   



We see if the sum of estimated $\beta$ coefficients equal to 0.


```julia
@test sum(β̂)≈0.0 atol=1e-5
```




    Test Passed
  



### Multiple tuning parameter values

Define `ρlist` to be a sequence of values from 1 to 10.


```julia
ρlist = 1.0:10.0
```




    1.0:1.0:10.0



Using the same equality constraints, we fit the constrained lasso.


```julia
β̂, = lsq_constrsparsereg(X, y, ρlist; Aeq = Aeq, beq = beq);
```


```julia
β̂
```




    20×10 Array{Float64,2}:
      0.0654744    0.0528563    0.0435276   …   1.36207e-7   1.06543e-7 
      0.0984093    0.0769516    0.0557896       1.55064e-7   1.16499e-7 
      0.256568     0.246248     0.236865        0.19077      0.184567   
     -0.0161896    1.24814e-6   3.09904e-7     -1.14652e-8  -1.07162e-8 
      0.0442985    0.0355132    0.0266877       9.93987e-8   6.88319e-8 
      0.978801     0.965257     0.953956    …   0.900293     0.891789   
      0.962883     0.952699     0.943744        0.880242     0.867016   
      0.910347     0.907963     0.907171        0.900036     0.89869    
      0.968082     0.947724     0.931277        0.845411     0.830042   
      0.936793     0.911403     0.887538        0.76097      0.740962   
     -9.39273e-8  -2.28297e-7  -3.90716e-8  …   4.38155e-8   3.21933e-8 
     -0.0819607   -0.0711052   -0.0585947      -0.00364202  -7.74822e-8 
     -3.72613e-8   2.38111e-7   8.712e-8        6.32877e-9  -6.04844e-12
     -0.0609711   -0.0433998   -0.0252136      -8.37749e-8  -6.67961e-8 
      0.0730144    0.0671585    0.0608885       0.0235996    0.0155802  
     -1.04989     -1.03264     -1.01746     …  -0.931341    -0.918892   
     -1.00933     -0.998772    -0.98755        -0.934456    -0.927699   
     -1.04134     -1.01861     -0.996088       -0.889298    -0.874741   
     -1.14266     -1.12595     -1.11046        -1.01705     -1.00308    
     -0.89232     -0.873294    -0.85207        -0.725531    -0.704239   



Now let's test if coefficients sum to 0 at each parameter value.


```julia
@testset "zero-sum for multiple param values" begin for i in sum(β̂, 1)
  @test i≈0.0 atol=1.0e-5
end
end
```

    Test Summary:                      | Pass  Total
    zero-sum for multiple param values |   10     10






