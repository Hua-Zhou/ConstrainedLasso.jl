# Brain Tumor Data
## Section 6.2

Here we estimate a generalized lasso model (sparse fused lasso) via the constrained lasso. 

The dataset here contains CGH measurements from 2 glioblastoma multiforme (GBM) brain tumors. Tibshirani and Wang (2008) proposed using the sparse fused lasso to approximate the CGH signal by a sparse, piecewise constant function in order to determine the areas with non-zero values, as positive (negative) CGH values correspond to possible gains (losses). The sparse fused lasso (Tibshirani et al., 2005) is given by

```math
\text{minimize} \hspace{1em} \frac 12 ||\boldsymbol{y}-\boldsymbol{\beta}||_2^2 + \rho_1||\boldsymbol{\beta}||_1 + \rho_2\sum_{j=2}^p |\beta_j - \beta_{j-1}|.
```
The sparse fused lasso is a special case of the generalized lasso with the penalty matrix 

```math
\boldsymbol{D} = \begin{pmatrix} 
1 & -1 &     &    	  &       & 	& \\
  & 1  & -1  &    	  &  		&	& \\
  &    &  1  & -1 	  & 		& 	& \\
  &		&		& \ddots & \ddots &  & \\
  &		&		&		 &			& 1 & -1 \\
1 &  &     &    	  &       & 	& \\
  & 1  &   &    	  &  		&	& \\
  &    &  \ddots  &  	  & 		& 	& \\
  &		&		&  &   & 1 & \\
  &		&		&		 &			&  & 1\\  
\end{pmatrix} \in \mathbb{R}^{(2P-1)\times p}.
```

```@setup tumor 
using ConstrainedLasso
using Mosek 
```
```@example tumor
# load data
y = readdlm("data/y.txt")
lambda_path = readdlm("data/lambda_path.txt")
beta_path_fused = readdlm("data/beta_path_fused.txt")[2:end, :]

# organize data
n = p = size(y, 1)
X = eye(n)

## estimate using constraiend lasso solution path algorithm
# model setup
D = [eye(p-1) zeros(p-1, 1)] - [zeros(p-1, 1) eye(p-1)]
m = size(D, 1)

# transform to constrained lasso
# calculate SVD
F = svdfact!(D, thin = false)
# extract singular values
singvals = F[:S]
# determine rank
rankD = countnz(F[:S] .> abs(F[:S][1]) * eps(F[:S][1]) * maximum(size(D)))

# extract submatrices of V and U
V1 = F[:V][:, 1:rankD]
V2 = F[:V][:, rankD+1:end]
U1 = F[:U][:, 1:rankD]
U2 = F[:U][:, rankD+1:end]

# calculate the Moore-Penrose inverse of D
Dplus = V1 * broadcast(*, U1', 1./F[:S])
# transform design matrix
XDplus = X * Dplus

# transform to "tilde" form
XV2 = X * V2
# projection onto C(XV2)
Pxv2 = (1 / dot(XV2, XV2)) * A_mul_Bt(XV2, XV2)
# orthogonal projection matrix
Mxv2 = eye(size(XV2, 1)) - Pxv2
# create "tilde" data
ỹ = vec(Mxv2 * y)
X̃ = Mxv2 * XDplus

# constrained solution path
logging(DevNull, ConstrainedLasso, :lsq_classopath, kind=:warn) # hide 
solver = MosekSolver(MSK_IPAR_BI_MAX_ITERATIONS=10e8)
α̂path, ρpath, = lsq_classopath(X̃, ỹ; solver = solver);
@show ρpath
@show α̂path[:, end]

# transform back to beta
β̂path = Base.LinAlg.BLAS.ger!(1.0, vec(V2 * ((1 / dot(XV2, XV2)) * 
		At_mul_B(XV2, y))), ones(size(ρpath)), (eye(size(V2, 1)) - 
		V2 * ((1 / dot(XV2, XV2)) * At_mul_B(XV2, X))) * Dplus * α̂path )
		
nothing # hide 
```

Now, let's plot the constrained lasso solution path. 

```@example tumor 
using Plots; pyplot(); # hide
plot(ρpath, β̂path', label="", xaxis = ("ρ", (minimum(ρpath),
      maximum(ρpath))), yaxis = ("β̂(ρ)"), width=0.5)
title!("Brain Tumor Data: Solution Path via Constrained Lasso")
savefig("tumor1.svg"); nothing # hide 
```

![](tumor1.svg)


Compare the above figure with the following. This figure below plots generalized lasso solution path, which was obtained using `genlasso` package in R. 

```@example tumor
plot(lambda_path, beta_path_fused', label="", xaxis = ("λ", (minimum(lambda_path),
      maximum(lambda_path))), yaxis = ("β̂(λ)"), width=0.5)
title!("Brain Tumor Data: Generalized Lasso Solution Path")
savefig("tumor2.svg"); nothing # hide 
```

![](tumor2.svg)

Now we extract common values of ``\rho`` and compare estimates at those values. 

```@example tumor 
sameρ = intersect(round.(ρpath, 4), round.(lambda_path, 4))
sameρ_err = []
for i in eachindex(sameρ)
 curρ = sameρ[i]
 idx1 = findmin(abs.(ρpath - curρ))[2]
 idx2 = findmin(abs.(lambda_path - curρ))[2]
 push!(sameρ_err, maximum(abs.(β̂path[:, idx1] - beta_path_fused[:, idx2])))
end
nothing # hide 
```

Below are the mean, median, and maximum of the errors between estimated coefficients at common ``\rho`` values. 

```@example tumor
println([mean(sameρ_err); median(sameρ_err); maximum(sameρ_err)])
```
