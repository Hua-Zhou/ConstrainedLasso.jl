# Real Data Example 2
## Brain Tumor Data - Section 6.2

Here we estimate a generalized lasso model (sparse fused lasso) via the constrained lasso. 

```@setup tumor 
using ConstrainedLasso
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
α̂path, ρpath, = lsq_classopath(X̃, ỹ);
@show ρpath
@show α̂path[:, end]

# transform back to beta
β̂path = Base.LinAlg.BLAS.ger!(1.0, vec(V2 * ((1 / dot(XV2, XV2)) * 
		At_mul_B(XV2, y))), ones(size(ρpath)), (eye(size(V2, 1)) - 
		V2 * ((1 / dot(XV2, XV2)) * At_mul_B(XV2, X))) * Dplus * α̂path )
## plot solution path
# constrained lasso solution path
using Plots; pyplot(); using LaTeXStrings; # hide
plot(ρpath, β̂path', label="", xaxis = (L"$\rho$", (minimum(ρpath),
      maximum(ρpath))), yaxis = (L"$\widehat{\beta}(\rho$)"), width=0.5)
title!("Brain Tumor Data: Solution Path via Constrained Lasso")
savefig("tumor1.svg") # hide
nothing # hide 
```
![](tumor1.svg)


```@example tumor
## plot generalized lasso solution path (from genlasso R package)
plot(lambda_path, beta_path_fused', label="", xaxis = (L"$\lambda$", (minimum(lambda_path),
      maximum(lambda_path))), yaxis = (L"$\widehat{\beta}(\lambda$)"), width=0.5)
title!("Brain Tumor Data: Generalized Lasso Solution Path")
savefig("tumor2.svg") # hide
nothing # hide 
```
![](tumor2.svg)

```@example tumor
# compare estimates at common values of rho 
```