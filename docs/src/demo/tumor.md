# Brain Tumor Data

Here we estimate a generalized lasso model (sparse fused lasso) via the constrained lasso. 

In this example, we use a version of the comparative genomic hybridization (CGH) data from [Bredel et al. (2005)](../references.md) that was modified and studied by [Tibshirani and Wang (2008)](../references.md#6)

The dataset here contains CGH measurements from 2 glioblastoma multiforme (GBM) brain tumors. Tibshirani and Wang (2008) proposed using the sparse fused lasso to approximate the CGH signal by a sparse, piecewise constant function in order to determine the areas with non-zero values, as positive (negative) CGH values correspond to possible gains (losses). The sparse fused lasso (Tibshirani et al., 2005) is given by

```math
\begin{split}
\text{minimize} \hspace{1em} \frac 12 ||\boldsymbol{y}-\boldsymbol{\beta}||_2^2 + \rho_1||\boldsymbol{\beta}||_1 + \rho_2\sum_{j=2}^p |\beta_j - \beta_{j-1}| \hspace{5em} (1)
\end{split}
```
The sparse fused lasso is a special case of the generalized lasso with the penalty matrix. Therefore, the problem ``(1)`` is equivalent to the following: 

```math 
\begin{split} 
\text{minimize} \hspace{1em} \frac 12 ||\boldsymbol{y}-\boldsymbol{X\beta}||_2^2 + \rho ||\boldsymbol{D\beta}||_1 \hspace{5em} (2)
\end{split}
```
where 

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
  & 	&.          & & \ddots & & \\
  &		&		     &      &       & 1 & \\
  &		&		&		 &			    &  & 1\\  
\end{pmatrix} \in \mathbb{R}^{(2P-1)\times p}.

```
As discussed in [Gaines, B.R. and Zhou, H., (2016)](../references.md), the sparse fused lasso can be reformulated and solved as a constrained lasso problem. The generalized lasso problem ``(2)`` is equivalent to 

```math 
\begin{split}
& \text{minimize} \hspace{1em} \frac 12 ||\tilde{\boldsymbol{y}} -\widetilde{\boldsymbol{X}}\boldsymbol{\alpha}||_2^2 + \rho||\boldsymbol{\alpha}||_1 \hspace{5em} (3) \\
& \text{subject to} \hspace{1em} \boldsymbol{U}^T_2\boldsymbol{\alpha} = \boldsymbol{0}
\end{split}
```
where $\widetilde{\boldsymbol{y}} = (\boldsymbol{I}-\boldsymbol{P}_{\boldsymbol{XV}_2})\boldsymbol{y},  \hspace{0.5em} \widetilde{\boldsymbol{X}} = (\boldsymbol{I}-\boldsymbol{P}_{\boldsymbol{XV}_2})\boldsymbol{XD}^+$. Note $D^+$ is the Moore-Penrose inverse of the matrix $\boldsymbol{D}.$ and $U_2, V_2$ are obtained from singular value decomposition (SVD) of ``\boldsymbol{D}`` matrix. Then, the solution path ``
\widehat{\boldsymbol{\alpha}}(\rho)`` can be translated back to that of the original generalized lasso problem via 

```math 
\hat{\boldsymbol{\beta}}(\rho) = [I-\boldsymbol{V}_2(\boldsymbol{V}_2^T\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{V}_2)^-\boldsymbol{V}_2^T\boldsymbol{X}^T\boldsymbol{X}]\boldsymbol{D}^+\hat{\boldsymbol{\alpha}(\rho) + \boldsymbol{V}_2(\boldsymbol{V}_2^T\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{V}_2)^-\boldsymbol{V}_2^T\boldsymbol{X}^T\boldsymbol{y}
```
where $\boldsymbol{X}^-$ denotes the generalized inverse of a matrix $\boldsymbol{X}$.

Details are found in Section 2 of [[3](../references.md)]. 

```@setup tumor 
using ConstrainedLasso
```
We load and organize the data first. Here, `y` is the response vector. The design matrix `X` is an identity matrix since the objective function in ``(1)`` does not involve `X`. Variables `lambda_path` and `beta_path_fused` are lambda values and estimated beta coefficients, respectively, obtained from `genlasso` package in `R`. 

```@example tumor
y = readdlm("data/y.txt")
n = p = size(y, 1)
X = eye(n)
lambda_path = readdlm("data/lambda_path.txt")
beta_path_fused = readdlm("data/beta_path_fused.txt")[2:end, :]
nothing # hide 
```
First we create a penalty matrix `D`. 

```@example tumor 
D = [eye(p-1) zeros(p-1, 1)] - [zeros(p-1, 1) eye(p-1)]
m = size(D, 1)
nothing # hide 
```
Now we transform the problem to the constrained lasso problem. We do the singular value decomposition on `D` and extract singular values and necessary submatrices.  

```@example tumor 
F = svdfact!(D, thin = false)
singvals = F[:S]
rankD = countnz(F[:S] .> abs(F[:S][1]) * eps(F[:S][1]) * maximum(size(D)))

V1 = F[:V][:, 1:rankD]
V2 = F[:V][:, rankD+1:end]
U1 = F[:U][:, 1:rankD]
U2 = F[:U][:, rankD+1:end]
nothing # hide 
```
Now we calculate the Moore-Penrose inverse of `D`, which is ``D^+`` in ``(3)``, and transform the design matrix by multiplying by ``D^+``. 

```@example tumor 
Dplus = V1 * broadcast(*, U1', 1./F[:S])
XDplus = X * Dplus
nothing # hide 
```
In the following code snippet, `Pxv2` is a projection matrix onto `C(XV2)` and `Mxv2` is the orthogonal projection matrix. Then we obtain the design matrix and response vector in their tilde form as shown in ``(3)``. 

```@example tumor
XV2 = X * V2
Pxv2 = (1 / dot(XV2, XV2)) * A_mul_Bt(XV2, XV2)
Mxv2 = eye(size(XV2, 1)) - Pxv2
ỹ = vec(Mxv2 * y)
X̃ = Mxv2 * XDplus
nothing # hide 
```
We solve the constrained lasso problem and obtain $\hat{\boldsymbol{\alpha}}(\rho)$. 

```@example tumor 
logging(DevNull, ConstrainedLasso, :lsq_classopath, kind=:warn) # hide 
using Mosek; solver = MosekSolver(MSK_IPAR_BI_MAX_ITERATIONS=10e8);
α̂path, ρpath, = lsq_classopath(X̃, ỹ; solver = solver)
nothing # hide 
```
Now we need to transform ``\widehat{\boldsymbol{\alpha}} (\rho)`` back to ``\widehat{\boldsymbol{\beta}} (\rho)`` as seen in (3).

```@example tumor 
# transform back to beta
β̂path = Base.LinAlg.BLAS.ger!(1.0, vec(V2 * ((1 / dot(XV2, XV2)) * 
		At_mul_B(XV2, y))), ones(size(ρpath)), (eye(size(V2, 1)) - 
		V2 * ((1 / dot(XV2, XV2)) * At_mul_B(XV2, X))) * Dplus * α̂path )
		
nothing # hide 
```

We plot the constrained lasso solution path below. 

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
