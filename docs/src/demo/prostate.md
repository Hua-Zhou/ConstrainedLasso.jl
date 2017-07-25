
# Prostate Data  

This demonstration solves a regular, unconstrained lasso problem using
the constrained lasso solution path (`lsq_classopath.jl`).

```@setup lasso
using ConstrainedLasso 
```
The `prostate` data come from a study that examined the correlation between the level of prostate specific antigen and a number of clinical measures in men who were about to receive a radical prostatectomy. ([Stamey et al. (1989)](../references.md))


Let's load and organize the `prostate` data. Since we are interested in the following variables as predictors, we extract them and create a design matrix `Xz`:

* `lcavol` : log(cancer volume)
* `lweight`: log(prostate weight)
* `age`    : age
* `lbph`   : log(benign prostatic hyperplasia amount)
* `svi`    : seminal vesicle invasion
* `lcp`    : log(capsular penetration)
* `gleason`: Gleason score
* `pgg45`  : percentage Gleason scores 4 or 5

The response variable is `lpsa`, which is log(prostate specific antigen). 

```@example lasso
prostate = readcsv("data/prostate.csv", header=true)
tmp = []
labels = ["lcavol" "lweight" "age" "lbph" "svi" "lcp" "gleason" "pgg45"]
for i in labels
    push!(tmp, find(x -> x == i, prostate[2])[1])
end
Xz = Array{Float64}(prostate[1][:, tmp])
y = Array{Float64}(prostate[1][:, end-1])
nothing # hide
```
First we standardize the data by subtracting its mean and dividing by its standard deviation. 

```@example lasso
for i in 1:size(Xz,2)
    Xz[:, i] -= mean(Xz[:, i])
    Xz[:, i] /= std(Xz[:, i])
end
n, p = size(Xz)
nothing # hide
```
Now we solve the problem using solution path algorithm. 

```@example lasso 
logging(DevNull, ConstrainedLasso, :lsq_classopath, kind=:warn) # hide 
βpath, ρpath, = lsq_classopath(Xz, y)
nothing # hide
```
We plot the solution path below. 

```@example lasso 
using Plots; pyplot(); 
colors = [:green :orange :black :purple :red :grey :brown :blue] 
plot(ρpath, βpath', xaxis = ("ρ", (minimum(ρpath),
      maximum(ρpath))), yaxis = ("β̂(ρ)"), label=labels, color=colors)
title!("Prostrate Data: Solution Path via Constrained Lasso")
savefig("prostate.svg"); nothing # hide
```
![](prostate.svg)

Below, we solve the same problem using `GLMNet.jl` package. 

```@example lasso
using GLMNet; 
logging(DevNull, GLMNet, :glmnet, kind=:warn) # hide 
path = glmnet(Xz, y)
plot(log.(path.lambda), flipdim(path.betas', 1), color=colors, label=labels, 
		xaxis=("log(λ)", :flip), yaxis= ("β̂(λ)"))
savefig("prostate2.svg"); nothing # hide
```
![](prostate2.svg)

