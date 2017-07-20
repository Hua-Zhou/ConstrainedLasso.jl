var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#ConstrainedLasso-1",
    "page": "Home",
    "title": "ConstrainedLasso",
    "category": "section",
    "text": "ConstrainedLasso estimates the following constrained lasso problem, using the approach of Gaines and Zhou (2016).beginalign\n textminimize hspace1em frac 12boldsymboly-boldsymbolXbeta^2_2 + rhobeta_1 \n text subject to hspace05em boldsymbolAbeta=boldsymbolb text and  boldsymbolCbeta leq boldsymbold\nendalignwhere boldsymboly in mathbbR^n is the response vector, boldsymbolXin mathbbR^ntimes p is the design matrix of predictor or covariates, boldsymbolbeta in mathbbR^p is the vector of unknown regression coefficients, and rho geq 0 is a tuning parameter that controls the amount of regularization."
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "Within Julia, use the package manager to install ConstainedLasso:Pkg.clone(\"https://github.com/Hua-Zhou/ConstrainedLasso.git\")This package supports Julia v0.6."
},

{
    "location": "index.html#Examples-1",
    "page": "Home",
    "title": "Examples",
    "category": "section",
    "text": "Examples are found here in this documentation. "
},

{
    "location": "index.html#Citation-1",
    "page": "Home",
    "title": "Citation",
    "category": "section",
    "text": "If you use ConstrainedLasso package in your research, please cite the following reference in the resulting publications:Gaines BR, Zhou H (2016) Algorithms for Fitting the Constrained Lasso. arXiv preprint arXiv:1611.01511."
},

{
    "location": "demo/prostate.html#",
    "page": "Example",
    "title": "Example",
    "category": "page",
    "text": ""
},

{
    "location": "demo/prostate.html#Example-1",
    "page": "Example",
    "title": "Example",
    "category": "section",
    "text": ""
},

{
    "location": "demo/prostate.html#Unconstrained-lasso-1",
    "page": "Example",
    "title": "Unconstrained lasso",
    "category": "section",
    "text": "This demonstration solves a regular, unconstrained lasso problem using the constrained lasso solution path (lsq_classopath.jl) and compares to two other methods.using ConstrainedLasso\nusing DataFrames## load data\nprostate = readtable(\"data/prostate.csv\")\n\n## organize data\n# combine predictors into data matrix\nX= prostate[:, [:lcavol, :lweight, :age, :lbph, :svi, :lcp, :gleason, :pgg45]]\n# demean predictors\nXz = Array{Float64}(X)\nfor i in 1:size(Xz,2)\n    Xz[:, i] -= mean(Xz[:, i])\n    Xz[:, i] /= std(Xz[:, i])\nend\n# define response\ny = Vector(prostate[:lpsa])\n# extract dimensions\nn, p = size(Xz)\n\n## solve using lasso solution path algorithm\nβpath, ρpath, = lsq_classopath(Xz, y);\n@show βpath\n@show ρpath\nnothing # hide## plot solution path \nusing Plots; using LaTeXStrings; pyplot(); # hide \nlabels = [\"lcavol\" \"lweight\" \"age\" \"lbph\" \"svi\" \"lcp\" \"gleason\" \"pgg45\"]\ncolors = [:green :orange :black :purple :red :grey :brown :blue] \nplot(ρpath, βpath', xaxis = (L\"$\\rho$\", (minimum(ρpath),\n      maximum(ρpath))), yaxis = (L\"$\\beta(\\rho$)\"), label=labels, color=colors)\ntitle!(\"Prostrate Data: Solution Path via Constrained Lasso\")\nsavefig(\"prostate.svg\") # hide(Image: )"
},

{
    "location": "demo/warming.html#",
    "page": "Global Warming Data",
    "title": "Global Warming Data",
    "category": "page",
    "text": ""
},

{
    "location": "demo/warming.html#Global-Warming-Data-1",
    "page": "Global Warming Data",
    "title": "Global Warming Data",
    "category": "section",
    "text": ""
},

{
    "location": "demo/warming.html#Section-6.1-1",
    "page": "Global Warming Data",
    "title": "Section 6.1",
    "category": "section",
    "text": "Here we estimate isotonic regression and constrained lasso solution path.using ConstrainedLasso \nusing DataFrames\nusing Base.Test## load & organize data\n# load data \nwarming = readcsv(\"data/warming.csv\", header=true)[1]\n# extract year & response\nyear = warming[:, 1]\ny    = warming[:, 2]\n# extract dimensions\nn = p = size(y, 1)\nX = eye(n)\n\n## estimate models with monotonicity constraints\n## isotonic regression\nmonoreg = readdlm(\"data/monoreg.txt\")\n\n## constrained lasso solution path \n# model set up: inequality constraints\nA = [eye(p-1) zeros(p-1, 1)] - [zeros(p-1, 1) eye(p-1)]\nm2 = size(A, 1)\nb = zeros(m2)\n# estimate constrained lasso solution path\nβ̂path, ρpath, = lsq_classopath(X, y; Aineq = A, bineq = b)\nnothing # hide## compare estimates\n@show maximum(abs.(monoreg - β̂path[:, end]))\n## graph estimates \nusing Plots; pyplot(); using LaTeXStrings; # hide\nscatter(year, y, label=\"Observed Data\", markerstrokecolor=\"darkblue\", \n        markercolor=\"white\")\nscatter!(year, β̂path[:, end], label=L\"Classopath $(\\rho=0)$\", \n        markerstrokecolor=\"black\", marker=:rect, markercolor=\"white\")\nscatter!(year, monoreg, label=\"Isotonic Regression\", marker=:x,\n        markercolor=\"red\", markersize=2)\nxaxis!(\"Year\") \nyaxis!(\"Temperature anomalies\")\ntitle!(\"Global Warming Data\")\nsavefig(\"warming.svg\") # hide(Image: )"
},

{
    "location": "demo/tumor.html#",
    "page": "Brain Tumor Data",
    "title": "Brain Tumor Data",
    "category": "page",
    "text": ""
},

{
    "location": "demo/tumor.html#Brain-Tumor-Data-1",
    "page": "Brain Tumor Data",
    "title": "Brain Tumor Data",
    "category": "section",
    "text": ""
},

{
    "location": "demo/tumor.html#Section-6.2-1",
    "page": "Brain Tumor Data",
    "title": "Section 6.2",
    "category": "section",
    "text": "Here we estimate a generalized lasso model (sparse fused lasso) via the constrained lasso. using ConstrainedLasso# load data\ny = readdlm(\"data/y.txt\")\nlambda_path = readdlm(\"data/lambda_path.txt\")\nbeta_path_fused = readdlm(\"data/beta_path_fused.txt\")[2:end, :]\n\n\n# organize data\nn = p = size(y, 1)\nX = eye(n)\n\n## estimate using constraiend lasso solution path algorithm\n# model setup\nD = [eye(p-1) zeros(p-1, 1)] - [zeros(p-1, 1) eye(p-1)]\nm = size(D, 1)\n\n# transform to constrained lasso\n# calculate SVD\nF = svdfact!(D, thin = false)\n# extract singular values\nsingvals = F[:S]\n# determine rank\nrankD = countnz(F[:S] .> abs(F[:S][1]) * eps(F[:S][1]) * maximum(size(D)))\n\n# extract submatrices of V and U\nV1 = F[:V][:, 1:rankD]\nV2 = F[:V][:, rankD+1:end]\nU1 = F[:U][:, 1:rankD]\nU2 = F[:U][:, rankD+1:end]\n\n# calculate the Moore-Penrose inverse of D\nDplus = V1 * broadcast(*, U1', 1./F[:S])\n# transform design matrix\nXDplus = X * Dplus\n\n# transform to \"tilde\" form\nXV2 = X * V2\n# projection onto C(XV2)\nPxv2 = (1 / dot(XV2, XV2)) * A_mul_Bt(XV2, XV2)\n# orthogonal projection matrix\nMxv2 = eye(size(XV2, 1)) - Pxv2\n# create \"tilde\" data\nỹ = vec(Mxv2 * y)\nX̃ = Mxv2 * XDplus\n\n# constrained solution path\nα̂path, ρpath, = lsq_classopath(X̃, ỹ);\n@show ρpath\n@show α̂path[:, end]\n\n# transform back to beta\nβ̂path = Base.LinAlg.BLAS.ger!(1.0, vec(V2 * ((1 / dot(XV2, XV2)) * \n		At_mul_B(XV2, y))), ones(size(ρpath)), (eye(size(V2, 1)) - \n		V2 * ((1 / dot(XV2, XV2)) * At_mul_B(XV2, X))) * Dplus * α̂path )\n## plot solution path\n# constrained lasso solution path\nusing Plots; pyplot(); using LaTeXStrings; # hide\nplot(ρpath, β̂path', label=\"\", xaxis = (L\"$\\rho$\", (minimum(ρpath),\n      maximum(ρpath))), yaxis = (L\"$\\widehat{\\beta}(\\rho$)\"), width=0.5)\ntitle!(\"Brain Tumor Data: Solution Path via Constrained Lasso\")\nsavefig(\"tumor1.svg\") # hide\nnothing # hide (Image: )## plot generalized lasso solution path (from genlasso R package)\nplot(lambda_path, beta_path_fused', label=\"\", xaxis = (L\"$\\lambda$\", (minimum(lambda_path),\n      maximum(lambda_path))), yaxis = (L\"$\\widehat{\\beta}(\\lambda$)\"), width=0.5)\ntitle!(\"Brain Tumor Data: Generalized Lasso Solution Path\")\nsavefig(\"tumor2.svg\") # hide\nnothing # hide (Image: )# compare estimates at common values of rho "
},

{
    "location": "demo/microbiome.html#",
    "page": "Microbiome Data",
    "title": "Microbiome Data",
    "category": "page",
    "text": ""
},

{
    "location": "demo/microbiome.html#Microbiome-Data-1",
    "page": "Microbiome Data",
    "title": "Microbiome Data",
    "category": "section",
    "text": ""
},

{
    "location": "demo/microbiome.html#Section-6.3-1",
    "page": "Microbiome Data",
    "title": "Section 6.3",
    "category": "section",
    "text": "Our last real data application with the constrained lasso uses microbiome data.     Here the problem is to beginalign\n textminimize hspace1em frac 12boldsymboly-boldsymbolXbeta^2_2 + rhoBig(boldsymbolbeta_1 + frac1-alpha2boldsymbolbeta_2^2Big) \n textsubject to hspace1em sum_j beta_j = 0\nendalignwhere alpha = 1. Hence this problem is reduced to the constrained lasso. using ConstrainedLasso## load & organize data \nzerosum = readcsv(\"data/zerosum.csv\", header=true)[1]\n# extract data \ny = zerosum[:, 1]\nX = zerosum[:, 2:end]\n# extract dimensions \nn, p = size(X)\n\n## model set-up\n# set up equality constraints\nAeq = ones(1, p)\nbeq = [0]\nm1 = size(Aeq, 1)\n\n## constrained Lasso solution path\n# estimate solution path\n@time β̂path, ρpath, = lsq_classopath(X, y; Aeq = Aeq, beq = beq)\n# scale the tuning parameter to match the zeroSum formulation (which\n#	   divides the loss fuction by 2n instead of just 2\nnewρpath = ρpath ./ n\n\n# @show β̂path[:, end]\n# @show ρpath\n\n# calculate L1 norm along path\nnorm1path = zeros(size(β̂path, 2))\nfor i in eachindex(norm1path)\n    norm1path[i] = norm(β̂path[:, i], 1)\nend\n\nnothing # hide Now, let's plot the solution path. using Plots; pyplot(); using LaTeXStrings; # hide\nplot(norm1path, β̂path', xaxis = (L\"$ \\|| \\widehat{\\beta} \\||_1$\"), yaxis=(L\"$\\widehat{\\beta}$\"), label=\"\")\ntitle!(\"Microbiome Data: Solution Path via Constrained Lasso\")\nsavefig(\"micro.svg\") # hideThe following figure plots the coefficient estimate solution paths, widehatboldsymbolbeta(rho), as a function of widehatboldsymbolbeta(rho)_1 using both the zero-sum regression and the constrained lasso. (Image: )As can be seen in the graphs, the coeffcient estimates are nearly indistinguishable except for some very minor differences, which are a result of the slightly different formulations of the two problems."
},

]}
