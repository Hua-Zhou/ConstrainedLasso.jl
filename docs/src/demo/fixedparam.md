# Optimize at a fixed tuning parameter value

`lsq_constrsparsereg.jl` fits constrained lasso

```math
\begin{split}
& \text{minimize} \hspace{1em} \frac 12||\boldsymbol{y}-\boldsymbol{X\beta}||^2_2 + \rho||\beta||_1 \\
& \text{ subject to} \hspace{0.5em} \boldsymbol{A\beta}=\boldsymbol{b} \text{ and } \boldsymbol{C\beta} \leq \boldsymbol{d}
\end{split}
```
at a fixed tuning parameter value ``\rho`` or several tuning parameter values provided by user.

### Single tuning parameter value

We demonstrate using a sum-to-zero constraint example
```math
\begin{split}
& \text{minimize} \hspace{1em} \frac 12||\boldsymbol{y}-\boldsymbol{X\beta}||^2_2 + \rho||\beta||_1 \\
& \text{ subject to} \hspace{0.5em} \sum_j \beta_j = 0
\end{split}
```

First, let's define a true parameter `β` such that `sum(β) = 0`.

```@example tuning
using ConstrainedLasso, Base.Test

n, p = 100, 20
β = zeros(p)
β[1:round(Int, p / 4)] = 0
β[(round(Int, p / 4) + 1):round(Int, p / 2)] = 1
β[(round(Int, p / 2) + 1):round(Int, 3p / 4)] = 0
β[(round(Int, 3p / 4) + 1):p] = -1
nothing # hide output
```
Next we generate data based on the true parameter `β`.

```@example tuning
srand(123)
X = randn(n, p)
y = X * β + randn(n)
nothing # hide
```
Since the equality constraint can be written as

```math
\begin{pmatrix} 1 & 1 & \cdots & 1 \end{pmatrix} \beta = 0,
```
we define the constraint as below.

```@example tuning
Aeq   = ones(1, p)
beq   = [0.0]
nothing # hide
```

Now we are ready to fit the constrained lasso problem, say at `ρ=10`.

```@example tuning
ρ = 10.0
logging(DevNull, ConstrainedLasso, :lsq_constrsparsereg, kind=:warn) # hide
β̂, = lsq_constrsparsereg(X, y, ρ; Aeq = Aeq, beq = beq)
nothing # hide
```
We see if the sum of estimated ``\beta`` coefficients equal to 0.

```@example tuning
@test sum(β̂)≈0.0 atol=1e-5
```

### Multiple tuning parameter values

Define `ρlist` to be a sequence of values from 1 to 10.

```@example tuning
ρlist = 1.0:10.0
```
Using the same equality constraints, we fit the constrained lasso.


```@example tuning
logging(DevNull, ConstrainedLasso, :lsq_constrsparsereg, kind=:warn) # hide
β̂, = lsq_constrsparsereg(X, y, ρlist; Aeq = Aeq, beq = beq,
    penwt = penwt)
nothing # hide
```

Now let's test if coefficients sum to 0 at each parameter value.

```@example tuning
@testset "zero-sum for multiple param values" begin for i in sum(β̂, 1)
  @test i≈0.0 atol=1.0e-5
end
end
```
