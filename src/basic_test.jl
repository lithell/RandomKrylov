using LinearAlgebra, SparseArrays, FFTW, Random, MatrixDepot, Plots, Printf

include("sFOM.jl")
include("setupSketchingHandle.jl")
include("whitenBasis.jl")

# Seed for reproducability
Random.seed!(1);

# Set up problem 
nn = 25;
A = matrixdepot("wathen", nn, nn);
A = -A;
N = size(A,1);
b = rand(N);

@printf "Problem size: %d\n" N

# Exact sol
f(x) = exp(x);
ex = f(Matrix(A))*b;

# sFOM params
num_it = 120;
trunc_len = 4;
mgs = false;
iter_diff_tol = 10^(-10);

# Set up sketching
sketch_param = 2*num_it;
sketch = setupSketchingHandle(N, sketch_param);

# Do sFOM
err_vec, approx, final_it = sFOM(A, b, f, num_it, trunc_len, mgs, iter_diff_tol, sketch, ex)

# Get rel err
err_vec = err_vec./norm(ex);

# Plot errs
plot(1:final_it, err_vec,
    yaxis=:log,
    linewidth=1.5,
    label=:none,
    color=:red,
    markershape=:rect,
    markersize=1.7,
    minorgrid=:true,
    yticks=10.0 .^(-14:2:0)
    )
xlabel!("Number of Iterations")
ylabel!("Relative Error")
title!("Error in sFOM")


