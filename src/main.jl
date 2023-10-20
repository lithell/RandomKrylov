print("Loading Packs...\n")
using LinearAlgebra, SparseArrays, FFTW, Random, MatrixDepot, Plots

include("sFOM.jl")
include("setupSketchingHandle.jl")
include("whitenBasis.jl")
print("Done Loading Packs\n")

# Seed for reproducability
Random.seed!(1);

# Set up problem 
nn = 25;
A = matrixdepot("wathen", nn, nn);
A = -A;
N = size(A,1);
b = rand(N);

print("Problem size: \n")
print(N)
print("\n")

# Exact sol
f(x) = exp(x);
print("Computing exact...\n")
ex = f(Matrix(A))*b;
print("Done computing exact\n")

# sFOM params
num_it = 120;
trunc_len = 4;
mgs = false;

# Set up sketching
sketch_param = 2*num_it;
sketch = setupSketchingHandle(N, sketch_param);

# Do sFOM
err_vec, approx = sFOM(A, b, f, num_it, trunc_len, mgs, sketch, ex)

# Get rel err
err_vec = err_vec./norm(ex);

# Plot errs
plot(1:num_it, err_vec,
    yaxis=:log,
    linewidth=1.5,
    label=:none,
    color=:red,
    markershape=:rect,
    markersize=1.7,
    minorgrid=:true,
    yticks=10.0 .^(-12:2:0)
    )
xlabel!("Number of Iterations")
ylabel!("Relative Error")
title!("Error in sFOM")


