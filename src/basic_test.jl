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

f(x) = exp(x);
ex = f(Matrix(A))*b;

# sFOM params
num_it = 120;
trunc_len = 4;
mgs = true;
iter_diff_tol = 10^(-10);

# Set up sketching
sketch_param = 2*num_it;
sketch = setupSketchingHandle(N, sketch_param);

# Do sFOM
approx, conv_flag, iter_diff = sFOM(A, b, f, num_it, trunc_len, mgs, iter_diff_tol, sketch);

# Get rel err
rel_err = norm(approx - ex)/norm(ex);
@printf "Rel. err: %e\n" rel_err
@printf "Conv-flag: %d\n" conv_flag


