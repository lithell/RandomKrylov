using LinearAlgebra, SparseArrays, FFTW, Random, MatrixDepot, Plots, Printf, LaTeXStrings
include("gen_err_vec.jl")
include("setupSketchingHandle.jl")
include("whitenBasis.jl")


# Set up problem 
# Wathen 
#nn = 25;
#A = matrixdepot("wathen", nn, nn);
#A = -A;

# Disc diff op
N = 35;
h = 1/N;

rows = vcat(1:N, 1:(N-1), 2:N);
cols = vcat(1:N, 2:N ,1:(N-1));
vals = vcat(-2ones(N), ones(N-1), ones(N-1));
T = 1/h^2*sparse(rows, cols, vals);
A = kron(sparse(I,N,N), T) + kron(T, sparse(I,N,N));

N = size(A,1);

@printf "Problem size: %d\n" N

# sFOM params
num_it = 200;
trunc_len = 4;
mgs = true;

# Set up sketching
sketch_param = Int(num_it);
sketch = setupSketchingHandle(N, sketch_param);

# ex
f(x) = exp(x);

num_runs = 10;

f_mat = f(Matrix(A));

err_mat = zeros(num_it, num_runs);

for i = 1:num_runs

    # Seed for reproducability
    Random.seed!(i);
    
    b = 1000*rand(N);
    ex = f_mat*b;

    # Do sFOM
    err_vec = gen_err_vec(A, b, f, num_it, trunc_len, mgs, sketch, ex);

    err_mat[:,i] = err_vec;

end

# Plots

p = plot(1:num_it, err_mat[:,1:(end-1)], 
        yaxis=:log,
        label=:none,
        minorticks=:true,
        grid=:false,
        linealpha=0.2,
        lw=0.5,
        lc=:purple,
        markershape=:+,
        markersize=1,
        markerstrokewidth=0.1,
        mc=:purple,
        ylimits=(10.0^-16, 10.0^0+1),
        yticks=10.0 .^(-16:2:0),
        framestyle=:box
        );

p = plot!(1:num_it, err_mat[:,end], 
        yaxis=:log,
        label="Errors",
        minorticks=:true,
        grid=:false,
        linealpha=0.2,
        lw=0.5,
        lc=:purple,
        markershape=:+,
        markersize=1,
        markerstrokewidth=0.1,
        mc=:purple,
        ylimits=(10.0^-16, 10.0^2),
        xlimits=(0, num_it+1),
        yticks=10.0 .^(-16:2:2),
        xticks=(0:20:num_it),
        framestyle=:box
        );

worst_run = maximum(err_mat, dims=2);
best_run = minimum(err_mat, dims=2);

plot!(1:num_it, best_run, fillrange=worst_run, 
    alpha=0.25,
    fillcolor=:purple,
    label="Best-worst interval",
    linealpha=0,
    framstyle=:box
    );

plot!(size=(600,600));

xlabel!(L"Number of iterations, $m$")
ylabel!(L"$||\widehat{f}_m - f(A)b || / ||f(A)b||$")
title!(L"Convergence of sFOM, $s=200$")

display(p)
savefig(p, "~/Documents/Julia/sketched_krylov/figs/sFOMs200.pdf")


