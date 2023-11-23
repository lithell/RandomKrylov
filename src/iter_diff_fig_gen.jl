using LinearAlgebra, SparseArrays, FFTW, Random, MatrixDepot, Plots, Printf, LaTeXStrings

include("gen_iter_diff_vec.jl")
include("setupSketchingHandle.jl")
include("whitenBasis.jl")


# Set up problem 
nn = 25;
A = matrixdepot("wathen", nn, nn);
A = -A;


N = size(A,1);

@printf "Problem size: %d\n" N

# sFOM params
num_it = 120;
trunc_len = 4;
mgs = true;
iter_diff_tol = 10^(-10);

# Set up sketching
sketch_param = 2*num_it;
sketch = setupSketchingHandle(N, sketch_param);

# func
f(x) = exp(x);

num_runs = 1;

iter_diff_approx_mat = zeros(num_it-1, num_runs);
iter_diff_mat = zeros(num_it-1, num_runs);

for i = 1:num_runs

    # Seed for reproducability
    Random.seed!(i);
    
    b = 100*rand(N);

    # Do sFOM
    approx_diff, iter_diff = gen_iter_diff_vec(A, b, f, num_it, trunc_len, mgs, iter_diff_tol, sketch);

    iter_diff_approx_mat[:,i] = iter_diff;

    iter_diff_mat[:,i] = approx_diff;

end

# Plots

p = plot(2:num_it, iter_diff_mat[:,1:(end-1)], 
        yaxis=:log,
        label=:none,
        minorticks=:true,
        grid=:false,
        linealpha=1,
        lw=1,
        lc=:purple,
        markershape=:+,
        markersize=3,
        markerstrokewidth=0.1,
        mc=:purple,
        ylimits=(10.0^-16, 10.0^0+1),
        yticks=10.0 .^(-16:2:0),
        framestyle=:box
        );

p = plot!(2:num_it, iter_diff_mat[:,end], 
        yaxis=:log,
        label="True",
        minorticks=:true,
        grid=:false,
        linealpha=1,
        lw=1,
        lc=:purple,
        markershape=:+,
        markersize=3,
        markerstrokewidth=0.1,
        mc=:purple,
        ylimits=(10.0^-16, 10.0^2),
        xlimits=(0, num_it+1),
        yticks=10.0 .^(-16:2:2),
        xticks=(0:20:num_it),
        framestyle=:box
        );

p = plot!(2:num_it, iter_diff_approx_mat[:,1:(end-1)], 
        yaxis=:log,
        label=:none,
        minorticks=:true,
        grid=:false,
        linealpha=1,
        lw=1,
        lc=:orange,
        markershape=:utriangle,
        markersize=3,
        markerstrokewidth=0,
        mc=:red,
        ylimits=(10.0^-16, 10.0^0+1),
        yticks=10.0 .^(-16:2:0),
        framestyle=:box
        );

p = plot!(2:num_it, iter_diff_approx_mat[:,end], 
        yaxis=:log,
        label="Approx",
        minorticks=:true,
        grid=:false,
        linealpha=1,
        lw=1,
        lc=:orange,
        markershape=:utriangle,
        markersize=3,
        markerstrokewidth=0,
        mc=:orange,
        ylimits=(10.0^-16, 10.0^2),
        xlimits=(0, num_it+1),
        yticks=10.0 .^(-14:2:2),
        xticks=(0:20:num_it)
        );
#worst_run = maximum(err_mat, dims=2);
#best_run = minimum(err_mat, dims=2);

#plot!(1:num_it, best_run, fillrange=worst_run, 
#    alpha=0.25,
#    fillcolor=:purple,
#    label="Best-worst interval",
#    linealpha=0,
#    );

plot!(size=(600,600));

xlabel!(L"Number of iterations, $m$")
ylabel!("Iterate difference")
title!("Difference in Iterates, True and Approximate")

display(p)
savefig(p, "~/Documents/Julia/sketched_krylov/figs/StoppingCrit.pdf")



