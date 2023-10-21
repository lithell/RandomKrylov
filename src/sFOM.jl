function sFOM(A, b, f, num_it, trunc_len, mgs, iter_diff_tol, sketch, ex)

    # Function performing the sFOM iters, returns vec of errors, and final approx

    # Initializations
    N = size(A, 1);
    err_vec = zeros(num_it);
    final_it = num_it;

    # Allocate for truncated Krylov basis
    V = zeros(ComplexF64, N, trunc_len);

    # Add first Krylov vector
    v = b/norm(b);
    V = hcat(V[:,2:end], v);

    # Sketch v and add to sketched basis
    SV = sketch(v);
    sketch_size = size(SV,1);
    SV = hcat(SV, zeros(ComplexF64, sketch_size, num_it));

    # Allocate for sketched AV
    SAV = zeros(ComplexF64, sketch_size, num_it+1);

    # Allocate for full Krylov basis
    Vfull = zeros(ComplexF64, N, num_it+1);
    Vfull[:,1] = v;

    # init approx, ym
    approx = zeros(3);
    ym = 0;

    # Do sFOM iters
    for m = 1:num_it

        # Compute mat-vec product with latest Arnolid vec
        v = V[:,end];
        Av = A*v;

        # Sketch Av and add to sketched basis 
        SAV[:,m] = sketch(Av);

        # Update v
        v = Av;

        # Orthogonalize
        if mgs
            for i=1:trunc_len
                v = v - V[:,i]*(V[:,i]'*v);
            end
        else
            v = v - V*(V'*v);
        end

        # Normalize 
        v = v/norm(v);

        # Add to truncated basis, and discard vectors below trunc_len
        V = hcat(V[:,2:end], v);

        # Update sketched V
        SV[:,m+1] = sketch(v);

        # Save v in full Krylov basis
        Vfull[:,m+1] = v;

        # Whiten basis
        SVw, SAVw, Rw = whitenBasis(view(SV,:,1:m), view(SAV,:,1:m));

        # Save previous approx
        if m >= 2
            ym_prev = ym;
        end

        # Compute sFOM approximant
        SVm = SVw;
        M = SVm'*SVm;

        coeffs = M\( f((SVm'*SAVw)/M)*(SVm'*sketch(b)) );
        ym = (Rw\coeffs);
        approx = view(Vfull,:,1:m)*ym;

        # Get errors
        if ex != false
            err_vec[m] = norm(approx-ex);
        end

        # Evaluate stopping criterion 
        if m > 2

            stop_crit = norm(Vfull[:,m]) / norm(SV[:,m]);
            stop_crit *= norm( SV[:,1:m]*(ym - vcat(ym_prev, 0)) );

            if stop_crit < iter_diff_tol
                final_it = m;
                err_vec = err_vec[1:m];
                break
            end

        end


    end

    if ex == false
        err_vec = false;
    end

    return err_vec, approx, final_it;
end
