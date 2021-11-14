function [Y, W, b, time_elapsed] = qwhiten(X, op)
%   function [Y, W, b] = qwhiten(X, op)
%   qwhiten is short for quasi-whitening.
%
%   Inputs:
%   X -- The inputted data.  Each column of the matrix is a random
%   sample of the linearly mixed observations.  The number of columns gives
%   the number of samples.
%
%   op -- 'whiten', 'id quas-orth'.  The "identity" based
%   algorithm is deterministic in nature, but requires more operations
%   than "random" (the time difference is a constant factor).  Both
%   "identity" and "random" are quasi-whitening algorithms robust to
%   Gaussian noise, whereas "whiten" is a standard whitening algorithm
%   based on PCA that is not robust to Gaussian noise.
%   "half rnd" only works when the signals all have kurtosis of the same
%   sign.
%
%   returns:
%       Y -- The quasi-orthogonalized (or whitened) data.
%       W -- The resulting quasi-whitening (or whitening) matrix.
%       b -- The offset of the data from the origin before preprocessing.
%   The samples in Y are computed as y = W(x-b)

    qwhiten_start = tic;
    [d, n] = size(X);    
    C = zeros(d);

    % mean subtract data
    b = mean(X, 2);
    % X = X - repmat(b, 1, n);
    X = bsxfun(@minus, X, b);
 
    % Generate the initial hessian / cumulant tensor matrix.
    XXt = X*X';  % Compute in order to (unnoticeably) increase hessian efficiency.
    switch op
        case 'whiten'
            C = cov(X');
            W = C^(-0.5);
            Y = W * X;
            time_elapsed = toc(qwhiten_start);
            return
        case 'id quas-orth' 
            step1_start = tic;
            for i = 1:d
                u = canonvec(i, d);
                C = C + cum4hes(X, u, XXt);
            end
            C = C / 12;  % Division by 12 makes this equivalent to a change of variable in the fourth cumulant tensor techniques.
            step1_time_elapsed = toc(step1_start);
        case 'quasi whiten' 
            step1_start = tic;
            use_canonical_vec = 0;
%             XtX = X' * X;
            if use_canonical_vec
                for i = 1:d
                    u = canonvec(i, d);
                    C = C + cum4hes_approx(X', u, XXt);
                end
            else
                m = 2;
                % Transpose X so that rows are samples.
                v_vectors_mat = zeros(n,m);
                for i = 1:m
                    u = get_u(d);
                    v = X*u;
                    v_vectors_mat(:,i) = v';
                    C = C + cum4hes_approx(X', u, v, XXt);
                end
            end
            C = C / 12;  % Division by 12 makes this equivalent to a change of variable in the fourth cumulant tensor techniques.
            step1_time_elapsed = toc(step1_start);
        otherwise
            fprintf(2, ['ERROR:  Invalid option flag:  ' op]);
            return
    end
            


    % Inversion and second step of the quasi-whitening.
    step2_start = tic;
    C2 = zeros(d);
    [U, D] = eig(C);
    step2_time_elapsed = toc(step2_start);
    % will be using the inverse of C, and placing half of the effect of D
    % in the U vectors.
    
    step3_start = tic;
    for i = 1:d
        C2 = C2 + (1/D(i, i))*cum4hes(X, U(:, i), XXt);
    end
    C2 = C2 / 12;
    step3_time_elapsed = toc(step3_start);
    
   step4_start = tic;
   [U, Lambda] = eig(C2);
   step4_time_elapsed = toc(step4_start);
   len = size(Lambda, 1);
   Lambda_mask = (diag(Lambda) > 0);
   if (sum(Lambda_mask) < len)
       warning(['Decomposition matrix for quasi-orthogonalization is not positive definite.  ' ...
           'Ignoring negative eigenvalue directions.  Output could very easily be bogus.  ' ...
           'Standard whitening may provide better results.']);
       Lambda
       Lambda(1:len+1:end) = Lambda * Lambda_mask;
       W = pinv(Lambda)^(1/2) * U';
   else
       W = Lambda^(-1/2)*U';
   end
    Y = W*X;
    time_elapsed = toc(qwhiten_start);
    
    fprintf('Quasi Orthogonalization: \n');
    fprintf('step1_time_elapsed (Cumulant): %d \n', step1_time_elapsed);
    fprintf('step2_time_elapsed: %d \n', step2_time_elapsed);
    fprintf('step3_time_elapsed (Cumulant): %d \n', step1_time_elapsed);
    fprintf('step4_time_elapsed: %d \n', step4_time_elapsed);
    fprintf('Total Runtime: %d \n', time_elapsed);
    fprintf('Step 1+3 of total Runtime: %.2f percent\n', 100*(step1_time_elapsed+step3_time_elapsed)/time_elapsed);


end

function ei = canonvec(i, dim)
%   function ei = canonvec(i, dim)
%   Produces the ith canonical vector in a dim dimensional space.
    ei = zeros(dim, 1);
    ei(i) = 1;
end

function u = rndunitvec(dim)
%  Generates a random vector in dim uniformly from the unit sphere.
    u = normrnd( zeros(dim, 1), ones(dim, 1) );
    u = u / norm(u);
end

function timestamp = get_curr_ts()
    n = now;
    ds = datestr(n);
    dt = datetime(ds);
    timestamp = posixtime(dt);
end

function u = get_u(d)
    u = randn(d, 1); 
    u = u / norm(u);
end



%% ideas of how to approximate H2 and H1 throughout the m iterations
function H2_approx = approx_H2(X, W2)
    % sqrt of diagonal matrix as W^1/2 and sketching
    sqrt_W2 = sqrtm(W2);
    Z = sqrt_W2 * X;
    % sketching ...
    H2_approx = Z' * Z;
end


function H1_approx = approx_H1(n, X, W1_diag, v_vectors_mat)
    W1_diag_sqrt = sqrtm(W1_diag) % is there a more efficient way? sqrt of diag entries?
%     W1_low_rank = v_vectors_mat * v_vectors_mat';
    % ricatti
    A_ricatti = W1_diag_sqrt;
    B_ricatti = eye(n);
    C_ricatti = sqrt(((2*n-2)/power(n,2))) * v_vectors_mat';
    
    % need to get it work from here
    [X_Riemannian, info_Riemannian] =  Riemannian_lowrank_riccati(A_ricatti, B_ricatti, C_ricatti, params);

    Delta = X_Riemannian.Y * X_Riemannian.Y';
    
    sqrt_W1_approx = W1_diag_sqrt + Delta;
    Z = sqrt_W1_approx * X;
    % sketching ...
    H1_approx = Z' * Z;
end





