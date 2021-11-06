function [Y, W, b, time_elapsed] = qwhiten_new(X)
%   function [Y, W, b] = qwhiten(X)
%   qwhiten is short for quasi-whitening.
%
%   Inputs:
%   X -- The inputted data.  Each column of the matrix is a random
%   sample of the linearly mixed observations.  The number of columns gives
%   the number of samples.
%
%
%   returns:
%       Y -- The quasi-orthogonalized (or whitened) data.
%       W -- The resulting quasi-whitening (or whitening) matrix.
%       b -- The offset of the data from the origin before preprocessing.
%   The samples in Y are computed as y = W(x-b)

    qwhiten_start = tic;
    [d, n] = size(X);  % notice we have X with size of [n, d]
    X = X';
    m = 2; % num of rand vector u, add code when m > 1
    W1 = 0; W2 = 0;
    
    v_vectors_mat = zeros(n,m);

    for i = 1:m
        u = rand(d,1);
        v = X*u;
        W1 = W1 + get_W1(v, n);
        W2 = W2 + get_W2(v, n);
        v_vectors_mat(:,i) = v';
    end
    
    H1 = X'*W1*X;
    H2 = X'*W2*X;
    H = c * (H1-H2);
    
    H2_approx = approx_H2(X, W2);
    H1_approx = approx_H1(X, W1);
    H_approx = c * (H1_approx - H2_approx)
    % decompose H_approx = BB^T
    % return W=B^-1, Y = W*X

    % for each u : 
    % calc v
    % calc H1, H2 (not yet)
    % approx is c(sum of calculation for each u)
    % D is sum of all diags
    % U is [v1,v2][v1,v2].. matrix of rank m 
    % run ricatti on it (plus sign)
    % W^1/2 = D^1/2+UU^T
    % res = X^T W^1/2 * W^1/2 X
    % decompose res = BB^T
    % return W=B^-1
    
    
    
    Y = W*X;
    time_elapsed = toc(qwhiten_start);
    

    fprintf('Total Runtime: %d \n', time_elapsed);


end

function W1 = get_W1(v, n)
    v_squred_diag = diag(power(v,2));
    W1 = ((n-1)/power(n,2))*trace(v_squred_diag)*eye(n) + ((2*n-2)/power(n,2)) * v*v';
end

function W2 = get_W2(v, n)
    v_squred_diag = diag(power(v,2));
    W2 = ((n+1)/n) * v_squred_diag;
end

function H2_approx = approx_H2(X, W2)
    % sqrt of diagonal matrix as W^1/2 and sketching
    sqrt_W2 = sqrtm(W2);
    Z = sqrt_W2 * X;
    % sketching ...
    H2_approx = Z' * Z;
end


function H1_approx = approx_H1(X, W1)
    % sqrt of diagonal matrix + low rank pertubation as W^1/2 using ricatti and sketching
end



