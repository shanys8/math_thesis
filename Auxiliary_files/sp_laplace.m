function A = sp_laplace(n)
    % Generates sparse 2D discrete Laplacian matrix of dimension n^2.
    
    r = zeros(1,n); %
    r(1:2) = [2, -1];
    T = toeplitz(r);
    E = speye(n); %
    A = kron(T, E) + kron(E, T);
    
end