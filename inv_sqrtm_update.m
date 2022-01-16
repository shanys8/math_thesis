function U = inv_sqrtm_update(D, alpha, Z, max_rank)
% U = inv_sqrtm_update(D, alpha, Z, max_rank)
%
% Find a U that has max_rank columns such that
%    D - alpha * U * U^T ~=~ (D^{-2} + alpha * Z * Z^T)^{-1/2} 
%
% max_rank is optional. Will default to twice the number of columns in Z. 

n = size(Z, 1); k = size(Z, 2);

if (alpha ~= 1 && alpha ~= -1)
    warning('Alpha should be +1 or -1. Taking sign.');
    alpha = sign(alpha);
end

if (nargin < 4)
    max_rank = 2 * size(Z, 2);
end

global pertubation_sign;
pertubation_sign = -alpha;


A = D;
B = speye(n);
G = D * Z;
DG = D * G;
[UU, SS, VV] = svd(G, 0);
C = (1./sqrt(1 + alpha * diag(SS).^2)) .* (VV' * DG');

params.rmax = max_rank; % Maximum rank
params.tol_rel = 1e-10; % Stopping criterion for rank incrementating procedure
params.tolgradnorm = 1e-12; % Stopping for fixed-rank optimization
params.maxiter = 100; % Number of iterations for fixed-rank optimization
params.maxinner = 30; % Number of trust-region subproblems for fixed-rank optimization
params.verbosity = 0; 2; 1; 2; 0; % Show output
[X_Riemannian, ~] =  Riemannian_lowrank_riccati(A, B, C, params);

    U = X_Riemannian.Y;