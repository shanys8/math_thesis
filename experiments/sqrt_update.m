clear all; close all; clc;

n = 15;
k = 3;
dd = rand(n, 1);
% D = diag(dd);
Z = rand(n,k);
% X = rand(n,d);
% global W;
% W = D + Z*Z';
% fun_Wsqrt = @(X) sqrtm(W)*X;


%% sqrt update
% [update_term, residual] = sqrtm_update(n, dd, diag(sqrt(dd)), Z, k); 

%% inv sqrt update
[update_term, residual] = inv_sqrtm_update(n, k, dd, diag(dd.^(-1/2)), Z); 


fprintf("done");

%% ---------------------- methods ---------------------- %%
function params = get_params(k)
    params.rmax = 2*k; % Maximum rank
    params.tol_rel = 1e-6; % Stopping criterion for rank incrementating procedure
    params.tolgradnorm = 1e-10; % Stopping for fixed-rank optimization
    params.maxiter = 100; % Number of iterations for fixed-rank optimization
    params.maxinner = 30; % Number of trust-region subproblems for fixed-rank optimization
    params.verbosity = 2; 1; 2; 0; % Show output
end

% inv = use woodberry go get otther U as input to ricatti

function U = get_C_ricatti(n, k, A_inv_sqrt, Z)
    inv_A = A_inv_sqrt*A_inv_sqrt;
    U = inv_A * Z * sqrtm(inv(eye(k) + Z'*inv_A*Z));
end

function [update_term, residual] = inv_sqrtm_update(n, k, dd, A_inv_sqrt, Z)
    global pertubation_sign;
    pertubation_sign = -1;
    A_ricatti = A_inv_sqrt;
    B_ricatti = speye(n);
    C_ricatti = get_C_ricatti(n, k, A_inv_sqrt, Z)';
    
    params = get_params(k);
    [X_Riemannian, ~] =  Riemannian_lowrank_riccati(A_ricatti, B_ricatti, C_ricatti, params);

    update_term = X_Riemannian.Y;
    approx = A_ricatti - update_term*update_term';
    true_val = (diag(dd)+Z*Z')^(-1/2);
    residual = norm(approx-true_val, 'fro');
end

function [update_term, residual] = sqrtm_update(n, dd, Asqrt, Z, k)

    global pertubation_sign;
    pertubation_sign = 1;
    A_ricatti = Asqrt;
    B_ricatti = speye(n);
    C_ricatti = Z';
    
    params = get_params(k);
    [X_Riemannian, ~] =  Riemannian_lowrank_riccati(A_ricatti, B_ricatti, C_ricatti, params);

    update_term = X_Riemannian.Y;
    approx = A_ricatti + update_term*update_term';
    true_val = sqrtm(diag(dd)+Z*Z');
    residual = norm(approx-true_val, 'fro');
    
end


    