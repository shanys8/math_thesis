clear all; close all; clc;



n = 1000;
% d = 7;
k = 3;
dd = rand(n, 1);
% D = diag(dd);
Z = rand(n,k);
% X = rand(n,d);
% global W;
% W = D + Z*Z';
% fun_Wsqrt = @(X) sqrtm(W)*X;


% [update_term, residual] = sqrtm_update(n, dd, diag(sqrt(dd)), Z, k); 

[update_term, residual] = inv_sqrtm_update(n, dd, diag(dd.^(-1/2)), Z, k); 


fprintf("done");

function params = get_params(k)
    params.rmax = 2*k; % Maximum rank
    params.tol_rel = 1e-6; % Stopping criterion for rank incrementating procedure
    params.tolgradnorm = 1e-10; % Stopping for fixed-rank optimization
    params.maxiter = 100; % Number of iterations for fixed-rank optimization
    params.maxinner = 30; % Number of trust-region subproblems for fixed-rank optimization
    params.verbosity = 2; 1; 2; 0; % Show output
end

% inv = use woodberry go get otther U as input to ricatti

function [update_term, residual] = inv_sqrtm_update(n, dd, Asqrt, Z, k)
    global pertubation_sign;
    pertubation_sign = -1;
    A_ricatti = Asqrt;
    B_ricatti = speye(n);
    C_ricatti = Z';
    
    params = get_params(k);
    [X_Riemannian, ~] =  Riemannian_lowrank_riccati(A_ricatti, B_ricatti, C_ricatti, params);

    update_term = X_Riemannian.Y;
    approx = A_ricatti + update_term*update_term';
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


    