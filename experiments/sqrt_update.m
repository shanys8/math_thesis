clear all; close all; clc;

n = 15;
k = 3;
dd = rand(n, 1);
% D = diag(dd);
Z = rand(n,k);
Z = 0.01*Z;



%% sqrt update W^(1/2) of W = D + Z*Z' - only need D^(1/2)
% [approx, update_term, residual] = sqrtm_update(n, dd, diag(sqrt(dd)), Z, k, sqrtm(diag(dd)+Z*Z')); 

%% inv sqrt update W^(-1/2) of W = D + Z*Z' -  need D^(1/2) &  D^(-1/2)
[approx, update_term, residual] = inv_sqrtm_update(n, k, dd, diag(sqrt(dd)), diag(dd.^(-1/2)), Z, (diag(dd) + Z*Z')^(-1/2)); 

%% inv sqrt update W^(-1/2) of W = D - Z*Z' - only need D^(-1/2)
% [approx, update_term, residual] = inv_sqrtm_with_minus_update(n, k, dd, diag(dd.^(-1/2)), Z, (diag(dd) - Z*Z')^(-1/2)); 

%% sqrt update W^(1/2) of W = D - Z*Z' -  need D^(1/2) &  D^(-1/2)
% [approx, update_term, residual] = sqrtm_with_minus_update(n, k, dd, diag(sqrt(dd)), diag(dd.^(-1/2)), Z, sqrtm(diag(dd)-Z*Z')); 


fprintf("done");

%% ---------------------- methods ---------------------- %%
function params = get_params(k)
    params.rmax = 5; % Maximum rank
    params.tol_rel = 1e-6; % Stopping criterion for rank incrementating procedure
    params.tolgradnorm = 1e-10; % Stopping for fixed-rank optimization
    params.maxiter = 100; % Number of iterations for fixed-rank optimization
    params.maxinner = 30; % Number of trust-region subproblems for fixed-rank optimization
    params.verbosity = 2; 1; 2; 0; % Show output
end


function U = get_C_ricatti(k, A_inv_sqrt, Z)
    inv_A = A_inv_sqrt*A_inv_sqrt;
    U = inv_A * Z * sqrtm(inv(eye(k) + Z'*inv_A*Z));
end

function U = get_inv_sqrtm_minus_update_C_ricatti(k, A_inv_sqrt, Z)
    inv_A = A_inv_sqrt*A_inv_sqrt;
    U = inv_A * Z * sqrtm(inv(eye(k) - Z'*inv_A*Z));
end

function U = get_sqrtm_minus_update_C_ricatti(k, A_inv_sqrt, Z)
    inv_A = A_inv_sqrt*A_inv_sqrt;
    U = inv_A * Z * sqrtm(inv(eye(k) - Z'*inv_A*Z));
end

function res = get_inv_sqrtm_update_C_ricatti(k, dd, Z)
    D = diag(dd);
    inv_D = diag(dd.^-1);
    H = inv_D*Z*sqrtm(inv(eye(k) + Z'*inv_D*Z));
    res = D*H*sqrtm(inv(eye(k) - H'*D*H));
end

function [approx, update_term, residual] = sqrtm_with_minus_update(n, k, dd, Asqrt, A_inv_sqrt, Z, true_val)
    global pertubation_sign;
    pertubation_sign = 1;
    A_ricatti = A_inv_sqrt;
    B_ricatti = speye(n);
    C_ricatti = get_sqrtm_minus_update_C_ricatti(k, A_inv_sqrt, Z)';
    
    params = get_params(k);
    [X_Riemannian, ~] =  Riemannian_lowrank_riccati(A_ricatti, B_ricatti, C_ricatti, params);

    Y = X_Riemannian.Y;

    % woodbury
    update_term = Asqrt*Y*sqrtm(inv(eye(size(Y,2))+Y'*Asqrt*Y));
    approx = Asqrt - update_term*update_term';
    residual = norm(approx-true_val, 'fro');
end

%% version 2 %%
function [approx, update_term, residual] = inv_sqrtm_update(n, k, dd, Asqrt, A_inv_sqrt, Z, true_val)
    global pertubation_sign;
    pertubation_sign = 1;
    A_ricatti = Asqrt;
    B_ricatti = speye(n);
    C_ricatti = Z';
    
    params = get_params(k);
    [X_Riemannian, ~] =  Riemannian_lowrank_riccati(A_ricatti, B_ricatti, C_ricatti, params);

    Y = X_Riemannian.Y;

    % woodbury
    update_term = A_inv_sqrt*Y*sqrtm(inv(eye(size(Y,2))+Y'*A_inv_sqrt*Y));
    approx = A_inv_sqrt - update_term*update_term';
    residual = norm(approx-true_val, 'fro');
end

%% version 1 %%
% function [update_term, residual] = inv_sqrtm_update(n, k, dd, Asqrt, A_inv_sqrt, Z, true_val)
%     global pertubation_sign;
%     pertubation_sign = 1;
%     A_ricatti = Asqrt;
%     B_ricatti = speye(n);
%     C_ricatti = get_inv_sqrtm_update_C_ricatti(k, dd, Z)';
%     
%     params = get_params(k);
%     [X_Riemannian, ~] =  Riemannian_lowrank_riccati(A_ricatti, B_ricatti, C_ricatti, params);
% 
%     Y = X_Riemannian.Y;
% 
%     % woodbury
%     update_term = A_inv_sqrt*Y*sqrtm(inv(eye(size(Y,2))+Y'*A_inv_sqrt*Y));
%     approx = A_inv_sqrt - update_term*update_term';
%     residual = norm(approx-true_val, 'fro');
% end

function [approx, update_term, residual] = inv_sqrtm_with_minus_update(n, k, dd, A_inv_sqrt, Z, true_val)
    global pertubation_sign;
    pertubation_sign = 1;
    A_ricatti = A_inv_sqrt;
    B_ricatti = speye(n);
    C_ricatti = get_inv_sqrtm_minus_update_C_ricatti(k, A_inv_sqrt, Z)';
    
    params = get_params(k);
    [X_Riemannian, ~] =  Riemannian_lowrank_riccati(A_ricatti, B_ricatti, C_ricatti, params);

    update_term = X_Riemannian.Y;
    approx = A_ricatti + update_term*update_term';
    residual = norm(approx-true_val, 'fro');
end

function [approx, update_term, residual] = sqrtm_update(n, dd, Asqrt, Z, k, true_val)

    global pertubation_sign;
    pertubation_sign = 1;
    A_ricatti = Asqrt;
    B_ricatti = speye(n);
    C_ricatti = Z';
    
    params = get_params(k);
    [X_Riemannian, ~] =  Riemannian_lowrank_riccati(A_ricatti, B_ricatti, C_ricatti, params);

    update_term = X_Riemannian.Y;
    approx = A_ricatti + update_term*update_term';
    residual = norm(approx-true_val, 'fro');
    
end


    