clear all; close all; clc;
RandStream.setGlobalStream(RandStream('mt19937ar','seed', 12121));  % fixed seed

n = 100;                    % num of samples
k = 1;                      % low rank pert dim
Z = randn(n,k);
Z = 0.01*Z/norm(Z);

min_rank = 0;
max_rank = 5;
update_rank_list = linspace(min_rank, max_rank, max_rank-min_rank+1);

% graph params - need to change only here
% 'sqrt_plus_update'|'sqrt_minus_update'|'inv_sqrt_plus_update'|'inv_sqrt_minus_update
optimization_function = 'inv_sqrt_minus_update'; 
diagonal_dist = 'uniform'; % 'uniform'|'logspace'

% uniform / logscale with sqrt - between 0-7 rank
% invsqrt - between 0-5 rank


dd = get_diag_vector(n, diagonal_dist);
get_diff_graph(n, dd, Z, optimization_function, update_rank_list)



%% ------------------------- methods ------------------------- %%
function dd = get_diag_vector(n, diagonal_dist)
    switch diagonal_dist
        case 'uniform'
            dd = rand(n, 1);  
        case 'logspace'
            dd = logspace(-3, 3, n)';
        otherwise
            error('not supportive diagonal_dist');
    end
end


function func = get_krylov_opt_func(optimization_function)
    switch optimization_function
        case {'sqrt_plus_update', 'sqrt_minus_update'}
            func = @(X) sqrtm(X);
        case {'inv_sqrt_plus_update', 'inv_sqrt_minus_update'}
            func = @(X) inv(sqrtm(X));
        otherwise
            error('not supportive opt function');
    end
end

function result_diag = get_diag(optimization_function, dd)
    switch optimization_function
        case {'sqrt_plus_update', 'sqrt_minus_update'}
            result_diag = diag(sqrt(dd));
        case {'inv_sqrt_plus_update', 'inv_sqrt_minus_update'}
           result_diag = diag(dd.^(-1/2));
        otherwise
            error('not supportive opt function');
    end
end

function krylov_sign = get_krylov_opt_sign(optimization_function)
    switch optimization_function
        case {'sqrt_plus_update', 'inv_sqrt_plus_update'}
            krylov_sign = 1;
        case {'sqrt_minus_update', 'inv_sqrt_minus_update'}
           krylov_sign = -1;
        otherwise
            error('not supportive opt function');
    end
end

function true_value = get_true_value(optimization_function, dd, Z)
    switch optimization_function
        case 'sqrt_plus_update'
            true_value = sqrtm(diag(dd)+Z*Z');
        case 'sqrt_minus_update'
            true_value = sqrtm(diag(dd)-Z*Z');
        case 'inv_sqrt_plus_update'
            true_value = (diag(dd) + Z*Z')^(-1/2);
        case 'inv_sqrt_minus_update'
            true_value = (diag(dd) - Z*Z')^(-1/2);
        otherwise
            error('not supportive opt function');
    end
end

function get_diff_graph(n, dd, Z, optimization_function, update_rank_list)

    ricatti_err_values = [];
    krylov_err_values = [];

    result_diag = get_diag(optimization_function, dd);
    true_value = get_true_value(optimization_function, dd, Z);

    global pertubation_sign;
    pertubation_sign = 1;
    for rank = update_rank_list
        if rank == 0
            ricatti_err =  get_norm_diff(true_value, result_diag);
            krylov_err =  get_norm_diff(true_value, result_diag);
            ricatti_err_values(end+1) = ricatti_err;
            krylov_err_values(end+1) = krylov_err;
            continue;
        end
        
        switch optimization_function
            case 'sqrt_plus_update'
                [approx, update_term, ricatti_err] = sqrtm_update(n, dd, diag(sqrt(dd)), Z, rank, true_value); 
            case 'sqrt_minus_update'
                [approx, update_term, ricatti_err] = sqrtm_with_minus_update(n, rank, dd, diag(sqrt(dd)), diag(dd.^(-1/2)), Z, true_value); 
            case 'inv_sqrt_plus_update'
                [approx, update_term, ricatti_err] = inv_sqrtm_update(n, rank, dd, diag(sqrt(dd)), diag(dd.^(-1/2)), Z, true_value); 
            case 'inv_sqrt_minus_update'
                [approx, update_term, ricatti_err] = inv_sqrtm_with_minus_update(n, rank, dd, diag(dd.^(-1/2)), Z, true_value); 
            otherwise
                error('not supportive opt function');
        end

        ricatti_err_values(end+1) = ricatti_err;
        
        % W^(1/2) or W^(-1/2) 
        func = get_krylov_opt_func(optimization_function);
        % W = D + ZZ' or W = D - ZZ'
        krylov_sign = get_krylov_opt_sign(optimization_function); 
        
        krylov_update = get_krylov_approx(func, n, dd, Z, rank, krylov_sign);
        krylov_approx = result_diag + krylov_sign * krylov_update;
        krylov_err =  get_norm_diff(true_value, krylov_approx);
        krylov_err_values(end+1) = krylov_err;
    end

    fprintf("krylov_err_values:");
    disp(krylov_err_values);

    fprintf("ricatti_err_values:");
    disp(ricatti_err_values);

    plot_graphs(update_rank_list, ricatti_err_values, krylov_err_values);

end



function diff = get_norm_diff(H_true, H_approx)
    diff = norm(H_approx-H_true, 'fro') / norm(H_true, 'fro');
end


function G_m = get_G_m(alpha_vals, beta_vals)
    G_1 = diag(alpha_vals);
    G_2 = diag(beta_vals(2:end), 1);
    G_3 = diag(beta_vals(2:end), -1);
    G_m = G_1 + G_2 + G_3;
end

function Xm_sqrt = get_X_m_sqrt(f, Z, G_m, rank, krylov_sign)
    Xm_sqrt = f(G_m + krylov_sign*(norm(Z))^2 *  eye(1,rank)'*eye(1,rank)) - f(G_m);
end


function krylov_update = get_krylov_approx(f, n, dd, Z, rank, krylov_sign)
    alpha_vals = [];
    beta_vals = [];

    u_0 = zeros(n,1);
    u_1 = Z / norm(Z);
    U = [u_0 u_1];
    next_beta = 0;
    beta_vals(end+1) = next_beta;
    
    for i=1:rank
        w = dd.*U(:,i+1) - next_beta*U(:,i);
        next_alpha = U(:,i+1)'*w;
        alpha_vals(end+1) = next_alpha;
        w = w - next_alpha * U(:,i+1);
        next_beta = norm(w);
        beta_vals(end+1) = next_beta;
        U = [U, (1/next_beta) * w];
    end
    
    beta_vals = beta_vals(1:end-1);
    U = U(:,2:end-1);
    
    G_m = get_G_m(alpha_vals, beta_vals);
    
    Xm_sqrt = get_X_m_sqrt(f, Z, G_m, rank, krylov_sign);

    krylov_update = U*Xm_sqrt*U';
end

function params = get_ricatti_params(rank)
    params.rmax = rank; % Maximum rank
    params.tol_rel = 1e-13; % Stopping criterion for rank incrementating procedure
    params.tolgradnorm = 1e-13; % Stopping for fixed-rank optimization
    params.maxiter = 100; % Number of iterations for fixed-rank optimization
    params.maxinner = 30; % Number of trust-region subproblems for fixed-rank optimization
    params.verbosity = 2; 1; 2; 0; % Show output
end


function plot_graphs(update_rank_list, ricatti_err_values, krylov_err_values)

    hold on;
    plot(update_rank_list, ricatti_err_values, '-', 'Color', 'red', 'LineWidth',2, 'DisplayName','ricatti err')
    grid on;   
    
    hold on;
    plot(update_rank_list, krylov_err_values, '-', 'Color', 'blue', 'LineWidth',2, 'DisplayName','krylov err')
    grid on;    

    title('Ricatti vs. Krylov', 'FontSize',16)
    ylabel('Diff - approx. vs. true value in fro norm')
    xlabel('update rank')
    ax = gca;
    ax.XAxisLocation = 'origin';
    set(gca, 'YScale', 'log')
    set(gca, 'YLim', [0, 10^0])
    ax.YAxisLocation = 'origin';
    legend;
end



%% ---------------------- methods ---------------------- %%
function params = get_params(rank)
    params.rmax = rank; % Maximum rank
    params.tol_rel = 1e-6; % Stopping criterion for rank incrementating procedure
    params.tolgradnorm = 1e-10; % Stopping for fixed-rank optimization
    params.maxiter = 100; % Number of iterations for fixed-rank optimization
    params.maxinner = 30; % Number of trust-region subproblems for fixed-rank optimization
    params.verbosity = 2; 1; 2; 0; % Show output
end


function U = get_C_ricatti(A_inv_sqrt, Z)
    inv_A = A_inv_sqrt*A_inv_sqrt;
    U = inv_A * Z * sqrtm(inv(eye(size(Z,2)) + Z'*inv_A*Z));
end

function U = get_inv_sqrtm_minus_update_C_ricatti(A_inv_sqrt, Z)
    inv_A = A_inv_sqrt*A_inv_sqrt;
    U = inv_A * Z * sqrtm(inv(eye(size(Z,2)) - Z'*inv_A*Z));
end

function U = get_sqrtm_minus_update_C_ricatti(A_inv_sqrt, Z)
    inv_A = A_inv_sqrt*A_inv_sqrt;
    U = inv_A * Z * sqrtm(inv(eye(size(Z,2)) - Z'*inv_A*Z));
end

% function res = get_inv_sqrtm_update_C_ricatti(k, dd, Z)
%     D = diag(dd);
%     inv_D = diag(dd.^-1);
%     H = inv_D*Z*sqrtm(inv(eye(size(Z,2)) + Z'*inv_D*Z));
%     res = D*H*sqrtm(inv(eye(size(H,2)) - H'*D*H));
% end

function [approx, update_term, residual] = sqrtm_with_minus_update(n, k, dd, Asqrt, A_inv_sqrt, Z, true_val)
    global pertubation_sign;
    pertubation_sign = 1;
    A_ricatti = A_inv_sqrt;
    B_ricatti = speye(n);
    C_ricatti = get_sqrtm_minus_update_C_ricatti(A_inv_sqrt, Z)';
    
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


function [approx, update_term, residual] = inv_sqrtm_with_minus_update(n, k, dd, A_inv_sqrt, Z, true_val)
    global pertubation_sign;
    pertubation_sign = 1;
    A_ricatti = A_inv_sqrt;
    B_ricatti = speye(n);
    C_ricatti = get_inv_sqrtm_minus_update_C_ricatti(A_inv_sqrt, Z)';
    
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

