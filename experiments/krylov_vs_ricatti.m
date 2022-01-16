clear all; close all; clc;
RandStream.setGlobalStream(RandStream('mt19937ar','seed', 12121));  % fixed seed

n = 100;                    % num of samples
k = 1;                      % low rank pert dim
Z = randn(n,k);

%todo rank 0
min_rank = 1;
max_rank = 6;
update_rank_list = linspace(min_rank, max_rank, max_rank-min_rank+1);

% graph params - need to change only here
optimization_function = 'sqrt'; % 'sqrt'|'inv_sqrt'
diagonal_dist = 'uniform'; % 'uniform'|'logspace'




[fun_mat_sqrt, dd] = get_graph_params(n, optimization_function, diagonal_dist);
get_diff_graph(n, dd, Z, fun_mat_sqrt, optimization_function, update_rank_list)



%% ------------------------- methods ------------------------- %%
function [fun_mat_sqrt, dd] = get_graph_params(n, optimization_function, diagonal_dist)
    switch optimization_function
        case 'sqrt'
            fun_mat_sqrt = @(X) sqrtm(X);
        case 'inv_sqrt'
            fun_mat_sqrt = @(X) inv(sqrtm(X));
        otherwise
            error('not supportive opt function');
    end

    switch diagonal_dist
        case 'uniform'
            dd = rand(n, 1);  
        case 'logspace'
            dd = logspace(-3, 3, n)';
        otherwise
            error('not supportive diagonal_dist');
    end
end

function result_diag = get_diag(optimization_function, dd)
    switch optimization_function
        case 'sqrt'
            result_diag = diag(sqrt(dd));
        case 'inv_sqrt'
            result_diag = diag(dd.^(-1/2));
        otherwise
            error('not supportive opt function');
    end
end


function get_diff_graph(n, dd, Z, func, optimization_function, update_rank_list)
    
    A = diag(dd) + Z * Z';
    
%     A = diag(dd) - Z * Z'; only in sqrt

    
    true_value = func(A);
    ricatti_err_values = [];
    krylov_err_values = [];

    result_diag = get_diag(optimization_function, dd);
    global pertubation_sign;
    if (strcmp(optimization_function, 'sqrt')); pertubation_sign = 1; else; pertubation_sign = -1; end
    
    for rank = update_rank_list
        ricatti_update = get_ricatti_approx(n, Z, rank, result_diag, optimization_function);
        ricatti_approx = result_diag + pertubation_sign * (ricatti_update*ricatti_update');
        ricatti_err =  get_norm_diff(true_value, ricatti_approx);
        ricatti_err_values(end+1) = ricatti_err;

        krylov_update = get_krylov_approx(func, n, dd, Z, rank);
        krylov_approx = result_diag + krylov_update;
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
%     diff = norm(H_approx-H_true) / norm(H_true);

end

function res = get_C_ricatti(result_diag, Z, optimization_function)
    switch optimization_function
        case 'sqrt'
            res = Z;
        case 'inv_sqrt'
            inv_D = result_diag*result_diag;
            res = inv_D * Z * sqrtm(inv(eye(size(Z,2)) + Z'*inv_D*Z));
        otherwise
            error('not supportive opt function');
    end
end

function ricatti_update = get_ricatti_approx(n, Z, rank, result_diag, optimization_function)
    A_ricatti = result_diag;
    B_ricatti = speye(n);
    C_ricatti = get_C_ricatti(result_diag, Z, optimization_function)';
    
    params = get_ricatti_params(rank);
    [X_Riemannian, ~] =  Riemannian_lowrank_riccati(A_ricatti, B_ricatti, C_ricatti, params);
    ricatti_update = X_Riemannian.Y;
    
end

function G_m = get_G_m(alpha_vals, beta_vals)
    G_1 = diag(alpha_vals);
    G_2 = diag(beta_vals(2:end), 1);
    G_3 = diag(beta_vals(2:end), -1);
    G_m = G_1 + G_2 + G_3;
end

function Xm_sqrt = get_X_m_sqrt(f, Z, G_m, rank)
    Xm_sqrt = f(G_m + (norm(Z))^2 *  eye(1,rank)'*eye(1,rank)) - f(G_m);
end


function krylov_update = get_krylov_approx(f, n, dd, Z, rank)
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
    
    Xm_sqrt = get_X_m_sqrt(f, Z, G_m, rank);

    krylov_update = U*Xm_sqrt*U';
end

function params = get_ricatti_params(rank)
    params.rmax = rank; % Maximum rank
    params.tol_rel = 1e-13; % Stopping criterion for rank incrementating procedure
    params.tolgradnorm = 1e-10; % Stopping for fixed-rank optimization
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
%     set(gca, 'XScale', 'log')
    % todo - ylim 10^0
    ax.XAxisLocation = 'origin';
    set(gca, 'YScale', 'log')
    ax.YAxisLocation = 'origin';
    legend;
%     legend(ca, 'Location', 'southwest')
end
