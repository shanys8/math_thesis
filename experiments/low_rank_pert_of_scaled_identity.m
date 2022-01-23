clear all; close all; clc;
RandStream.setGlobalStream(RandStream('mt19937ar','seed', 1));  % fixed seed


alpha = 0.001;
epsilon = 0.1;

min_rank = 1;
max_rank = 7;
update_rank_list = linspace(min_rank, max_rank, max_rank-min_rank+1);


% Load from files
B1 = double(load('test_mat_1.mat').pickle_data);
B2 = double(load('test_mat_2.mat').pickle_data);


plot_graph(B1, epsilon, alpha, update_rank_list);
% plot_graph(B2, epsilon, alpha, update_rank_list);

fprintf("Done");

%% methods %%

function plot_graph(B, epsilon, alpha, update_rank_list)
    pert_rank = get_rank(B, epsilon);
    pert_rank = 2;  % not working with (82) high perturbation rank
    Z = get_pert_approx(B, pert_rank);
    A = alpha*eye(size(B,1));
%     W = A + U*U';
    true_value = sqrtm(A+Z*Z');
    Asqrt = sqrt(alpha)*speye(size(B,1));
    residuals = [];
    for rank = update_rank_list
        [approx, residual] = sqrtm_update(Asqrt, Z, rank, true_value);
        residuals(end+1) = residual;
    end
    
    disp(residuals);
    plot(update_rank_list, residuals, '-', 'Color', 'blue', 'LineWidth',2, 'DisplayName','ricatti err')     
    
    title('test', 'FontSize',16);
    ylabel('Diff - approx. vs. true value in frobinous norm')
    xlabel('update rank')
    ax = gca;
    ax.XAxisLocation = 'origin';
    set(gca, 'YScale', 'log')
    set(gca, 'YLim', [0, 10^0])
    ax.YAxisLocation = 'origin';
    legend;
end



function params = get_params(rank)
    params.rmax = rank; % Maximum rank
    params.tol_rel = 1e-12; % Stopping criterion for rank incrementating procedure
    params.tolgradnorm = 1e-12; % Stopping for fixed-rank optimization
    params.maxiter = 100; % Number of iterations for fixed-rank optimization
    params.maxinner = 50; % Number of trust-region subproblems for fixed-rank optimization
    params.verbosity = 2; 1; 2; 0; % Show output
end

function [approx, residual] = sqrtm_update(Asqrt, Z, k, true_val)

    global pertubation_sign;
    pertubation_sign = 1;
    A_ricatti = Asqrt;
    B_ricatti = speye(size(Asqrt,1));
    C_ricatti = Z';
    
    
    
    params = get_params(k);
    [X_Riemannian, ~] =  Riemannian_lowrank_riccati(A_ricatti, B_ricatti, C_ricatti, params);

    update_term = X_Riemannian.Y;
    approx = A_ricatti + update_term*update_term';
    residual = norm(approx-true_val, 'fro');
    
end

function pert_rank = get_rank(B, epsilon)
    pert_rank = -1;
    eigs_list = sort(eig(B), 'descend');
    for i=1:length(eigs_list)
        if eigs_list(i) < epsilon
            pert_rank = i-1;
            break;
        end
    end
    fprintf("Done");
end

function pert_approx = get_pert_approx(B, rank)
    [V,D] = eig(B);
    [d,ind] = sort(diag(D), 'descend');
    Ds = D(ind,ind);
    Vs = V(:,ind);
    pert_approx = Vs(:,1:rank)*sqrtm(Ds(1:rank,1:rank));
end

