clear all; close all; clc;
RandStream.setGlobalStream(RandStream('mt19937ar','seed', 1));  % fixed seed

% ------------------------- START ------------------------- %

% General values
with_noise = false;
d = 20; % dimension
A = randn(d, d); % ICA demixing matrix

global pertubation_sign
pertubation_sign = 1;

print_noise(with_noise)

min_n = 2;
max_n = 5; % 10^7 is the max dim possible to run
min_m = 3;
max_m = 3;

n_choices = logspace(min_n, max_n, max_n-min_n + 1);
m_choices = linspace(min_m, max_m, max_m-min_m + 1);
% m_choices = 1:10:30;

% Generate latent and samples
S = sign(randn(d, 10^max_n));

op = 'ricatti'; % 'plain'|'ricatti'|'sketching'
plot_graphs(op, m_choices, n_choices, d, A, S)

% ------------------------- END ------------------------- %




% ----------------------- METHODS ----------------------- %


% diagnonal in which the kth entry is 24(Ak · u)^2 when Ak is the kth
% column of A - this is correct
function DAu = get_DAu(A,u)
    Atu = A' * u;
    DAu = diag(24 * Atu .* Atu);
end

% frobinous norm diff between true value approx. relative to true value
% norm
function diff = get_norm_diff(H_true, H_approx)
    diff = norm(H_approx-H_true, 'fro') / norm(H_true, 'fro');
end

function H1_diag = get_H1_diag(n, v, XtX)
    H1_diag = ((n-1)/(n*n)) * sum(v.^2) * XtX;
end

function H1_low_rank_pert = get_H1_low_rank_pert(n, vtX)
    H1_low_rank_pert = ((2*n-2)/(n*n)) * (vtX' * vtX);
end

function H2 = get_H2(n, XtvX)
    H2 = ((n+1)/n) * XtvX ;
end

function u = get_u(d)
    u = randn(d, 1); 
%     u = u / norm(u);
%     u = u / sqrt(d);
end

function diff = get_diff(op, n, d, A, X, m)

    switch op
        case 'plain'
            diff = calc_diff_plain(n, d, A, X, m); % approx hessian value - W cummulative with multiple v
        case 'ricatti'
            diff = calc_diff_ricatti(n, d, A, X, m);
        otherwise
            fprintf('Not supported op \n');
            diff = -1;
    end       
end


function W1_diag = get_W1_diag_part(v_vectors_mat, n)
    W1_diag = ((n-1)/(n*n))*sum(v_vectors_mat.^2, 'all');
end

function H1_approx = get_approx_H1(n, X, v_vectors_mat)
    W1_diag_scalar = get_W1_diag_part(v_vectors_mat, n);
    W1_diag_sqrt_scalar = sqrt(W1_diag_scalar); 
%     W1_low_rank = v_vectors_mat * v_vectors_mat';
    % ricatti
    A_ricatti = W1_diag_sqrt_scalar*eye(n);
    B_ricatti = eye(n);
    C_ricatti = (sqrt(2*n-2)/n) * v_vectors_mat';
    
    params = get_ricatti_params();
    [X_Riemannian, info_Riemannian] =  Riemannian_lowrank_riccati(A_ricatti, B_ricatti, C_ricatti, params);

    Delta = X_Riemannian.Y * X_Riemannian.Y';
    
    sqrt_W1_approx = A_ricatti + Delta;
    Z = sqrt_W1_approx * X;
    % sketching ...
    H1_approx = Z' * Z;
end

function diff = calc_diff_ricatti(n, d, A, X, m)
    c = 12*(n*n) / ((n-1)*(n-2)*(n-3)); % should we insert this coef into the sqrt optimization?
    % Transpose so that rows are samples.
    H2 = 0;
    DAu = 0 ;
    v_vectors_mat = zeros(n, m);
    for i = 1:m
        u = get_u(d);
        DAu = DAu + get_DAu(A,u);
        v = X*u;
        v_vectors_mat(:,i) = v';
        dvX = v .* X;
        XtvX = dvX' * dvX;     
        H2 = H2 + get_H2(n, XtvX);
    end
    
    H1_approx = get_approx_H1(n, X, v_vectors_mat);
    
    H_approx = c * (H1_approx-H2);  % approx hessian value
    H_true = A*DAu*A';              % True hesian value - DAu cummulative with multiple u
    diff = get_norm_diff(H_true, H_approx);
end


function diff = calc_diff_plain(n, d, A, X, m)
    c = 12*(n*n) / ((n-1)*(n-2)*(n-3));
    % Transpose so that rows are samples.
    H1 = 0; H2 = 0;
    DAu = 0 ;
    step2 = tic;
    XtX = X' * X; % also need improve
    fprintf('Runtime of XtX %d \n', toc(step2));
    for i = 1:m
        u = get_u(d);
        DAu = DAu + get_DAu(A,u);
        v = X*u;
        dvX = v .* X;
        XtvX = dvX' * dvX;     
        vtX = v' * X;       
        H1 = H1 + get_H1_diag(n, v, XtX)  +  get_H1_low_rank_pert(n, vtX);
        H2 = H2 + get_H2(n, XtvX);
    end
    H_approx = c * (H1-H2); % approx hessian value - W cummulative with multiple v
    H_true = A*DAu*A'; % True hesian value - DAu cummulative with multiple u
    diff = get_norm_diff(H_true, H_approx);
end

function print_noise(with_noise)
    if with_noise
        fprintf('Noisy version \n');
    else 
        fprintf('Non-Noisy version \n');
    end
end


function plot_graphs(op, m_choices, n_choices, d, A, S)
    ca = cell(1, length(m_choices));
    CM = jet(length(m_choices)); 

    X = (A*S)';
    for j = 1:length(m_choices)
        m = m_choices(j);
        ca{j} = sprintf('m=%d', m);
        values = [];
        for n = n_choices
            curr_X = X(1:n,:);
            curr_diff = get_diff(op, n, d, A, curr_X, m);
            disp(['diff for n=10^', num2str(log10(n)), ' m=', num2str(m), ' is: ', num2str(curr_diff)]);
            values(end+1) = curr_diff;
        end

        hold on;
        plot(n_choices, values, '-', 'Color', CM(j,:), 'LineWidth',2)
        grid on;    
    end

    title('Results','FontSize',16)
    ylabel('Diff - approx. vs. true value in fro norm')
    xlabel('num of samples')
    ax = gca;
    set(gca, 'XScale', 'log')
    ax.XAxisLocation = 'origin';
    set(gca, 'YScale', 'log')
    ax.YAxisLocation = 'origin';
    legend(ca, 'Location', 'southwest')
end

function params = get_ricatti_params()
    params.rmax = 5; % Maximum rank
    params.tol_rel = 1e-6; % Stopping criterion for rank incrementating procedure
    params.tolgradnorm = 1e-10; % Stopping for fixed-rank optimization
    params.maxiter = 100; % Number of iterations for fixed-rank optimization
    params.maxinner = 30; % Number of trust-region subproblems for fixed-rank optimization
    params.verbosity = 2; 1; 2; 0; % Show output
end

% function W = get_W(v, n)
%     v_squred_diag = diag(power(v,2));
%     % divide into H1, H2
%     W = ((n-1)/power(n,2))*trace(v_squred_diag)*eye(n) - ((n+1)/n) * v_squred_diag + ((2*n-2)/power(n,2)) * v*v';
% end
% 
% function W_diag = get_W_diag_part(v, n)
%     v_squared = v.^2;
%     W_diag = ((n-1)/(n*n))*sum(v_squared)*ones(size(v)) - ((n+1)/n) * v_squared;
% end
% 
% function W_low_rank = get_W_low_rank(v, n)
%     W_low_rank = (sqrt(2*n-2)/n) * v;
% end
