clear all; close all; clc;
RandStream.setGlobalStream(RandStream('mt19937ar','seed', 1));  % fixed seed

% ------------------------- START ------------------------- %

% General values
with_noise = false;
d = 20; % dimension
A = randn(d, d); % ICA demixing matrix

print_noise(with_noise)

min_n = 7;
max_n = 7; % 10^7 is the max dim possible to run
min_m = 1;
max_m = 1;
m_steps = 1;
[n_choices, m_choices] = get_params(min_n, max_n, min_m, max_m, m_steps);
% m_choices = 1:10:30;

% Generate latent and samples
S = sign(randn(d, 10^max_n));

plot_graphs(m_choices, n_choices, d, A, S)

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
function diff = get_diff_norm(H_true, H_approx)
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

function diff = get_diff(n, d, A, S, m)

    c = 12*(n*n) / ((n-1)*(n-2)*(n-3));
    % Transpose so that rows are samples.
    step1 = tic;
    X = (A*S)'; % highest runtime
    fprintf('Runtime of X = (A*S)t %d \n', toc(step1));
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
    
    diff = get_diff_norm(H_true, H_approx);
end


function print_noise(with_noise)
    if with_noise
        fprintf('Noisy version \n');
    else 
        fprintf('Non-Noisy version \n');
    end
end


function plot_graphs(m_choices, n_choices, d, A, S)
    ca = cell(1, length(m_choices));
    CM = jet(length(m_choices)); 

    for j = 1:length(m_choices)
        m = m_choices(j);
        ca{j} = sprintf('m=%d', m);
        values = [];
        for n = n_choices
            curr_diff = get_diff(n, d, A, S(:,1:n), m);
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

% multiply diag with X is diag(d).*X
% check W with minus and plus


function [n_choices, m_choices] = get_params(min_n, max_n, min_m, max_m, m_steps)
    % iterate over num of samples (10^val)
    n_choices = logspace(min_n, max_n, max_n-min_n + 1);

    % iterate over num of random vectors u to use
    m_choices = linspace(min_m, max_m, m_steps);
end

function ei = canonvec(i, dim)
%   function ei = canonvec(i, dim)
%   Produces the ith canonical vector in a dim dimensional space.
    ei = zeros(dim, 1);
    ei(i) = 1;
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
