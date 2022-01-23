clear all; close all; clc;
% RandStream.setGlobalStream(RandStream('mt19937ar','seed', 12121));  % fixed seed

n = 100;                    % num of samples
k = 1;                      % low rank pert dim - only works for k = 1
Z = randn(n,k);
Z = Z/norm(Z);

diagonal_dist = 'uniform'; % 'uniform'|'logspace'
dd = get_diag_vector(n, diagonal_dist);
A = diag(dd);
D = Z*Z';
B = A + D;

eigenvalues_num = 10;

delta = sqrtm(B) - sqrtm(A);
eigs_list = sort(eig(delta), 'descend');
max_eig_delta = eigs_list(1);
% A = arrayfun(@(x) calc_bound(x),eigs_list);
 
min_eig_A = min(eig(A));

bounds = [];

for i=1:eigenvalues_num
    if i == 1
        bounds(i) = max_eig_delta;
    else
        kappa = (2*(sqrt(norm(A)+norm(D))+sqrt(min_eig_A)/2)) /  sqrt(min_eig_A);
        bounds(i) = 4 * max_eig_delta * (exp((pi*pi)/(2*log(4*kappa))))^(-2*(i-1));
    end
end

plot_graph(eigenvalues_num, eigs_list(1:10), bounds)

fprintf('\nDone');


%% methods %%

function plot_graph(eigenvalues_num, eigs_list, bounds)
    hold on;
    plot(1:eigenvalues_num, bounds, '-o', 'Color', 'red', 'LineWidth',2, 'DisplayName','upper bound')
    grid on;   
    
    hold on;
    plot(1:eigenvalues_num, eigs_list, '->', 'Color', 'blue', 'LineWidth',2, 'DisplayName','delta eigenvalue')
    grid on;    

    title('Delta eigenvalues bounds', 'FontSize',16);
    xlabel('eigenvalue index')
    ax = gca;
    ax.XAxisLocation = 'origin';
    set(gca, 'YScale', 'log')
    ax.YAxisLocation = 'origin';
    legend;
end


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