clear all; close all; clc;
RandStream.setGlobalStream(RandStream('mt19937ar','seed', 1));  % fixed seed

eigenvalues_spaced = 'Equidistantly'; % Equidistantly|Logarithmically
if strcmp(eigenvalues_spaced, 'Equidistantly')
mat_file_name = 'A1.mat';
else
mat_file_name = 'A2.mat';
end

% % our case for weighted SVD
n = (10)^2;
% U_1 = rand(n,1);
% A_1 = diag(rand(n,1));
U = load('b.mat').x;
A = load(mat_file_name).x;

W = A + U*U';
d = eig(W);
min_eigenvalue = min(d);



%% Riemannian Algorithm
fprintf('Riemannian approach. \n');
min_rank = 3;
max_rank = 6;
max_rank_arr = linspace(min_rank,max_rank, max_rank-min_rank+1);
approx_arr = [];
approx_power2_arr = [];
approx_arr_fro = [];
approx_power2_arr_fro = [];

for i=1 : length(max_rank_arr)

    %     fprintf(['---------- Max Rank: %s ----------\n' num2str(max_rank_arr(i))]);

    disp(['---------- Max Rank: ',num2str(max_rank_arr(i)),'  ----------'])
    % All "params" options not obligatory
    %params.rmax = 50; % Maximum rank
    params.rmax = max_rank_arr(i); % Maximum rank
    params.tol_rel = 1e-6; % Stopping criterion for rank incrementating procedure
    params.tolgradnorm = 1e-14; % Stopping for fixed-rank optimization
    params.maxiter = 100; % Number of iterations for fixed-rank optimization
    params.maxinner = 30; % Number of trust-region subproblems for fixed-rank optimization
    params.verbosity = 1; 2; 0; % Show output

    [X_Riemannian, info_Riemannian] =  Riemannian_lowrank_riccati(sqrtm(A), eye(n), U', params);

    H = sqrtm(A) + X_Riemannian.Y * X_Riemannian.Y'; % approx of sqrt(W)

    residual_sqrt = sqrtm(W) - H;
    residual = W - H^2;

    approx_arr(i) = norm(residual_sqrt);
    approx_power2_arr(i) = norm(residual);

    approx_arr_fro(i) = norm(residual_sqrt, 'fro');
    approx_power2_arr_fro(i) = norm(residual, 'fro');
    
end



% Plots
f1 = figure;
f2 = figure;
figure(f1);
fprintf('Plot \n');
size_arr = [600, 500, 400, 300, 200, 100];
semilogy(max_rank_arr,approx_arr, 'o-r', 'LineWidth',1.5);
hold on
semilogy(max_rank_arr,approx_power2_arr, 'o-b', 'LineWidth',1.5);
semilogy(max_rank_arr,approx_arr_fro, 'o-g', 'LineWidth',1.5);
% semilogy(max_rank_arr,approx_power2_arr_fro, 'o-m', 'LineWidth',1.5);
% semilogy(max_rank_arr,approx_power2_arr_fro/sqrt(min_eigenvalue), 'o-y', 'LineWidth',1.5);
% semilogy(max_rank_arr,sqrt(approx_power2_arr_fro*sqrt(n)),'o-c', 'LineWidth',1.5);
% legend('Residual sqrt in l2 norm','Residual in l2 norm', 'Residual sqrt in fro norm', 'Residual in fro norm', 'tau/sqrt(min_eigenval)', 'sqrt(sqrt(n)*tau)', 'Location','Best');
legend('Residual sqrt in l2 norm','Residual in l2 norm','Residual sqrt in fro norm');

title([eigenvalues_spaced, ' spaced eigenvalues'])
xlabel('Max rank');
ylabel('Residual');

figure(f2);
semilogy(max_rank_arr,approx_arr, 'o-r', 'LineWidth',1.5);
hold on
semilogy(max_rank_arr,approx_power2_arr_fro/sqrt(min_eigenvalue), '<-g', 'LineWidth',1.5);
semilogy(max_rank_arr,sqrt(approx_power2_arr_fro*sqrt(n)), '>-b', 'LineWidth',1.5);

legend('Residual sqrt in l2 norm','bound - tau/sqrt(min_eigenval)', 'bound - sqrt(sqrt(n)*tau)', 'Location','Best');
title('Bounds')
xlabel('Max rank');
ylabel('Residual');
