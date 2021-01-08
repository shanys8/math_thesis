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
% U_1 = rand(m,1);
% A_1 = diag(rand(m,1));
U_test = load('b.mat').x;
A_test = load(mat_file_name).x;

B_test = A_test + U_test*U_test';
target = sqrtm(B_test); % we aim to approx sqrt(B_1)
d = eig(B_test)
min_eigenvalue = min(d)
% % input to algorithm
A = sqrtm(A_test);
B = eye(n);
C = U_test';


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

[X_Riemannian, info_Riemannian] =  Riemannian_lowrank_riccati(A, B, C, params);


approx_arr(i) = norm(target-(A + X_Riemannian.Y * X_Riemannian.Y'));
approx_power2_arr(i) = norm(B_test - (A + X_Riemannian.Y * X_Riemannian.Y')^2 );

approx_arr_fro(i) = norm((target-A) - (X_Riemannian.Y * X_Riemannian.Y'), 'fro');
approx_power2_arr_fro(i) = norm(B_test - (A + X_Riemannian.Y * X_Riemannian.Y')^2 , 'fro');
end



%% Plots
fprintf('Plot \n');
size_arr = [600, 500, 400, 300, 200, 100];
scatter(max_rank_arr,approx_arr,size_arr(1), 'red', 'o', 'LineWidth',1.5);
hold on
scatter(max_rank_arr,approx_power2_arr,size_arr(2), 'blue', 'o', 'LineWidth',1.5);
scatter(max_rank_arr,approx_arr_fro,size_arr(3), 'green', 'o', 'LineWidth',1.5);
scatter(max_rank_arr,approx_power2_arr_fro,size_arr(4), 'magenta', 'o', 'LineWidth',1.5);
scatter(max_rank_arr,approx_power2_arr_fro/sqrt(min_eigenvalue),size_arr(5), 'yellow', 'o', 'LineWidth',1.5);
scatter(max_rank_arr,sqrt(approx_power2_arr_fro*sqrt(n)),size_arr(6), 'cyan', 'o', 'LineWidth',1.5);

legend('Residual sqrt in l2 norm','Residual in l2 norm', 'Residual sqrt in fro norm', 'Residual in fro norm', 'tau/sqrt(min_eigenval)', 'sqrt(sqrt(n)*tau)', 'Location','Best');
title([eigenvalues_spaced, ' spaced eigenvalues'])
xlabel('Max rank');
ylabel('Residual');

