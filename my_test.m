clear all; close all; clc;
RandStream.setGlobalStream(RandStream('mt19937ar','seed', 1));  % fixed seed



n = 100;

% Load from files
%U = load('b.mat').x;
%A = load(mat_file_name).x;

% % Equidistantly spaced eigenvalues
% A = diag(linspace(0.001, 1000, n));

% whether we want to solve W = A + UUt OR W = A - UUt
global pertubation_sign
pertubation_sign = -1;

% Logarithmic spaced eigenvalues
A = diag(logspace(-3, 3, 100));

U = randn(n,1);
%U = (U / norm(U)) * 10^-3;
U = 0.10 * U / norm(U);

W = A + pertubation_sign * (U*U');
% W = A - U*U';
d = eig(W);
min_eigenvalue = min(d);

ee = eig(pertubation_sign * (sqrtm(W) - sqrtm(A)));



%% Riemannian Algorithm
fprintf('Riemannian approach. \n');
min_rank = 5;
max_rank = 5;
max_rank_arr = linspace(min_rank,max_rank, max_rank-min_rank+1);
approx_arr = [];
approx_power2_arr = [];
approx_arr_fro = [];
approx_power2_arr_fro = [];

for i=1 : length(max_rank_arr)


    disp(['---------- Max Rank: ',num2str(max_rank_arr(i)),'  ----------'])
    % All "params" options not obligatory
    %params.rmax = 50; % Maximum rank
    params.rmax = max_rank_arr(i); % Maximum rank
    params.tol_rel = 1e-6; % Stopping criterion for rank incrementating procedure
    params.tolgradnorm = 1e-10; % Stopping for fixed-rank optimization
    params.maxiter = 100; % Number of iterations for fixed-rank optimization
    params.maxinner = 30; % Number of trust-region subproblems for fixed-rank optimization
    params.verbosity = 2; 1; 2; 0; % Show output
    
    A_ricatti = sqrtm(A);
    B_ricatti = eye(n);
    C_ricatti = U';
    [X_Riemannian, info_Riemannian] =  Riemannian_lowrank_riccati(A_ricatti, B_ricatti, C_ricatti, params);

    Delta = X_Riemannian.Y * X_Riemannian.Y';
    cost = info_Riemannian.cost(end);
    
    H = sqrtm(A) + pertubation_sign * Delta; % approx of sqrt(W)
    residual_sqrt = sqrtm(W) - H;
    residual = W - H^2;
    tau = norm(residual, 'fro');
    approx_arr(i) = norm(residual_sqrt);
    approx_power2_arr(i) = norm(residual);

    approx_arr_fro(i) = norm(residual_sqrt, 'fro');
    approx_power2_arr_fro(i) = tau;
    fprintf('>> tau (residual) = %f \n', tau);
    fprintf('>> Lemma3 / Lemma 4: 0.25*tau^2   = %f \n', 0.25*tau^2);
    fprintf('>> Lemma3 / Lemma 4: Minimal cost = %f \n', cost);
    fprintf('>> residual_sqrt = %f \n', norm(residual_sqrt, 'fro'));
    

    
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
semilogy(max_rank_arr,approx_power2_arr_fro, 'o-m', 'LineWidth',1.5);
semilogy(max_rank_arr,approx_power2_arr_fro/sqrt(min_eigenvalue), 'o-y', 'LineWidth',1.5);
semilogy(max_rank_arr,sqrt(approx_power2_arr_fro*sqrt(n)),'o-c', 'LineWidth',1.5);
legend('Residual sqrt in l2 norm','Residual in l2 norm', 'Residual sqrt in fro norm', 'Residual in fro norm', 'tau/sqrt(min_eigenval)', 'sqrt(sqrt(n)*tau)', 'Location','Best');


title(['Min eigenvalue: ', num2str(min_eigenvalue)]);
xlabel('Max rank');
ylabel('Residual');

figure(f2);
semilogy(max_rank_arr,approx_arr, 'o-r', 'LineWidth',1.5);
hold on
semilogy(max_rank_arr,approx_power2_arr_fro/sqrt(min_eigenvalue), '<-g', 'LineWidth',1.5);
semilogy(max_rank_arr,sqrt(approx_power2_arr_fro*sqrt(n)), '>-b', 'LineWidth',1.5);

legend('Residual sqrt in l2 norm','bound - tau/sqrt(min_eigenval)', 'bound - sqrt(sqrt(n)*tau)', 'Location','Best');
title(['Bounds with min eigenvalue: ', num2str(min_eigenvalue)]);

xlabel('Max rank');
ylabel('Residual');
