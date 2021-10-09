clear all; close all; clc;

% ICA input
clear all; close all; clc;
RandStream.setGlobalStream(RandStream('mt19937ar','seed', 1));  % fixed seed

n = 100;

u = randn(n, 1);
X = 
v = X*u;


coef = 12*power(n,2) / ((n-1)*(n-2)*(n-3));
v_squred_diag = diag(power(v,2));
A = coef * (((n+1)/n) * v_squred_diag - ((n-1)/power(n,2)) * trace(v_squred_diag) * eye(n));
u_coef = coef * ((2*n-2)/power(n,2));
sqrt_u_coef = sqrt(u_coef);
U = sqrt_u_coef * v;
UUt = coef * ((-1)*((2*n-2)/power(n,2)) * (v*v'));

% send to ricatti as input (sqrtm(A), U) for W = A - UU' as W of weighted SVD
% problem for ICA

params.rmax = 5; % Maximum rank
params.tol_rel = 1e-6; % Stopping criterion for rank incrementating procedure
params.tolgradnorm = 1e-10; % Stopping for fixed-rank optimization
params.maxiter = 100; % Number of iterations for fixed-rank optimization
params.maxinner = 30; % Number of trust-region subproblems for fixed-rank optimization
params.verbosity = 2; 1; 2; 0; % Show output

A_ricatti = sqrtm(A);
B_ricatti = eye(n);
C_ricatti = U;
[X_Riemannian, info_Riemannian] =  Riemannian_lowrank_riccati(A_ricatti, B_ricatti, C_ricatti, params);

Delta = X_Riemannian.Y * X_Riemannian.Y';
cost = info_Riemannian.cost(end);

H = sqrtm(A) - Delta; % approx of sqrt(W)
residual_sqrt = sqrtm(W) - H;
residual = W - H^2;
tau = norm(residual, 'fro');
residual_sqrt_norm = norm(residual_sqrt);
residual_norm = norm(residual);
residual_sqrt_fro_norm = norm(residual_sqrt, 'fro');

fprintf('>> tau (residual norm) = %f \n', tau);
fprintf('>> Lemma3 / Lemma 4: 0.25*tau^2   = %f \n', 0.25*tau^2);
fprintf('>> Lemma3 / Lemma 4: Minimal cost = %f \n', cost);
fprintf('>> residual_sqrt = %f \n', residual_sqrt_fro_norm);
