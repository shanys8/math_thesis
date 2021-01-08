% This a test file to obtain low-rank solutions of the equation
% A'X + XA + XBB'X = C'C.
% We minimize the residual error ||A'X + XA + XBB'X - C'C||_F ^2 on the
% set of low-rank symmetric positive definite matrices.
% 
% B = 0 implies that we are solving the Lyapunov equations.
%

% Algorithm is based on the work:
% B. Mishra and B. Vandereycken,
% "A Riemannian approach to low-rank algebraic Riccati equations",
% arXiv:1312.4883, 2013.
%
% Paper link: http://arxiv.org/abs/1312.4883.
% Email: b.mishra@ulg.ac.be

clear all; close all; clc;
RandStream.setGlobalStream(RandStream('mt19937ar','seed', 1));  % fixed seed


% % our case for weighted SVD
m = (10)^2;
U_1 = rand(m,1); 
A_1 = diag(rand(m,1));
B_1 = A_1 + U_1*U_1';
target = sqrtm(B_1); % we aim to approx sqrt(B_1)
% % input to algorithm
A = sqrtm(A_1);
B = eye(m);
C = U_1';


% Example 1
% m = (10)^2;
% A = sp_laplace(sqrt(m));
% A = sparse(A);
% B = ones(m,1);
% C = zeros(1, m); C(m) = 1;


% % Example 2
% m = 100;
% Ac = [2.5, -1, zeros(1,m-2)];
% Ar = [2.5, 1, 1, 1, 1, zeros(1, m-5)];
% A = sparse(toeplitz(Ac, Ar));
% B = ones(m,1);
% D = [1; -2]*ones(1, m/2);
% C = (D(:))';


% % Example 3
% m = 50;
% e = ones(m, 1);
% A = -spdiags([e -2*e e], -1:1, m, m);
% B = ones(m,1);
% C = zeros(1, m); C(m) = 1;




%% Riemannian Algorithm
fprintf('Riemannian approach. \n');

% All "params" options not obligatory
%params.rmax = 50; % Maximum rank
params.rmax = 5; % Maximum rank

params.tol_rel = 1e-10; % Stopping criterion for rank incrementating procedure
params.tolgradnorm = 1e-14; % Stopping for fixed-rank optimization
params.maxiter = 100; % Number of iterations for fixed-rank optimization
params.maxinner = 30; % Number of trust-region subproblems for fixed-rank optimization
params.verbosity = 1; 2; 0; % Show output


[X_Riemannian, info_Riemannian] =  Riemannian_lowrank_riccati(A, B, C, params);

% approx = sqrtm(A_1) + X_result;

% compare approx vs target
%fprintf('Compare target and result \n');
%fprintf(norm(target - approx,'fro'));

%% Matlab ARE
% ARE does not extend well to very large dimensional problems
fprintf('Matlab ARE. \n');
if size(A, 1) <= 500,
    Xopt_are = are(-full(A), B*B', full(C'*C));
    [Uopt_are, Sopt_are, Vopt_are] = svd(Xopt_are);
    %         % debug
    %         norm(A'*Xopt_are + Xopt_are*A + Xopt_are*B*B'*Xopt_are - C'*C, 'fro')
    
    info_ARE.residual = [];
    for r = 1: size(X_Riemannian.Y, 2)
        Xopt_are_r = Uopt_are(:, 1:r)*Sopt_are(1:r, 1:r)*(Vopt_are(:, 1:r))';
        residual_opt_r = norm(A'*Xopt_are_r + Xopt_are_r*A + (Xopt_are_r*B)*(B'*Xopt_are_r) - C'*C, 'fro');
        info_ARE.residual = [info_ARE.residual;  residual_opt_r];
    end
end



%% Plots
normC = norm(C,'fro');

figure;
semilogy(info_Riemannian.time_all, sqrt(4*info_Riemannian.cost_all)/normC,'o', 'Color', 'yellow', 'LineWidth', 2,'MarkerSize',15);
ax1 = gca;
set(ax1,'FontSize',20);
xlabel(ax1,'Time (in seconds)','FontSize',20);
ylabel(ax1,'Relative residual norm','FontSize',20);
legend('Riemannian')
title('Low-rank algebraic Riccati equation')


figure;
semilogy(sqrt(4*info_Riemannian.cost_all)/normC,'o', 'Color', 'green', 'LineWidth', 2,'MarkerSize',15);
ax1 = gca;
set(ax1,'FontSize',20);
xlabel(ax1,'Iterations','FontSize',20);
ylabel(ax1,'Relative residual norm','FontSize',20);
legend('Riemannian')
title('Low-rank algebraic Riccati equation')

figure;
semilogy(sqrt(4*info_Riemannian.cost)/normC,'o', 'Color', 'blue', 'LineWidth', 2,'MarkerSize',15);
ax1 = gca;
set(ax1,'FontSize',20);
xlabel(ax1,'Rank','FontSize',20);
ylabel(ax1,'Relative residual norm','FontSize',20);
legend('Riemannian')
title('Low-rank algebraic Riccati equation')

figure;
semilogy(norm((target-A) - (X_Riemannian.Y * X_Riemannian.Y'), 'fro'),'o', 'Color', 'black', 'LineWidth', 2,'MarkerSize',15);
ax1 = gca;
set(ax1,'FontSize',20);
xlabel(ax1,'Rank','FontSize',20);
ylabel(ax1,'residual norm','FontSize',20);
legend('Riemannian')
title('Low-rank algebraic Riccati equation')

if m <= 500,
    figure;
    semilogy(sqrt(4*info_Riemannian.cost)/normC,'o', 'Color', 'red', 'LineWidth', 2,'MarkerSize',15);
    hold on;
    semilogy(info_ARE.residual/normC,'s', 'Color', 'black', 'LineWidth', 2,'MarkerSize',15);
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',20);
    xlabel(ax1,'Rank','FontSize',20);
    ylabel(ax1,'Relative residual norm','FontSize',20);
    legend('Riemannian','Matlab ARE')
    title('Low-rank algebraic Riccati equation')
end