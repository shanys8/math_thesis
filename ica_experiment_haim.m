function ica_experiment
 
randn('seed', 232223);
rand('seed', 34323423);

d = 10;
n = 100000;

A = randn(d, d);   % Mixing matrix

% Generate latent and samples
% no noise for now.
S = sign(randn(d, n));
X = A * S;

% Transpose so that rows are samples.
X = X';

u = randn(d, 1); u = u / norm(u);

% Create Hessian estimator
v = X * u;
XtX = X' * X;
dvX = v .* X;
XtvX = dvX' * dvX;
vtX = v' * X;
H_approx = ((n - 1) * sum(v.^2) * XtX / (n*n) -...
    (n+1) * XtvX / n +...
    (2 * n - 2) * vtX' * vtX / (n * n)) * 12 * (n * n) / ((n - 1) * (n - 2) * (n - 3));

% Compute exact Hessian according to formula
Atu = A' * u;
DAu = diag(24 * Atu .* Atu);
H = A * DAu * A';

disp(norm(H - H_approx, 'fro'));

% Split to H1 and H2
H1 = ((n - 1) * sum(v.^2) * XtX / (n*n) +...
    (2 * n - 2) * vtX' * vtX / (n * n)) * 12 * (n * n) / ((n - 1) * (n - 2) * (n - 3));
H2 = ((n+1) * XtvX / n) * 12 * (n * n) / ((n - 1) * (n - 2) * (n - 3));

% [fval,gradval] = Aobj(A);
% disp(sqrt(fval));
% disp(gradval);
% 
% [fval,gradval] = Aobj(eye(d));
% disp(sqrt(fval));
% disp(gradval);
% 
% options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true);
% problem.options = options;
% problem.x0 = eye(d);
% problem.objective = @Aobj;
% problem.solver = 'fminunc';
% AA = fminunc(problem);
% 
% [fval,gradval] = Aobj(AA);
% disp(sqrt(fval));
% disp(gradval);
% 
% disp(AA);
% rS = AA \ (X(1:10, :)');
% disp(rS);
% disp(S(:, 1:10));
% 
%     function [v, df] = Aobj(A)
%         [fval,gradval] = dlfeval(@Aobj0, dlarray(A));
%         v = extractdata(fval);
%         df = extractdata(gradval);
%     end
% 
%     function [v, df] = Aobj0(A)
%         Atu = A' * u;
%         dAu = 24 * Atu .* Atu;
%         H = A * (dAu .* A');
%         E = H - H_approx;
%         EE = E' * E;
%         v = 0;
%         for i = 1:d
%             v = v + EE(i, i);
%         end
%         df = dlgradient(v, A);
%     end
% end
% 
% 
