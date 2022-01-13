clear all; close all; clc;
RandStream.setGlobalStream(RandStream('mt19937ar','seed', 12121));  % fixed seed

n = 100;                    % num of samples
d = 5;                      % dim
alpha = 1;      
dd = rand(n, 1);          
t = rand(n, 1);             % target var
k = 2;                      % low rank pert dim
Z = rand(n,k);
C = diag(dd) + Z * Z';
phi = rand(n,d);            % basis functions - n functions, each function d dimentional



x = build_vector_from_vars(alpha, dd, Z);
func = @(x) get_marginal_likelihood(n, d, k, x(1), x(2:n+1), reshape(x(n+2:end), n, k), phi, t);
grad_f = @(x) grad_marginal_likelihood_derivatives(x, n, d, k, phi, t);
gradtest(length(x), func, grad_f, x, ones(length(x), 1));


% func = @(alpha) get_marginal_likelihood(n, d, alpha, dd, Z, phi, t);
% grad_f = @(alpha) get_d_alpha(n, d, alpha, dd, Z, phi, t);
% gradtest(1, func, grad_f, alpha, 1);


% func = @(dd) get_marginal_likelihood(n, d, alpha, dd, Z, phi, t);
% grad_f = @(dd) get_d_D(n, d, alpha, dd, Z, phi, t);
% gradtest(length(dd), func, grad_f, dd);




function x = build_vector_from_vars(alpha, dd, Z)
    x = [alpha; dd; reshape(Z,[],1)];
end

function [alpha, dd, Z] = extract_vars_from_vector(x, n, k)
    alpha = x(1);
    dd = x(2:n+1);
    Z = reshape(x(n+2:end), n, k);
end

function derivative_vector = grad_marginal_likelihood_derivatives(x, n, d, k, phi, t)
    [alpha, dd, Z] = extract_vars_from_vector(x, n, k);
    [d_alpha, d_D, d_Z] = get_partial_derivatives(n, d, k, alpha, dd, Z, phi, t);
    derivative_vector = build_vector_from_vars(d_alpha, d_D, d_Z);
end


function [val] = get_marginal_likelihood(n, d, k, alpha, dd, Z, phi, t)
    C = diag(dd) + Z*Z';
    inv_D_dot_Z = (diag(dd.^-1)*Z);
    inv_C_dot_t = diag(dd.^-1)*t - inv_D_dot_Z*((eye(k)+Z'*inv_D_dot_Z)\(inv_D_dot_Z'*t));
    A = alpha*eye(d) + phi'*(C\phi);
    m_n = A\(phi'*inv_C_dot_t);
    log_det_by_lemma = log(prod(dd)) + logdet(eye(k)+Z'*inv_D_dot_Z);

    
    val = (d/2) * log(alpha)...
        - (1/2) * log_det_by_lemma...
        - (n/2) * log(2*pi)...
        - (1/2) * sum(log(eig(A)))...    
        - (1/2) * t'*inv_C_dot_t...       % E(m)
        + (1/2) * m_n'* A * m_n;    % E(m)
end


function [d_alpha, d_D, d_Z] = get_partial_derivatives(n, d, k, alpha, dd, Z, phi, t)
    
    C = diag(dd) + Z*Z';
    A = alpha*eye(d) + phi'*(C\phi);
    
%     inv_C_dinv_C_dot_t_simple_t = C\t;
    inv_D_dot_Z = (diag(dd.^-1)*Z);
    inv_C_dot_t = diag(dd.^-1)*t - inv_D_dot_Z*((eye(k)+Z'*inv_D_dot_Z)\(inv_D_dot_Z'*t));
    
    inv_C_dot_Z = C\Z;
    t4 = (C\phi)*(A\phi');
    t8 = t4*inv_C_dot_t;
    t9 = t4*inv_C_dot_Z;

    d_alpha = (1/2)*(d/alpha - trace(inv(A)) - (t'*(C\phi)*(A\(A\phi')) * (C\t)));
    d_D = (1/2)*( diag(t4/C) - diag(inv(C)) + inv_C_dot_t.*inv_C_dot_t - (inv_C_dot_t.*t8) + t8.*t8 - t8.*inv_C_dot_t);
    d_Z = t9 - inv_C_dot_Z + inv_C_dot_t*t'*inv_C_dot_Z - inv_C_dot_t*t'*t9 - t8*(t'*inv_C_dot_Z) + t8*t'*t9 ;
end


% function [d_alpha] = get_d_alpha(n, d, alpha, dd, Z, phi, t)
%     C = diag(dd) + Z*Z';
%     A = alpha*eye(d) + phi'*(C\phi);
%     d_alpha = (1/2)*(d/alpha - trace(inv(A)) - (t'*(C\phi)*(A\(A\phi')) * (C\t)));
% end
% 
% function [d_D] = get_d_D(n, d, alpha, dd, Z, phi, t)
%     
%     C = diag(dd) + Z*Z';
%     A = alpha*eye(d) + phi'*(C\phi);
%     
%     t4 = (C\phi)*(A\phi');
%     t8 = t4*(C\t);
%     inv_C_dot_t = C\t;
% 
%     d_D = (1/2)*( diag(t4/C) - diag(inv(C)) + inv_C_dot_t.*inv_C_dot_t - (inv_C_dot_t.*t8) + t8.*t8 - t8.*inv_C_dot_t);
% end


