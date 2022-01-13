clear all; close all; clc;
RandStream.setGlobalStream(RandStream('mt19937ar','seed', 12121));  % fixed seed

n = 100;                    % num of samples
d = 5;                      % dim
alpha = 1;      
beta = 3;          
t = rand(n, 1);             % target var
k = 2;                      % low rank pert dim
Z = rand(n,k);
C = (1/beta)*eye(n) + Z * Z';
phi = rand(n,d);            % basis functions - n functions, each function d dimentional



x = build_vector_from_vars(alpha, beta, Z);
func = @(x) get_marginal_likelihood(n, d, k, x(1), x(2), reshape(x(3:end), n, k), phi, t);
grad_f = @(x) grad_marginal_likelihood_derivatives(x, n, d, k, phi, t);
gradtest(length(x), func, grad_f, x, ones(length(x), 1));


% func = @(alpha) get_marginal_likelihood(n, d, k, alpha, beta, Z, phi, t);
% grad_f = @(alpha) get_d_alpha(n, d, alpha, beta, Z, phi, t);
% gradtest(1, func, grad_f, alpha, 1);

% func = @(beta) get_marginal_likelihood(n, d, k, alpha, beta, Z, phi, t);
% grad_f = @(beta) get_d_beta(n, d, alpha, beta, Z, phi, t);
% gradtest(1, func, grad_f, beta, 1);

% vec_Z = reshape(Z,[],1);
% func = @(vec_Z) get_marginal_likelihood_with_vector_Z(n, d, alpha, beta, vec_Z, phi, t, k);
% grad_f = @(vec_Z) get_d_Z(n, d, alpha, beta, vec_Z, phi, t, k);
% gradtest(length(vec_Z), func, grad_f);


function x = build_vector_from_vars(alpha, beta, Z)
    x = [alpha; beta; reshape(Z,[],1)];
end

function [alpha, beta, Z] = extract_vars_from_vector(x, n, k)
    alpha = x(1);
    beta = x(2);
    Z = reshape(x(3:end), n, k);
end

function derivative_vector = grad_marginal_likelihood_derivatives(x, n, d, k, phi, t)
    [alpha, beta, Z] = extract_vars_from_vector(x, n, k);
    [d_alpha, d_beta, d_Z] = get_partial_derivatives(n, d, k, alpha, beta, Z, phi, t);
    derivative_vector = build_vector_from_vars(d_alpha, d_beta, d_Z);
end


% C\t inv(C)*t - > improve inv(C) multiply by woodbery D*t + U(U'*t)
% logdet -> replace with matrix determinant lemma - works for C, check for
% A


function [val] = get_marginal_likelihood(n, d, k, alpha, beta, Z, phi, t)
    C = (1/beta)*eye(n) + Z*Z';
%     inv_C_dot_t_simple = C\t;
    inv_C_dot_t = beta*t - (beta^2) * (Z*((eye(k)+beta*(Z'*Z))\(Z'*t))); %% woodbury
    A = alpha*eye(d) + phi'*(C\phi);
%     log_det_simple = sum(log(eig(C)));
    log_det_by_lemma = -n*log(beta) + logdet(eye(k)+beta*(Z'*Z));
    
    m_n = A\(phi'*inv_C_dot_t);
    val = (d/2) * log(alpha)...
        - (1/2) * log_det_by_lemma...
        - (n/2) * log(2*pi)...
        - (1/2) * sum(log(eig(A)))...    
        - (1/2) * t'*inv_C_dot_t...       % E(m)
        + (1/2) * m_n'* A * m_n;    % E(m)
end


function [d_alpha, d_beta, d_Z] = get_partial_derivatives(n, d, k, alpha, beta, Z, phi, t)
    
    C = (1/beta)*eye(n) + Z*Z';
    A = alpha*eye(d) + phi'*(C\phi);
    
%     inv_C_dot_t = C\t;
    inv_C_dot_t = beta*t - (beta^2) * (Z*((eye(k)+beta*(Z'*Z))\(Z'*t))); %% woodbury
    inv_C_dot_Z = C\Z;
    t4 = (C\phi)*(A\phi');
    t8 = t4*inv_C_dot_t;
    t9 = t4*inv_C_dot_Z;

    d_alpha = (1/2)*(d/alpha - trace(inv(A)) - (t'*(C\phi)*(A\(A\phi')) * (C\t)));
    d_beta = (1/(2*beta^2))*( trace(inv(C)) - trace(t4/C) -  inv_C_dot_t'*inv_C_dot_t + inv_C_dot_t'*t8 - (t'*(t4/C))*(t'*(t4/C))' + t'*(t4/C)*inv_C_dot_t);
    d_Z = t9 - inv_C_dot_Z + inv_C_dot_t*t'*inv_C_dot_Z - inv_C_dot_t*t'*t9 - t8*(t'*inv_C_dot_Z) + t8*t'*t9 ;
end


% function [d_alpha] = get_d_alpha(n, d, alpha, beta, Z, phi, t)
%     C = (1/beta)*eye(n) + Z*Z';
%     A = alpha*eye(d) + phi'*(C\phi);
%     d_alpha = (1/2)*(d/alpha - trace(inv(A)) - (t'*(C\phi)*(A\(A\phi')) * (C\t)));
% end
% 
% function [d_beta] = get_d_beta(n, d, alpha, beta, Z, phi, t)
%     
%     C = (1/beta)*eye(n) + Z*Z';
%     A = alpha*eye(d) + phi'*(C\phi);
%     
%     inv_C_dot_t = C\t;
%     t4 = (C\phi)*(A\phi');
%     t8 = t4*inv_C_dot_t;
% 
%     d_beta = (1/(2*beta^2))*( trace(inv(C)) - trace(t4/C)...
%         -  inv_C_dot_t'*inv_C_dot_t...
%         + inv_C_dot_t'*t8...
%         - (t'*(t4/C))*(t'*(t4/C))'...
%         + t'*(t4/C)*inv_C_dot_t);
% end
% 
% function [d_Z] = get_d_Z(n, d, alpha, beta, vec_Z, phi, t, k)
%     
%     Z = reshape(vec_Z, n, k);
%     C = (1/beta)*eye(n) + Z*Z';
%     A = alpha*eye(d) + phi'*(C\phi);
%     
%     inv_C_dot_t = C\t;
%     inv_C_dot_Z = C\Z;
%     t4 = (C\phi)*(A\phi');
%     t8 = t4*inv_C_dot_t;
%     t9 = t4*inv_C_dot_Z;
% 
%     res = t9 - inv_C_dot_Z + inv_C_dot_t*t'*inv_C_dot_Z - inv_C_dot_t*t'*t9 - t8*(t'*inv_C_dot_Z) + t8*t'*t9 ;
%     
%     d_Z = reshape(res,[],1);
% end
% function [val] = get_marginal_likelihood_with_vector_Z(n, d, alpha, beta, vec_Z, phi, t, k)
%     Z = reshape(vec_Z, n, k);
%     
%     C = (1/beta)*eye(n) + Z*Z';
%     A = alpha*eye(d) + phi'*(C\phi);
%     m_n = A\(phi'*(C\t));
%     val = (d/2) * log(alpha)...
%         - (1/2) * sum(log(eig(C)))...
%         - (n/2) * log(2*pi)...
%         - (1/2) * sum(log(eig(A)))...    
%         - (1/2) * t'*(C\t)...       % E(m)
%         + (1/2) * m_n'* A * m_n;    % E(m)
% end
