clear all; close all; clc;
RandStream.setGlobalStream(RandStream('mt19937ar','seed', 12121));  % fixed seed

n = 100;                    % num of samples
d = 5;                      % dim
alpha = 3;      
D = diag(rand(n, 1));          
t = rand(n, 1);             % target var
k = 2;                      % low rank pert dim
Z = rand(n,k);
C = D + Z * Z';
phi = rand(n,d);            % basis functions - n functions, each function d dimentional

[val, d_alpha, d_D, d_Z] = get_partial_derivatives(n, d, alpha, D, t, Z, phi);

fprintf('Done');


function [val, d_alpha, d_D, d_Z] = get_partial_derivatives(n, d, alpha, D, t, Z, phi)
    
    C = D + Z*Z';
    A = alpha*eye(d) + phi'*(C\phi);
    m_n = A\phi'*(C\t);

    val = (d/2) * log(alpha)...
        - (1/2) * sum(log(eig(C)))...
        - (n/2) * log(2*pi)...
        - (1/2) * sum(log(eig(A)))...    
        - (1/2) * t'*(C\t)...       % E(m)
        + (1/2) * m_n'* A * m_n;    % E(m)
    
    t8 = (C\phi)*(A\phi')*(C\t);
    t9 = (C\phi)*(A\phi')*(C\Z);

    d_alpha = (1/2)*(d/alpha - trace(inv(C)) - (t'*(C\phi)*(A\(A\phi')) * (C\t)));
    d_D = (1/2)*( diag((C\phi)*(A\phi')\C) - diag(inv(C)) + (C\t).*(C\t) - (C\t.*t8) + t8.*t8 - t8 );
    d_Z = (1/2)*( 2*t9 - 2*C\Z + 2*(C\t)*t'*(C\Z) - (C\t)*t'*t9 - t8*(t'*(C\Z)) + 2*t8*t'*t9 - t8*t'*(C\Z) - (C\t)*t'*t9 );
end


         
% dalpha = dlgradient(marginal_likelihood_log, alpha);             
%                         + (1/2) * m_n'* A * m_n;    % E(m)
