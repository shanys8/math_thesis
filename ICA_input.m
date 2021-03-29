clear all; close all; clc;

% ICA input

n = 10;
v = rand(n, 1);
coef = 12*power(n,2) / ((n-1)*(n-2)*(n-3));
v_squred_diag = diag(power(v,2));
A = coef * (((n+1)/n) * v_squred_diag - ((n-1)/power(n,2)) * trace(v_squred_diag) * eye(n));
u_coef = coef * ((2*n-2)/power(n,2));
sqrt_u_coef = sqrt(u_coef);
u = sqrt_u_coef * v;

% send to ricatti as input (A, u) for W = A - uu' as W of weighted SVD
% problem for ICA

u_final = coef * ((-1)*((2*n-2)/power(n,2)) * (v*v'));


