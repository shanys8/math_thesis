% function H = cum4hes_approx(X, u, v, XtX)
% H -- Returned hessian matrix.
% X -- Matrix where each column represents a data point.  The data is
%      assumed to be centered (i.e., mean subtracted).
% u -- directional vector.  The Hessian of the u-directional fourth cumulant
%      is taken.
% v -- X*u
% XtX -- optional input argument.  If provided, then the calculation of
%        X'*X can be avoided.  This is used for efficiency purposes when
%        the function is called multiple times.
% Computes and returns an approximation to the hessian with respect to the
% directional variable u of k_4(u'x) where x denotes the random variable
% being sampled.  Here, k_4 is the fourth k-statistic estimate to the
% fourth cumulant.

function H = cum4hes_approx(X, u, v, XtX)
    if ~exist('XtX', 'var')
        XtX = X'*X;
    end
    [n, d] = size(X);

    c = 12*(n*n) / ((n-1)*(n-2)*(n-3));
    dvX = v .* X;
    XtvX = dvX' * dvX;
    vtX = v' * X;
    H1 = get_H1_diag(n, v, XtX)  +  get_H1_low_rank_pert(n, vtX);
    H2 = get_H2(n, v, X, XtvX);
    H = c * (H1-H2);
   
end


function H1_diag = get_H1_diag(n, v, XtX)
    H1_diag = ((n-1)/(n*n)) * sum(v.^2) * XtX;
end

function H1_low_rank_pert = get_H1_low_rank_pert(n, vtX)
    H1_low_rank_pert = ((2*n-2)/(n*n)) * (vtX' * vtX);
end

function H2 = get_H2(n, v, X, XtvX)
    H2 = ((n+1)/n) * XtvX ;
%     Z = sqrt((n+1)/n) * v .* X;
%     % sketching ... Z'S'SZ
%     H2_approx = Z' * Z;
end



        

