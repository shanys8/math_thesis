function M = symfixedrankYYfactory_riccati(n, k, add_info)
% Manifold of nxn pos. semidef. matrices of rank k tuned to least-squares.
%
% function M = symfixedrankYYfactory(n, k, add_info)
%
%
% The goemetry is based on the paper,
% B. Mishra and B. Vandereycken,
% "A Riemannian approach to low-rank algebraic Riccati equations",
% arXiv:1312.4883, 2013.
% Paper link: http://arxiv.org/abs/1312.4883.
%
% A point X on the manifold is parameterized as YY^T where Y is a matrix of
% size nxk. The matrix Y (nxk) is a full colunn-rank natrix. Hence, we deal
% directly with Y.

% Original author: Bamdev Mishra, Dec. 30, 2013.

    M.name = @() sprintf('YY'' quotient manifold of %dx%d PSD matrices of rank %d tuned to least-squares problem.', n, n, k);
    
    M.dim = @() k*n - k*(k-1)/2;
    
    % Some computations are needed only once per iteration
    function X = prepare(X)
        if ~all(isfield(X,{'M1','M2'}) == 1)
            Y = X.Y;
            X.AAt = add_info.AAt;
            X.AtY = add_info.A'*Y;
            X.AAtY = add_info.A*X.AtY;
            X.YtAAtY = Y'*X.AAtY;
            X.B = add_info.B;
            X.FY = X.B*(X.B'*Y);
            X.YtY = Y'*Y;
            X.YtFY = Y'*X.FY;
            X.YtFYYtY = X.YtFY*X.YtY;
            X.FYYtYYtFY = X.FY*(X.YtFYYtY)';
            X.YtFYYtYYtFY = Y'*X.FYYtYYtFY;
            X.A1Y = X.AAtY + X.FYYtYYtFY;
            X.M1 = X.YtY;
            X.M2 = X.YtAAtY + X.YtFYYtYYtFY;
        end
    end
    
    symm = @(D) .5*(D+D');
    skew = @(D) .5*(D-D');
    
    
    % Metric on the total space
    M.inner = @iproduct;
    function ip = iproduct(X, eta, zeta)
        X = prepare(X);    
        A1eta = (add_info.A*(add_info.A'*eta.Y)) + X.FY*(X.YtY*(X.FY'*eta.Y));
        M1 = X.M1;
        M2 = X.M2;
        ip =  trace((zeta.Y'*A1eta)*M1) + trace((zeta.Y'*eta.Y)*M2);
    end
    
    M.norm = @(X, eta) sqrt(M.inner(X, eta, eta));
    
    M.dist = @(X, Z) error('symfixedrankYYfactory_riccati.dist not implenented yet.');
    
    M.typicaldist = @() 10*k;
    
    % Projection onto the horizontal space
    M.proj = @projection;
    function etaproj = projection(X, eta) % OK
        X = prepare(X);
        Y = X.Y;
        A1Y = X.A1Y;
        M1 = X.M1;
        M2 = X.M2;
        
        RHS = skew((A1Y'*eta.Y)*M1)...
            + skew((Y'*eta.Y)*M2);
        
        % Projection onto the horizontal space
        % M1 Omega M2 + M2 Omega M1 = RHS
        
        Omega  = lyap(M1\M2, - ((M1\RHS)/M1)); % Omega  = lyap(M2\M1, - ((M2\RHS)/M2))
        etaproj.Y = eta.Y - Y*Omega;
        
        %         % Debug
        %         norm(M1*Omega*M2 + M2*Omega*M1 - RHS, 'fro')
        %         vertical.Y = Y*Omega;
        %         inner_product(X,vertical, etaproj)
    end
    
    M.tangent = M.proj;
    M.tangent2ambient = @(X, eta) eta;
    
    % Retraction
    M.retr = @retraction;
    function Xnew = retraction(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        Xnew.Y = X.Y + t*eta.Y;
        Xnew = prepare(Xnew);
    end
    
    % Linear equation
    function [eta] = linear_solve(X, dir)
        % We solve A1*eta*M1 + eta*M2 = dir
        % A1 = AAt + low-rank
        X = prepare(X);
        
        FY = X.FY;
        YtY = X.YtY;
        M1 = X.M1;
        M2 = X.M2;
        [V, ~] = eig(M1, M2);
        M1 = symm(M1);
        M2 = symm(M2);
        d1 = real(diag(V'*M1*V)); % M1 = V*D1*V';
        d2 = real(diag(V'*M2*V)); % M2 = V*D2*V';
        
        dir_tilde = dir.Y*V;
        eta_tilde = zeros(size(dir.Y));
        
        for ii = 1 : k
            updateSparse(add_info.sp_skeleton,(d1(ii)*add_info.entries) + (d2(ii)*add_info.Ivec));
            Q = add_info.sp_skeleton;
            d = dir_tilde(:, ii);
            Qinvd = Q\d;
            
            if add_info.normB ==0,
                % Lyapunov equation
                eta_tilde(:, ii) = Qinvd;
            else
                % Riccati equation
                QinvFY =  Q\FY;
                eta_tilde(:, ii) = Qinvd  - QinvFY*(((YtY\eye(k))/d1(ii) + FY'*QinvFY)\ (QinvFY'*d));
            end
        end
        eta.Y = eta_tilde*V';
    end
    
    M.egrad2rgrad = @egrad2rgrad;
    function rgrad = egrad2rgrad(X, egrad) %OK
        X = prepare(X);
        rgrad = linear_solve(X, egrad);
    end
    
    M.ehess2rhess = @ehess2rhess;
    function rhess = ehess2rhess(X, egrad, ehess, U)
        X = prepare(X);
        
        % Riemmanian gradient
        if ~isfield(egrad, 'rgrad')
            rgrad_struct = egrad2rgrad(X, egrad);
            egrad.rgrad = rgrad_struct; % one exta field in egrad
        end
        rgrad = egrad.rgrad.Y; % matrix, not a structure anymore
        
        % Some matrix computations
        Y = X.Y;
        YtY = X.YtY;
        AAtY = X.AAtY;
        YtFY = X.YtFY;
        M1 = X.M1;
        eta = U.Y; % search direction matrix
        FY = X.FY;  % O(n) computations
        Feta = X.B*(X.B'*eta);
        Frgrad = X.B*(X.B'*rgrad);
        YtFrgrad = FY'*rgrad; % O(r^3) computations
        Yteta = Y'*eta;
        YtFeta =  Y'*Feta;
        Ytrgrad = Y'*rgrad;
        
        % Directional derivative of rgrad
        % O(|AA^T| + n) computations
        
        A1rgrad = (add_info.A*(add_info.A'*rgrad))... % correct
            + X.FY*(YtY*YtFrgrad); % correct
        
        A1dotrgrad = Feta*(YtY*YtFrgrad)... % correct
            + FY*((2*symm(Yteta)*YtFrgrad) + YtY*(Feta'*rgrad));   % correct
        
        M1dot = 2*symm(Yteta); % correct
        M2dot = 2*symm(eta'*AAtY)... % correct
            + 4*(symm(symm(YtFeta)*YtY*YtFY))... % correct
            + 2*(YtFY*symm(Yteta)*YtFY); % correct
        
        T1 = A1dotrgrad*M1... % correct
            + A1rgrad*M1dot... % correct
            + rgrad*M2dot; % correct
        
        
        % Riemannian connection in the Total space using the Koszul formula
        % T1 already computed earlier...
        % O(|AA^T| + n) computations
        
        A1eta = (add_info.A*(add_info.A'*eta)) + X.FY*(YtY*YtFeta);
        
        T2 = (Frgrad*(YtY*YtFeta) + FY*((2*symm(Ytrgrad)*YtFeta) + YtY*(rgrad'*Feta)))*M1... % correct
            + A1eta*(2*symm(Ytrgrad))... % correct
            + eta*(2*symm(rgrad'*AAtY) + 4*symm(symm(rgrad'*FY)*YtY*YtFY)+ 2*YtFY*symm(Ytrgrad)*YtFY); % correct
        
        T3 = Y*(2*symm(rgrad'*A1eta))... % correct
            +Y*(2*symm((FY'*rgrad)*M1*(eta'*FY)))... % correct
            +Frgrad*(M1*(eta'*FY)*YtY)... % correct
            +Feta*(M1*(rgrad'*FY)*YtY)... % correct
            +AAtY*(2*symm(eta'*rgrad))... % correct
            +FY*(4*symm(YtY*YtFY*symm(eta'*rgrad)))... % correct
            +Y*(2*(YtFY*symm(eta'*rgrad)*YtFY)); % correct
        
        % Linear solves
        %         RHS.Y = ehess.Y - T1;
        %         RHS2.Y = (T1 + T2 - T3)/2;
        %         RHS3.Y = RHS.Y + RHS2.Y;
        %         RHS3.Y = ehess.Y - T1 + (T1 + T2 - T3)/2
        RHS3.Y = ehess.Y + (-T1 + T2 - T3)/2;
        rhess = linear_solve(X, RHS3);
        
        % Projection onto the horizontal space
        rhess = M.proj(X, rhess);
        
    end
    
    M.exp = @exponential;
    function Xnew = exponential(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        
        Xnew = retraction(X, eta, t);
        warning('manopt:symfixedrankYYfactory_riccati:exp', ...
            ['Exponential for fixed rank ' ...
            'manifold not implenented yet. Used retraction instead.']);
    end
    
    % Notice that the hash of two equivalent points will be different...
    M.hash = @(X) ['z' hashmd5(X.Y(:))];
    
    M.rand = @random;
    function X = random()
        X.Y = randn(n, k);
        X = prepare(X);
    end
    
    M.randvec = @randomvec;
    function eta = randomvec(X)
        eta.Y = randn(n, k);
        eta = projection(X, eta);
        nrm = M.norm(X, eta);
        eta.Y = eta.Y / nrm;
    end
    
    M.lincomb = @lincomb;
    
    M.zerovec = @(X) struct('Y', zeros(n, k));
    
    M.transp = @(X1, X2, d) projection(X2, d);
    
    M.vec = @(X, U) U.Y(:);
    M.mat = @(X, u) struct('Y', reshape(u, n, k));
    M.vecmatareisometries = @() true;
end


% Linear conbination of tangent vectors
function d = lincomb(X, a1, d1, a2, d2) %#ok<INUSL>
    
    if nargin == 3
        d.Y  = a1*d1.Y;
    elseif nargin == 5
        d.Y = a1*d1.Y + a2*d2.Y;
    else
        error('Bad use of symfixedrankYYfactory_riccati.lincomb.');
    end
    
end





