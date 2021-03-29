function[X, infoupdate] =  Riemannian_lowrank_riccati(A, B, C, params)
% Returns low-rank solution to  A'X + XA + XBB'X - C'C  = 0.
%
% function X =  Riemannian_lowrank_riccati(A, B, C, params)
% function [X, infoupdate] =  Riemannian_lowrank_riccati(A, B, C, params)
%
%
% We would like to solve for X, A'X + XA + XBB'X = C'C with appropriate
% dimensions
% 0.25 ||A'X + XA + XBB'X - C'C  ||^2 _F is minimized with minimum rank.
%
% The goemetry is based on the paper:
% B. Mishra and B. Vandereycken,
% "A Riemannian approach to low-rank algebraic Riccati equations",
% arXiv:1312.4883, 2013.
%
% Paper link: http://arxiv.org/abs/1312.4883.

    
% Original author: Bamdev Mishra, Dec. 30, 2013.
% Contr_incbutors:
% Change log:


    if nargin == 0
        warning('No input provided! Going for a standard demo...');
        % Example 1
        m = (10)^2;
        A = sp_laplace(sqrt(m));
        A = sparse(A);
        B = ones(m,1);
        C = zeros(1, m); C(m) = 1;
    end
    
    if nargin < 3 && nargin > 0
        error('Please provide 3 inputs (or none for a demo).');
    end
    
    if nargin < 4
        rmax = 50; % Maximum rank
        tol_rel = 1e-7; % Stopping criterion for rank incrementating procedure
        tolgradnorm = 1e-10; % Stopping for fixed-rank optimization
        maxiter = 100; % Number of iterations for fixed-rank optimization
        maxinner = 30; % Number of trust-region subproblems for fixed-rank optimization
        verbosity = 0; % Show output
    else
        if ~isfield(params,'rmax'); params.rmax = 50; end
        if ~isfield(params,'tol_rel'); params.tol_rel = 1e-6; end
        if ~isfield(params,'tolgradnorm'); params.tolgradnorm = 1e-10; end
        if ~isfield(params,'maxiter'); params.maxiter = 100; end
        if ~isfield(params,'maxinner'); params.verbosity = 30; end
        if ~isfield(params,'verbosity'); params.verbosity  = 0; end
        
        rmax = params.rmax;
        tol_rel = params.tol_rel;
        tolgradnorm = params.tolgradnorm;
        maxiter = params.maxiter;
        maxinner = params.maxinner;
        verbosity = params.verbosity;
    end
    
    
    %% Info field + Sparse computations
    m = size(A, 1);
    A = sparse(A);
    AAt = sparse(A*A');
    normC = norm(C,'fro');
    normB = norm(B, 'fro');
    
    sp_skeleton = 5*AAt;
    [rows, cols, entries] = find(AAt);
    Ivec = rows == cols;
    diagentries = entries(Ivec);
    
    info_input.m = m;
    info_input.A = A;
    info_input.C = C;
    info_input.B = B;
    info_input.normB = normB;
    info_input.normC = normC;
    info_input.tol_rel = tol_rel;
    info_input.AAt = sparse(AAt);
    info_input.sp_skeleton = sp_skeleton;
    info_input.Ivec = Ivec;
    info_input.entries = entries;
    info_input.maxiter = maxiter;
    info_input.maxinner = maxinner;
    info_input.tolgradnorm = tolgradnorm;
    info_input.verbosity = verbosity;
    info_input.diagentries = diagentries; % Diagonal entries of AAt
    
    %% Main algorithm
    
    r_inc = 1; % Increase by rank 1 until tolerance is satisfied
    X0.Y = randn(m, 1);
    r0 = size(X0.Y, 2);
    infoupdate.rank = [];
    infoupdate.cost = [];
    infoupdate.time_all = [];
    infoupdate.cost_all = [];
    
    delta0 = m*r0;
    
    r = r0; % Initial rank
    while r <= rmax
        fprintf('>> Rank %i \n', r);
        
        % Perform fixed-rank optimization
        info_input.r = r;
        info_input.delta0 = delta0;
        [X, dir, eigval, info] = Fixed_rank_YY_riccati(X0, info_input);
        
        % Few information collection
        Y = X.Y;
        
        inforel_residual = [info.rel_residual];
        infocost = [info.cost];
        infotime = [info.time];
        costold = infocost(end);
        infodelta = [info.Delta];
        delta0 = infodelta(end);
        
        
        infoupdate.cost = [infoupdate.cost; costold];
        infoupdate.rank = [infoupdate.rank; r];
        infoupdate.cost_all = [infoupdate.cost_all; infocost'];
        if isempty(infoupdate.time_all); time_shift = 0; else time_shift = infoupdate.time_all(end); end;
        infoupdate.time_all = [infoupdate.time_all; time_shift + (infotime)'];
        
        % Check stopping criterion
        if inforel_residual(end) < tol_rel;
            break;
        end
        
        % Update the rank using a descent direction
        r = r + r_inc;
        Yold = [Y, zeros(m, r_inc)];
        Z = [zeros(m, r- r_inc), dir];
        stepsize = abs(eigval);
        armijo = 0;
        kk = 0;
        while (~armijo) && (kk < 50)
            kk = kk + 1;
            stepsize = stepsize/2;
            Y = Yold + stepsize*Z;
            costnew = cost_computation(Y, r, A, B, C);
            armijo = costold - costnew > 0;
            if armijo && (kk == 1) % Probably an understep
                armijo = 0;
                kk = 0;
                stepsize = 4*stepsize; % This translates to increase by 2 (not 4);
            end
        end
        X0.Y = Y;
    end
end


%% Residual computation
function val= cost_computation(Y, r, A, B, C)
    AtY = A'*Y;
    BtY = B'*Y;
    YtFY = (BtY)'*BtY;
    s = size(C, 1);
    L = [C', AtY, Y];
    [~, Lr] = qr(L, 0);
    
    Mat = [-eye(s), zeros(s, r), zeros(s, r);
        zeros(r,s),  zeros(r,r), eye(r);
        zeros(r,s), eye(r), YtFY];
    val = 0.25*(norm(Lr*Mat*Lr','fro')^2); % Stable computation
end


%% Fixed-rank optimization
function[Xopt, dir, eigval, info] = Fixed_rank_YY_riccati(X0, info_input)
% Fixed-rank optimization of the Riccati residual
% We would like to solve, A'X + XA + XBB'X = C'C; with appropriate
% matrices.
% 0.25 ||A'X + XA + XBB'X - C'C  ||^2 _F is minimized with
% rank(X) = r
% dimenstions and X being symmetric positive semidefinite.
% X is factorized as YY'.

% Original author: Bamdev Mishra, Dec. 30, 2013.
% Contr_incbutors:
% Change log:
    
    %% Info to familiar notations
    m = info_input.m;
    r = info_input.r;
    A = info_input.A;
    AAt = info_input.AAt;
    B = info_input.B;
    C = info_input.C;
    normB = info_input.normB;
    normC = info_input.normC;
    delta0 = info_input.delta0;
    tol_rel = info_input.tol_rel;
    sp_skeleton = info_input.sp_skeleton;
    Ivec = info_input.Ivec;
    entries = info_input.entries;
    diagentries = info_input.diagentries;
    
    %% Call the preconditioned geometry file
    add_info.AAt = sparse(AAt);
    add_info.B = B;
    add_info.A = A;
    add_info.normB = normB;
    add_info.sp_skeleton = sp_skeleton;
    add_info.Ivec = Ivec;
    add_info.entries = entries;
    add_info.diagentries = diagentries;
    
    
    problem.M = symfixedrankYYfactory_riccati(m, r, add_info);
    
    %% Storing for chaching some already computed values
    function store = prepare_store(X, store)
        if ~all(isfield(store,{'AtY', 'YtFY','YtAAtY'}) == 1)
            Y = X.Y;
            AY = A*Y;
            AtY = A'*Y;
            AAtY = A*AtY;
            FY = B*(B'*Y);
            YtFY = Y'*FY;
            YtY = Y'*Y;
            YtAY = Y'*AY;
            YtAAtY = Y'*AAtY;
            CY = C*Y;
            
            store.AY = AY;
            store.AtY = AtY;
            store.AAtY = AAtY;
            store.FY = FY;
            store.YtFY = YtFY;
            store.YtY = YtY;
            store.YtAY = YtAY;
            store.YtAtY = YtAY';
            store.YtAAtY =YtAAtY ;
            store.CY = CY;
            rnames = cell(r,1);
            for i=1:r
                rnames{i} = ['r' num2str(i)];
            end
            store.rnames = rnames;
        end
    end
    
    %% Cost function
    problem.cost = @cost;
    function [val, store] = cost(X, store)
        Y = X.Y;
        store = prepare_store(X, store);
        AtY = store.AtY;
        YtFY = store.YtFY;
        
        % Computing residual by taking the structure into account
        s = size(C, 1);
        L = [C', AtY, Y];
        [~, Lr] = qr(L, 0);
        
        Mat = [-eye(s), zeros(s, r), zeros(s, r);
            zeros(r,s),  zeros(r,r), eye(r);
            zeros(r,s), eye(r), YtFY];
        val = 0.25*(norm(Lr*Mat*Lr','fro')^2); % Stable computation
    end
    
    %% Euclidean gradient
    problem.grad = @grad;
    function [rgrad, store] = grad(X, store)
        Y = X.Y;
        store = prepare_store(X, store);
        AtY = store.AtY;
        YtY = store.YtY;
        YtAY = store.YtAY;
        YtFY = store.YtFY;
        CY = store.CY;
        YtAtY = YtAY';
        YtAAtY = store.YtAAtY;
        FY = store.FY;
        
        % Looks ugly here but should work out nicely on paper
        SY = AtY*YtY + Y*YtAY  + Y*(YtFY*YtY) - C'*CY;
        ASY = A*SY;
        SAtY = AtY*YtAtY + Y*YtAAtY  + Y*(YtFY*YtAtY) - C'*(C*AtY);
        SYYtFY = SY*YtFY;
        FYYtSY = FY*(Y'*SY);
        PY = ASY + SAtY + SYYtFY + FYYtSY;
        
        egrad.Y = PY;
        [rgrad, store]= egrad2rgrad_local(X, egrad, store);
        store.egrad = egrad; % for subsequent inner iterations
        store.rgrad = rgrad; % for subsequent inner iterations
    end
    
    
    %% Euclidean Hessian
    problem.hess = @hess;
    function [rhess, store] = hess(X, U, store)
        Y = X.Y;
        eta = U.Y;
        
        % Caching computed information
        store = prepare_store(X, store);
        AAtY = store.AAtY;
        AY = store.AY;
        AtY = store.AtY;
        YtY = store.YtY;
        YtAY = store.YtAY;
        YtFY = store.YtFY;
        CY = store.CY;
        YtAtY = YtAY';
        YtAAtY = store.YtAAtY;
        FY = store.FY;
        
        % Some extra computations for computing the Hessian
        Ateta = A'*eta;
        Aeta = A*eta;
        Yteta = Y'*eta;
        Ceta = C*eta;
        
        etatAY = eta'*AY;
        etatY = eta'*Y;
        Feta = B*(B'*eta);
        
        % Systematica derivation of the directional derivative of the
        % gradient
        % Looks ugly here but should work out nicely on paper
        SY = AtY*YtY + Y*YtAY  + Y*(YtFY*YtY) - C'*CY; % correct
        
        Seta = AtY*Yteta + Y*(Y'*Aeta)  + Y*(YtFY*Yteta) - C'*Ceta; % correct
        
        SdotY = AtY*(etatY) + Ateta*YtY... % correct
            + Y*(etatAY) + eta*(YtAY)... % correct
            + Y*(eta'*FY)*YtY + eta*(YtFY*YtY)... % correct
            + Y*(YtFY*(etatY)) + Y*((Y'*Feta)*YtY); % correct
        
        SAteta = AtY*(Y'*Ateta) + Y*(AtY'*Ateta)  + Y*YtFY*(Y'*Ateta) - C'*(C*Ateta);
        
        SdotAtY = AtY*(eta'*AtY) + Ateta*YtAtY... % correct
            + Y*(eta'*AAtY) + eta*(Y'*AAtY)... % correct
            + Y*((eta'*FY)*YtAtY) + eta*(YtFY)*(YtAtY)... % correct
            + Y*(YtFY*(eta'*AtY)) + Y*((Y'*Feta)*(YtAtY)); % correct
        
        PdotY = A*SdotY...
            + SdotAtY...
            + SdotY*YtFY + SY*(eta'*FY) + Seta*(YtFY)...
            + FY*((SdotY'*Y) + (Seta'*Y)) + Feta*(SY'*Y) ;
        
        Peta = A*Seta...
            + SAteta...
            + SY*(Y'*Feta)...
            + FY*(Y'*Seta);
        
        ehess.Y = PdotY + Peta;
        
        if ~isfield(store, 'egrad')
            ASY = A*SY;
            SAtY = AtY*YtAtY + Y*YtAAtY  + Y*(YtFY*YtAtY) - C'*(C*AtY);
            SYYtFY = SY*YtFY;
            FYYtSY = FY*(Y'*SY);
            store.egrad.Y = ASY + SAtY + SYYtFY + FYYtSY;
            store.rgrad = problem.M.egrad2rgrad(X, store.egrad);
        end
        egrad = store.egrad;
        egrad.rgrad = store.rgrad; % An extra field, a small tr_incck
        [rhess, store] = ehess2rhess_local(X, egrad, ehess, U, store);
    end
    
    
    
    %% Converting the gradient and Hessian to their Riemannian counterparts
    
    % To Riemannian gradient
    function [rgrad, store] = egrad2rgrad_local(X, egrad, store) %OK
        X = prepare(X);
        [rgrad, store] = linear_solve(X, egrad, store); % Compute the linear solution
    end
    
    % To Riemannian Hessian
    function [rhess, store] = ehess2rhess_local(X, egrad, ehess, U, store)
        X = prepare(X);
        
        % Riemmanian gradient
        if ~isfield(egrad, 'rgrad')
            rgrad_struct = egrad2rgrad(X, egrad);
            egrad.rgrad = rgrad_struct; % one exta field in egrad
        end
        rgrad = egrad.rgrad.Y; % matr_incx, not a structure anymore
        
        
        % Some matrix computations
        Y = X.Y;
        YtY = X.YtY;
        AAtY = X.AAtY;
        YtFY = X.YtFY;
        M1 = X.M1;
        
        eta = U.Y; % search direction matr_incx
        
        FY = X.FY;  % O(n) computations
        Feta = X.B*(X.B'*eta);
        Frgrad = X.B*(X.B'*rgrad);
        
        YtFrgrad = FY'*rgrad; % O(r^3) computations
        Yteta = Y'*eta;
        YtFeta =  Y'*Feta;
        Ytrgrad = Y'*rgrad;
        
        
        % Directional der_incvative of rgrad
        % O(|AA^T| + n) computations
        
        A1rgrad = (A*(A'*rgrad))... % correct
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
        
        A1eta = (A*(A'*eta)) + X.FY*(YtY*YtFeta);
        
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
        RHS3.Y = ehess.Y + (-T1 + T2 - T3)/2;
        [rhess, store] = linear_solve(X, RHS3, store); % Compute the linear solution
        
        % Projection onto the hor_inczontal space
        rhess = problem.M.proj(X, rhess);
    end
    
    %% For caching
    function X = prepare(X)
        if ~all(isfield(X,{'M1','M2'}) == 1)
            Y = X.Y;
            X.AAt = add_info.AAt;
            X.AtY = A'*Y;
            X.AAtY = A*X.AtY;
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
    
    %% Computing the linear solution
    function [eta, store] = linear_solve(X, dir, store)
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
        
        for ii = 1 : r
            
            % *** Computationally most expensive step is solving the linear system
            % ***
            d = dir_tilde(:, ii);
            if m <= 10000
                % The sparse structure of Q becomes critical
                updateSparse(add_info.sp_skeleton,(d1(ii)*add_info.entries) + (d2(ii)*add_info.Ivec));
                Q = add_info.sp_skeleton;
                Qinvd = Q\d;
            else
                % Use an iterative solver
                updateSparse(add_info.sp_skeleton,(d1(ii)*add_info.entries) + (d2(ii)*add_info.Ivec));
                Q = add_info.sp_skeleton;
                alpha = 0.1; % Should be tuned (try a smaller value) when it breaks down!
                L = ichol(Q,...
                    struct('michol','on','diagcomp',alpha));% Method 3
                Qinvd = use_iterative_solver(d, d1(ii), d2(ii), L, Q);
            end
            % ***
            % ***
            
            if add_info.normB ==0,
                % Lyapunov equation
                eta_tilde(:, ii) = Qinvd;
            else
                % Riccati equation
                if ~isfield(store, 'QinvFYmat')
                    rnames = store.rnames;
                    for jj = 1:r
                        store.QinvFYmat.(rnames{jj}) = [];
                    end
                end
                rnames = store.rnames;
                if isempty(store.QinvFYmat.(rnames{ii})),
                    if m <= 10000
                        % The sparse structure of Q becomes critical
                        store.QinvFYmat.(rnames{ii}) = Q\FY;
                    else
                        % Use an iterative solver
                        store.QinvFYmat.(rnames{ii}) = use_iterative_solver(FY, d1(ii), d2(ii), L, Q);
                    end
                end
                QinvFY = store.QinvFYmat.(rnames{ii});
                
                %                 %debug
                %                 norm(Q\FY - store.QinvFYmat.(rnames{ii}),'fro')
                
                eta_tilde(:, ii) = Qinvd  - QinvFY*(((YtY\eye(r))/d1(ii) + FY'*QinvFY)\ (QinvFY'*d));
            end
        end
        eta.Y = eta_tilde*V';
    end
    
    % Solving "A\b" for very large matrices, greater than 5000 \times 5000
    function Y_sol = use_iterative_solver(D, d1, d2, L, Q)
        % We intend to solve (d1*AAt + d2*I)Y_sol = D
        % This can be solved by "\" but for larger matrices, iterative
        % solvers may be required.
        % L is the incomplete cholesky factorization of Q = (d1*AAt + d2*I).
        Y_sol = D;
        for kk = 1:size(D, 2)
            [Y_sol(:,kk), ~] = pcg(Q, D(:,kk), 1e-6, 25, L, L');
        end
    end
    
    symm = @(D) .5*(D+D');
    
    %% Call the trust-region algorithm
    options.statsfun = @statsfun;
    function stats = statsfun(problem, X, stats)
        stats.dist  = sqrt(4*stats.cost);
        stats.rel_residual  = sqrt(4*stats.cost)/normC;
    end
    
    
    options.stopfun = @stopfun;
    function stop = stopfun(problem, X, info, last)
        inforel_residual = [info.rel_residual];
        stop = inforel_residual(end)< tol_rel;
    end
    
    
    % Options (not mandatory) for fixed-rank optimization
    options.maxiter = info_input.maxiter;
    options.maxinner = info_input.maxinner;
    options.tolgradnorm = info_input.tolgradnorm;
    options.verbosity = info_input.verbosity;
    
    options.maxtime = Inf;
    options.Delta_bar = (2^20)*delta0;
    options.Delta0 = delta0;
    options.storedepth = 25;
    options.rho_regularization = 1e3;
    options.theta = 0;
    % Pick an algorithm to solve the problem
    checkhessian(problem);
    [Xopt, ~, info] = trustregions(problem, X0, options);
    
    
    % Few fields
    if ~all(isfield(Xopt,{'YtFY'}) == 1)
        Xopt.YtY = Xopt.Y'*Xopt.Y;
        Xopt.FY = B*(B'*Xopt.Y);
        Xopt.AtY = A'*Y;
        Xopt.YtFY = Xopt.Y'*Xopt.FY;
    end
    
    %% Computing the descent direction
    [dir, eigval] = compute_descent(Xopt);
    function [dir, eigval] = compute_descent(X)
        % We exploit the fact that the gradient is low-rank
        Y = X.Y;
        AtY = X.AtY;
        FY = X.FY;
        YtFY = X.YtFY;
        s = size(C, 1);
        
        % Relevant manipulation
        % Looks ugly here but should work out nicely on paper
        
        L = [C', AtY, Y];
        [Lq, Lr] = qr(L, 0);
        Mat = [-eye(s), zeros(s, r), zeros(s, r);
            zeros(r,s),  zeros(r,r), eye(r);
            zeros(r,s), eye(r), YtFY];
        K = A*Lq + FY*(Y'*Lq);
        [Pq,  Pr] = qr([Lq, K], 0);
        Mat1 = Lr*Mat*Lr';
        P_middle = [zeros(size(Mat1)), Mat1;
            Mat1, zeros(size(Mat1))];
        [Pq_small, Ps] = eig(Pr*P_middle*Pr');
        Ps = diag(Ps);
        [Ps, idx] = sort(Ps,'ascend');
        Pq = Pq*Pq_small;
        Pq = Pq(:,idx);
        
        % Eigen direction and Eigenvalue
        eigval = real(Ps(1));
        dir = Pq(:, 1);
    end
    
end