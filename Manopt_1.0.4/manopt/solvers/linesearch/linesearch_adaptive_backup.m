function [stepsize newx storedb lsmem lsstats] = ...
                  linesearch_adaptive_backup(problem, x, d, f0, df0, ...
                                                   options, storedb, lsmem)
% OBSOLETE -- Adaptive line search algorithm
%
% THIS VERSION OF THE FILE linesearch_adaptive WAS DISCONTINUED ON Nov. 7
% 2013. We keep it for a while for future reference if need be. Please use
% linesearch_adaptive instead.



%
% function [stepsize newx storedb lsmem lsstats] = 
%      linesearch_adaptive(problem, x, d, f0, df0, options, storedb, lsmem)
%
% Adaptive linesearch algorithm for descent methods, based on a simple
% backtracking method. On average, this line search intends to do only one
% or two cost evaluations.
%
% Inputs/Outputs : see help for linesearch
%
% See also : linesearch

% This file is part of Manopt: www.manopt.org.
% Original author: Bamdev Mishra, Dec. 30, 2012.
% Contributors: 
% Change log: 
%     Sept. 13, 2013 (NB) : The automatic direction reversal feature was
%                           removed (it triggered when df0 > 0). Direction
%                           reversal is a decision that needs to be made by
%                           the solver, so it can know about it.


    % Backtracking parameters
    contraction_factor = .5;
    suff_decr = 0.5;
    max_ls_steps = 10;
    
    % Initial guess for the step size
    if ~isempty(lsmem)
        initial_stepsize =  lsmem.initial_stepsize;
    else
        % The initial choice of stepsize is necessarily disputable. The
        % optimal step size is invariant under rescaling of the cost
        % function, but df0, on the other hand, will scale like f. Hence,
        % using df0 to guess a stepsize may be considered ill-advised. It
        % is not so if one further assumes that f is "well conditionned".
        % At any rate, to prevent this initial step size guess from
        % exploding, we limit it to 1 (arbitrarily).
        initial_stepsize = min(abs(df0), 1);
    end
    
    % Backtrack
    stepsize = initial_stepsize;
    for iter = 1 : max_ls_steps
        
        % Look further down the line
        newx = problem.M.retr(x, d, stepsize);
        [newf storedb] = getCost(problem, newx, storedb);
        
        % If that point is not satisfactory, reduce the stepsize and retry.
        if newf > f0 + suff_decr*stepsize*df0
            stepsize = contraction_factor * stepsize;
            
        % Otherwise, stop here.
        else
            break;
        end
        
    end
    
    % If we got here without obtaining a decrease, we reject the step.
    if newf > f0
        stepsize = 0;
        newx = x;
    end
    
    % On average we intend to do only one extra cost evaluation
    if iter == 1
        lsmem.initial_stepsize = 2 * initial_stepsize;
    elseif iter == 2
        lsmem.initial_stepsize = stepsize;
    else
        lsmem.initial_stepsize = 2 * stepsize;
    end
    
    stepsize = stepsize*problem.M.norm(x, d);
    
    % Save some statistics also, for possible analysis.
    lsstats.costevals = iter;
    lsstats.stepsize = stepsize;
    
end
