function gradtest(n, f, gradf, x, d)
% gradtest(n, f, df, x, d)
%
% Tester for gradient implementation. Shamelessly based on 
% checkgradient in manopt.
% Parameters:
%    n - dimensions of the input to the functin.
%    f - handle for the function. If gradf is [] then f
%        should return both f and the gradient. 
%    gradf - handle for the gradient of the function. Ignored
%            if [].
%    x (optional) - point to test.
%    d (optional) - direction to use.
%
% Example use for correct gradient:
%   A = rand(10); A = A' * A;
%   f = @(x) 0.5 * x' * A * x;
%   gradf = @(x) A * x;
%   gradtest(10, f, gradf);

% Example use for bad gradient:
%   A = rand(10); A = A' * A;
%   f = @(x) 0.5 * x' * A * x;
%   gradf = @(x) x;
%   gradtest(10, f, gradf);

if (nargin < 4)
    x = randn(n, 1);
end

if (nargin < 5)
    d = randn(size(x)); d = d / norm(d);
end

if ~isequal(size(d), size(x))
    error('d and x should have same dimension');
end

% The model is compared to the actual value at locations
% x + h * d where h varies from very small to 1 (in logarithmic 
% spacing)
h = logspace(-8, 0, 51);

% Compute model values
if isempty(gradf)
    [ff, gf] = f(x);
else
    ff = f(x);
    gf = gradf(x);
end
df = sum(gf(:) .* d(:));
model = polyval([df ff], h);

% Compute actual values
for i = 1:length(h)
    xx = x + h(i) * d;
    val(i) = f(xx);
end

err = abs(model - val);

% plot error
loglog(h, err);
title(sprintf(['Directional derivative check.\nThe slope of the '...
               'continuous line should match that of the dashed\n'...
               '(reference) line over at least a few orders of '...
               'magnitude for h.']));
xlabel('h');
ylabel('Approximation error');
    
line('xdata', [1e-8 1e0], 'ydata', [1e-8 1e8], ...
     'color', 'k', 'LineStyle', '--', ...
      'YLimInclude', 'off', 'XLimInclude', 'off');
  
% Now identify & plot the linear part
[range, poly] = identify_linear_piece(log10(h), log10(err), 10);
hold all;
loglog(h(range), 10.^polyval(poly, log10(h(range))), 'LineWidth', 3);
hold off;
fprintf('The slope should be 2. It appears to be: %g.\n', poly(1));
fprintf(['If it is far from 2, then directional derivatives ' ...
         'might be erroneous.\n']);

  
function [range, poly] = identify_linear_piece(x, y, window_length)
% Identify a segment of the curve (x, y) that appears to be linear.
%
% function [range poly] = identify_linear_piece(x, y, window_length)
%
% This function attempts to identify a contiguous segment of the curve
% defined by the vectors x and y that appears to be linear. A line is fit
% through the data over all windows of length window_length and the best
% fit is retained. The output specifies the range of indices such that
% x(range) is the portion over which (x, y) is the most linear and the
% output poly specifies a first order polynomial that best fits (x, y) over
% that range, following the usual matlab convention for polynomials
% (highest degree coefficients first).
%
% See also: checkdiff checkgradient checkhessian

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, July 8, 2013.
% Contributors: 
% Change log: 

    residues = zeros(length(x)-window_length, 1);
    polys = zeros(2, length(residues));
    for i = 1 : length(residues)
        range = i:(i+window_length);
        [poly, meta] = polyfit(x(range), y(range), 1);
        residues(i) = meta.normr;
        polys(:, i) = poly';
    end
    [unused, best] = min(residues); %#ok<ASGLU>
    range = best:(best+window_length);
    poly = polys(:, best)';

end

end
