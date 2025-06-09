function alphaK = calculate_weights(KH, initial_weights)
% Compute view weights based on Maximum Mean Discrepancy (MMD)
% Inputs:
%   D_Kernels: (N x N x num_views) kernel matrix collection
%   initial_weights: optional initial weights (num_views x 1)
% Output:
%   alphaK: normalized MMD-based weights (num_views x 1)

[N, ~, num_views] = size(KH);

if nargin < 2 || isempty(initial_weights)
    initial_weights = ones(num_views, 1) / num_views;
end

H = zeros(N, N);
for v = 1:num_views
    H = H + initial_weights(v) * KH(:, :, v);
end

global_sim = H * H';
alphaK = zeros(num_views, 1);

for v = 1:num_views
    K_v = KH(:, :, v);
    view_sim = K_v * K_v';
    related_sim = K_v * H';
    mmd_value = (sum(view_sim(:)) + sum(global_sim(:)) - 2 * sum(related_sim(:))) / (N^2);
    alphaK(v) = exp(-mmd_value);
end

alphaK = alphaK / sum(alphaK);
end
