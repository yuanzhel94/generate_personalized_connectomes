function [r,p] = masked_pearsonr(v1,v2)
%Compute pearson r ignoring all position may contain inf or nan

mask = ~(isnan(v1) | isinf(v1) | isnan(v2) | isinf(v2));
if sum(mask) > 1
    [r,p] = corr(v1(mask), v2(mask));
else
    r = nan; p = nan;
end
end