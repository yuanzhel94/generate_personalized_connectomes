function r_avg = fisherz_avg_r(r_values)
%compute the average pearson r using fisher z transformation

valid_r_values = r_values(~isnan(r_values));
if isempty(valid_r_values)
    r_avg = NaN;
else
    z_values = atanh(valid_r_values);
    z_values = z_values(~isinf(z_values));
    z_mean = mean(z_values);
    r_avg = tanh(z_mean);
end
end