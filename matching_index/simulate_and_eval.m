function simulate_and_eval(type,trial_idx)
% generate connectomes and evaluate how well it replicates empirical,
% benchmarked against connectomes generated from group avg params

rng('shuffle');
load(sprintf('emp_%s.mat',type));

n_sbj = size(bc,3);
n_node = size(bc,1);
bc_gen = zeros(size(bc));
bc_avg = zeros(size(bc));
bc_fit = zeros(size(bc));
avg_params = mean(params,1);

for i=1:n_sbj
    %generate with fitted params
    b = generative_model(s,D,m(i),modeltype,modelvar,params(i,:));
    c = zeros(n_node,n_node);
    c(b) = 1;
    c = c + c';
    bc_gen(:,:,i) = c;

    %generate with avg params
    b_avg = generative_model(s,D,m(i),modeltype,modelvar,avg_params);
    c_avg = zeros(n_node,n_node);
    c_avg(b_avg) = 1;
    c_avg = c_avg + c_avg';
    bc_avg(:,:,i) = c_avg;

    %generate with fitted params
    b_fit = generative_model(s,D,m(i),modeltype,modelvar, params_fit(i,:));
    c_fit = zeros(n_node,n_node);
    c_fit(b_fit) = 1;
    c_fit = c_fit + c_fit';
    bc_fit(:,:,i) = c_fit;

end

% [edge_lvl, node_lvl, network_lvl] = eval_graph_measures(bc,ci);
[edge_gen, node_gen, network_gen] = eval_graph_measures(bc_gen,ci);
[edge_avg, node_avg, network_avg] = eval_graph_measures(bc_avg,ci);
[edge_fit, node_fit, network_fit] = eval_graph_measures(bc_fit,ci);

% [edge_gen, node_gen, network_gen] = eval_graph_measures(bc_rand(:,:,1:100),ci);
% [edge_avg, node_avg, network_avg] = eval_graph_measures(bc_rand(:,:,1:100),ci);

[iden1,edge_existence_jaccard1,edge_corr1,node_corr1,network_corr1] = eval_replication(bc,edge_lvl,node_lvl,network_lvl,bc_gen,edge_gen,node_gen,network_gen);
[iden2,edge_existence_jaccard2,edge_corr2,node_corr2,network_corr2] = eval_replication(bc,edge_lvl,node_lvl,network_lvl,bc_avg,edge_avg,node_avg,network_avg);
[iden3,edge_existence_jaccard3,edge_corr3,node_corr3,network_corr3] = eval_replication(bc,edge_lvl,node_lvl,network_lvl,bc_fit,edge_fit,node_fit,network_fit);

save(sprintf('%s_results/trial_%d.mat', type, trial_idx), 'iden1','edge_existence_jaccard1',...
    'edge_corr1','node_corr1','network_corr1','iden2','edge_existence_jaccard2',...
    'edge_corr2','node_corr2','network_corr2', 'iden3','edge_existence_jaccard3',...
    'edge_corr3','node_corr3','network_corr3');
end



