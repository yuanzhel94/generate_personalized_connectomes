addpath(genpath('BCT'));
type='eq_den';
rng('shuffle');
load(sprintf('emp_%s.mat',type));

bc = bc(:,:,1:100);

n_sbj = size(bc,3);
n_node = size(bc,1);
bc_gen = zeros(size(bc));
bc_avg = zeros(size(bc));
bc_rand = zeros(size(bc));
triu_idx = find(triu(ones(size(bc,[1,2])),1));
avg_params = mean(params,1);
tic;
for i=1:size(bc,3)
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

end
toc;

[edge_lvl, node_lvl, network_lvl] = eval_graph_measures(bc,ci);
[edge_gen, node_gen, network_gen] = eval_graph_measures(bc_gen,ci);
[edge_avg, node_avg, network_avg] = eval_graph_measures(bc_avg,ci);

% [edge_gen, node_gen, network_gen] = eval_graph_measures(bc_rand(:,:,1:100),ci);
% [edge_avg, node_avg, network_avg] = eval_graph_measures(bc_rand(:,:,1:100),ci);

[iden1,edge_existence_jaccard1,edge_corr1,node_corr1,network_corr1] = eval_replication(bc,edge_lvl,node_lvl,network_lvl,bc_gen,edge_gen,node_gen,network_gen);
[iden2,edge_existence_jaccard2,edge_corr2,node_corr2,network_corr2] = eval_replication(bc,edge_lvl,node_lvl,network_lvl,bc_avg,edge_avg,node_avg,network_avg);
toc;

save(sprintf('results/trial_%d.mat', trial_idx), 'iden1','edge_existence_jaccard1',...
    'edge_corr1','node_corr1','network_corr1','iden2','edge_existence_jaccard2',...
    'edge_corr2','node_corr2','network_corr2');
