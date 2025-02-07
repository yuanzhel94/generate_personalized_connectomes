function [iden,edge_existence_jaccard,edge_corr,node_corr,network_corr] = eval_replication(bc1,edge1,node1,network1,bc2,edge2,node2,network2)
% evaluate how well generated connectomes replicate empirical obesrvations
% bc1/2, edge1/2, node1/2, network1/2 are the binary connectomes, edge-lvl,
% node-lvl, and network-lvl features, for connectome group 1 and 2 respectively. 
% mode can be 'jaccard' or 'dice'
n_node = size(bc1,1);
n_sbj = size(bc1,3);
mask = repmat(triu(true(n_node),1),1,1,n_sbj);
iden_size = min(200,n_sbj);
iden_idx = randsample(n_sbj,iden_size);

bc1_flatten = reshape(bc1(mask),n_node * (n_node - 1) / 2, n_sbj);
bc2_flatten = reshape(bc2(mask),n_node * (n_node - 1) / 2, n_sbj);

bc1_iden = bc1_flatten(:,iden_idx);
bc2_iden = bc2_flatten(:,iden_idx);

intersections = bc1_iden' * bc2_iden;
unions = sum(bc1_iden,1)' + sum(bc2_iden,1) - intersections;
jaccard_index = intersections ./ unions;
iden = cell(2,4);
iden(1,:) = {'diff_iden','rank_prc','top1_acc','hungarian_acc'};
iden{2,1} = mean(diag(jaccard_index)) - mean(jaccard_index(find(triu(ones(size(jaccard_index)),1))));
prc_smaller = sum((jaccard_index - diag(jaccard_index)') < 0, 1)/iden_size;
prc_equal = sum((jaccard_index - diag(jaccard_index)') == 0, 1)/iden_size;
iden{2,2} = mean(prc_smaller + prc_equal/2);
iden{2,3} = mean(diag(jaccard_index)' == max(jaccard_index,[],1));
assignments = assignDetectionsToTracks(1-jaccard_index, iden_size^2);
iden{2,4} = sum((1:iden_size)' == assignments(:,1)) / iden_size;

edgewise_intersections = sum(bc1_flatten .* bc2_flatten,2);
edgewise_unions = sum(bc1_flatten,2) + sum(bc2_flatten,2) - edgewise_intersections;
edge_existence_jaccard = edgewise_intersections ./ edgewise_unions;
edge_existence_jaccard = mean(edge_existence_jaccard(~(isinf(edge_existence_jaccard) | isnan(edge_existence_jaccard))));

edge_corr = edge1;
node_corr = node1;
network_corr = network1;

for i=1:size(edge1,2)
    n_eval = size(edge1{2,i},1);
    rs = zeros(n_eval,1);
    measure1 = edge1{2,i};
    measure2 = edge2{2,i};
    for j = 1:n_eval
        rs(j) = masked_pearsonr(measure1(j,:)',measure2(j,:)');
    end
    r_avg = fisherz_avg_r(rs);
    edge_corr{2,i} = r_avg;
end

for i=1:size(node1,2)
    n_eval = size(node1{2,i},1);
    rs = zeros(n_eval,1);
    measure1 = node1{2,i};
    measure2 = node2{2,i};
    for j = 1:n_eval
        rs(j) = masked_pearsonr(measure1(j,:)',measure2(j,:)');
    end
    r_avg = fisherz_avg_r(rs);
    node_corr{2,i} = r_avg;
end

for i=1:size(network1,2)
    measure1 = network1{2,i};
    measure2 = network2{2,i};
    r = masked_pearsonr(measure1,measure2);
    network_corr{2,i} = r;
end

end