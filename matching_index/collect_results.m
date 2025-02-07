type='eq_den';
for i=1:1000
    load(sprintf('%s_results/trial_%d.mat', type, i));
    iden_i = iden1;
    for j=1:size(iden1,2)
        iden_i{2,j} = cat(2,iden_i{2,j},iden2{2,j},iden3{2,j});
    end
    edge_existence_i = [edge_existence_jaccard1, edge_existence_jaccard2, edge_existence_jaccard3];
    edge_corr_i = edge_corr1;
    for j=1:size(edge_corr1,2)
        edge_corr_i{2,j} = cat(2,edge_corr_i{2,j},edge_corr2{2,j},edge_corr3{2,j});
    end
    node_corr_i = node_corr1;
    for j=1:size(node_corr1,2)
        node_corr_i{2,j} = cat(2,node_corr_i{2,j},node_corr2{2,j},node_corr3{2,j});
    end
    network_corr_i = network_corr1;
    for j=1:size(network_corr1,2)
        network_corr_i{2,j} = cat(2,network_corr_i{2,j},network_corr2{2,j},network_corr3{2,j});
    end
    
    if i==1
        iden = iden_i;
        edge_existence = edge_existence_i;
        edge_corr = edge_corr_i;
        node_corr = node_corr_i;
        network_corr = network_corr_i;
    else
        for j=1:size(iden,2)
            iden{2,j} = cat(1,iden{2,j},iden_i{2,j});
        end
        edge_existence = cat(1,edge_existence,edge_existence_i);
        for j=1:size(edge_corr,2)
            edge_corr{2,j} = cat(1,edge_corr{2,j},edge_corr_i{2,j});
        end
        for j=1:size(node_corr,2)
            node_corr{2,j} = cat(1,node_corr{2,j},node_corr_i{2,j});
        end
        for j=1:size(network_corr,2)
            network_corr{2,j} = cat(1,network_corr{2,j},network_corr_i{2,j});
        end
    end
    fprintf('%d \n',i);
end
save(sprintf('%s_collected.mat',type), 'iden', 'edge_existence', 'edge_corr','node_corr','network_corr');

