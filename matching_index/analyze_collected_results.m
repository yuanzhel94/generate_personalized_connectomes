
type = 'eq_den';
load(sprintf('%s_collected.mat',type));

n_measures = size(iden,2) + 1 + size(edge_corr,2) + size(node_corr,2) + size(network_corr,2);
measures = cell(4,n_measures); % 4 rows are: name, values, raw permutation p, observed diff
record = zeros(1000,3,n_measures);
count = 1;
for i=1:size(iden,2)
    record(:,:,count) = iden{2,i};
    measures{1,count} = iden{1,i};
    count = count + 1;
end
record(:,:,count) = edge_existence;
measures{1,count} = 'edge_existence';
for i=1:size(edge_corr,2)
    record(:,:,count) = edge_corr{2,i};
    measures{1,count} = edge_corr{1,i};
    count = count + 1;
end
for i=1:size(node_corr,2)
    record(:,:,count) = node_corr{2,i};
    measures{1,count} = node_corr{1,i};
    count = count + 1;
end
for i=1:size(network_corr,2)
    record(:,:,count) = network_corr{2,i};
    measures{1,count} = network_corr{1,i};
    count = count + 1;
end

all_p = [];
for i=1:size(record,3)
    [p1, diff1] = permutationTest(record(:,1,i), record(:,2,i), 10000);
    [p2, diff2] = permutationTest(record(:,3,i), record(:,2,i), 10000);
    measures{2,i} = mean(record(:,:,i),1)';
    measures{3,i} = [p1; p2];
    measures{4,i} = [diff1; diff2];
    all_p = cat(2, all_p, [p1; p2]);
end

measures{4,:}
rank prc