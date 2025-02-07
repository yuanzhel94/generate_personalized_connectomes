addpath(genpath('BCT'))
% type='eq_th';
type='eq_den';

load(sprintf('%s_params.mat',type));
t = readtable('test_trait_processed.csv');
t_train = readtable('train_trait_processed.csv');
t = t(:,2:end);
t_train = t_train(:,2:end);
test_id = t.Subject_ID;
train_id = t_train.Subject_ID;
[select_id,ia,ib] = intersect(test_id, ukb_id);
[select_train,ia_train,ib_train] = intersect(train_id, ukb_id);
params = mean(EminParams_pop(ib,:,:),3);
params_train = mean(EminParams_pop(ib_train,:,:),3);

features_train = t_train{ia_train,2:end};
features_test = t{ia,2:end};
[bs1,info1] = lasso(features_train, params_train(:,1),'CV',10);
bs1 = bs1(:,info1.IndexMinMSE); intercept1 = info1.Intercept(info1.IndexMinMSE);
[bs2,info2] = lasso(features_train, params_train(:,2),'CV',10);
bs2 = bs2(:,info2.IndexMinMSE); intercept2 = info2.Intercept(info2.IndexMinMSE);
train_fit1 = features_train * bs1 + intercept1;
train_fit2 = features_train * bs2 + intercept2;
figure;scatter(train_fit1, params_train(:,1));
corr(train_fit1, params_train(:,1))
figure;scatter(train_fit2, params_train(:,2));
corr(train_fit2, params_train(:,2))

test_fit1 = features_test * bs1 + intercept1;
test_fit2 = features_test * bs2 + intercept2;
figure;scatter(test_fit1, params(:,1));
corr(test_fit1, params(:,1))
figure;scatter(test_fit2, params(:,2));
corr(test_fit2, params(:,2))
params_fit = [test_fit1,test_fit2];

load(sprintf('aparc/%s.mat',type));
bc = ukb_bc(:,:,ib);
bc = double(bc);
m = ukb_m(ib);
modeltype = 'matching';
% get group consensus community assignment
n_rep = 100;
avg_c = mean(bc,3);
ci_all = zeros(size(bc,1),n_rep);
for i = 1:100
    [Mi, ~] = community_louvain(avg_c, 1);
    ci_all(:,i) = Mi;
end
numm = max(ci_all,[],1);
use = (numm > 1) & (numm < size(bc,1));
ci_all = ci_all(:,use);
Aall = agreement(ci_all) / size(ci_all,2);
anull = 0;
for i = 1: size(ci_all,2)
    hist_counts = histcounts(ci_all(:, i), 'Normalization', 'count');
    anull = anull + sum((hist_counts / size(bc,1)) .* ((hist_counts - 1) / (size(bc,1) - 1)));
end

A = Aall - anull / size(ci_all,2);
tau = 0;
ci = consensus_und(A, tau, 10); % consensus community for group average connectome

% what measures to evaluate? Calculate for empirical and save sbj of interest
s = zeros(size(bc,[1,2]));
edge_lvl = cell(2,2);
edge_lvl(1,:) = {'ebc','topological_distance_pairs'};
node_lvl = cell(2,8);
node_lvl(1,:) = {'participation_coef','mdz','eigen_centrality','nbc',...
    'local_clustering', 'nodal_ecc', 'local_efficiency', 'degree'};
network_lvl = cell(2,7);
network_lvl(1,:) = {'global_clustering','Q','network_assortativity','cpl',...
    'global_efficiency','radius','diameter'};
for i=1:size(bc,3)
    c = bc(:,:,i);
    triu_idx = find(triu(ones(size(c)),1));
    % 1. identifiability: |A and B| / |A or B| Jaccard index
    %       only compute when evaluate
    
    % 2. edge-level: a. edge existence - jaccard index (compute when evaluate); 
    % b. edge betweenness centrality; 
    % c. topological distance between endpoints
    
    [ebc,nbc] = edge_betweenness_bin(c); % edge and node betweeness centrality
    d = distance_bin(c); %topological distance
    edge_v = {ebc(triu_idx), d(triu_idx)};
    % 3. node level: a. participation coefficient;
    % b. module degree zscore
    % c. eigenvector centrality
    % d. node betweenness centrality (previously computed with ebc)
    % e. local clustering
    % f. (not eligible - weighted networks only) local assortativity
    % g. nodal degree, (nodal strength not eligible - weighted networks only) 
    % h. nodal eccentricity
    % i. local efficiency
    
    par = participation_coef(c, ci); %participation coefficient
    mdz = module_degree_zscore(c, ci);
    eigenvector_centrality = eigenvector_centrality_und(c);
    local_clustering = clustering_coef_bu(c);
    [cpl,global_efficiency,nodal_ecc,radius,diameter] = charpath(d,0,0);
    local_efficiency = efficiency_bin(c,1);
    degree = degrees_und(c)';
    node_v = {par, mdz, eigenvector_centrality, nbc, local_clustering, nodal_ecc, local_efficiency, degree};

    % 4. network level: a. global average clustering
    % b. modularity Q
    % c. network assortativity (available for binary network)
    % d. (not eligible - weighted networks only) network totoal strengths
    % e. characteristic path length (previously computed cpl)
    % f. global efficiency (previously computed global_efficiency)
    % g. network radius (previously computed radius)
    % h. network diameter (previously computed diameter)
    average_clustering = mean(local_clustering);
    [~,Q]=community_louvain(c,1);
    r = assortativity_bin(c,0);
    network_v = {average_clustering, Q, r, cpl, global_efficiency, radius, diameter};


    % cat to record
    for j=1:size(edge_lvl,2)
        edge_lvl{2,j} = cat(2, edge_lvl{2,j}, edge_v{j});
    end
    for j=1:size(node_lvl,2)
        node_lvl{2,j} = cat(2, node_lvl{2,j}, node_v{j});
    end
    for j=1:size(network_lvl,2)
        network_lvl{2,j} = cat(1, network_lvl{2,j}, network_v{j});
    end

end
save(sprintf('emp_%s.mat',type), 's', 'D', 'modeltype', 'modelvar', 'bc', 'm', 'params', 'params_fit', 'ci', 'edge_lvl','node_lvl','network_lvl');
