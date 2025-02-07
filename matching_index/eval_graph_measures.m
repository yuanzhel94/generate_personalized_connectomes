function [edge_lvl, node_lvl, network_lvl] = eval_graph_measures(bcs,ci)
% compute the selected graph measures or a given binary network
% inputs:
%   c: the binary network, size (n_node,n_node,n_sbj)
%   ci: precomputed community assignment - for consistency between networks

edge_lvl = cell(2, 2);
edge_lvl(1,:) = {'ebc', 'topological_distance_pairs'};
node_lvl = cell(2, 8);
node_lvl(1,:) = {'participation_coef', 'mdz', 'eigen_centrality','nbc',...
    'local_clustering', 'nodal_ecc', 'local_efficiency','degree'};
network_lvl = cell(2, 7);
network_lvl(1,:) = {'global_clustering','Q', 'network_assortativity','cpl',...
    'global_efficiency', 'radius', 'diameter'};

triu_idx = find(triu(ones(size(bcs,1)),1));
for i=1:size(bcs,3)
    c = bcs(:,:,i);
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
    % g. (not eligible - weighted networks only) nodal strength
    % h. nodal eccentricity
    % i. local efficiency
    
    par = participation_coef(c, ci); %participation coefficient
    mdz = module_degree_zscore(c, ci);
    eigenvector_centrality = eigenvector_centrality_und(c);
    local_clustering = clustering_coef_bu(c);
    [cpl,global_efficiency,nodal_ecc,radius,diameter] = charpath(d,0,0);
    local_efficiency = efficiency_bin(c,1);
    degree = degrees_und(c)';
    node_v = {par, mdz, eigenvector_centrality, nbc, local_clustering, nodal_ecc, local_efficiency,degree};
    
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