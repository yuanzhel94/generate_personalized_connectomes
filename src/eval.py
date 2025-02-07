import torch
import numpy as np
import bct
import scipy.stats as stats
from concurrent.futures import ProcessPoolExecutor as PPE
import os
from utils import *
import time
import pickle
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

def crosscorr(inputs, target, eps=1e-8):
    '''
    Compute the pairwise correlation between inputs and target
    input is expected to be tensors
    in output, corr[i,j] gives the correlation between ith row in puts and jth row in target
    '''
    inputs_num = inputs - inputs.mean(dim=1, keepdim=True) # (B1,N)
    target_num = target - target.mean(dim=1, keepdim=True) # (B2,N)
    inputs_denom = inputs_num.square().sum(dim=1, keepdim=True).sqrt() + eps # (B1,1)
    target_denom = target_num.square().sum(dim=1, keepdim=True).sqrt() + eps # (B2,1)
    corr = torch.divide(torch.mm(inputs_num, target_num.t()), torch.mm(inputs_denom, target_denom.t()))
    return corr

def crosscorr_np(inputs, target, eps=1e-8):
    '''
    Compute the pairwise correlation between inputs and target.
    inputs and target are expected to be numpy arrays.
    In the output, corr[i,j] gives the correlation between the ith row in inputs and the jth row in target.
    '''
    # Subtract the mean from each row
    inputs_num = inputs - inputs.mean(axis=1, keepdims=True)  # (B1, N)
    target_num = target - target.mean(axis=1, keepdims=True)  # (B2, N)
    
    # Compute the denominator (sqrt of the sum of squared deviations)
    inputs_denom = np.sqrt((inputs_num ** 2).sum(axis=1, keepdims=True)) + eps  # (B1, 1)
    target_denom = np.sqrt((target_num ** 2).sum(axis=1, keepdims=True)) + eps  # (B2, 1)
    
    # Compute the correlation
    corr = np.dot(inputs_num, target_num.T) / (np.dot(inputs_denom, target_denom.T))
    
    return corr

def vector2adj(vector, n_region, triu_indices=None):
    adj = np.zeros((n_region, n_region))
    if triu_indices is None:
        triu_indices = np.triu_indices(n_reigon, k=1)
    adj[triu_indices] = vector
    adj = adj + np.transpose(adj)
    return adj

def get_graph_metrics(adj, group_community=None):

    edge_metrics = ["connectivity", "edge betweenness centrality", "node pair topological distance"]
    node_metrics = ["participation coefficient", "module degree zscore", "eigenvector centrality", "node betweenness centrality", "local clustering", \
        "local_assortativity", "nodal strengths", "nodal eccentricity", "local efficiency"]
    network_metrics = ["global average clustering", "modularity", "assortativity", "network total strengths", "charpath", "global efficiency", "radius", "diameter"]

    L = 1 / adj

    [M,Q]=bct.community_louvain(adj, gamma=1)
    if group_community is None:
        group_community = M

    nodal_strengths = bct.strengths_und(adj)
    total_strengths = nodal_strengths.sum()
    participation_coef = bct.participation_coef(adj, group_community)
    module_degree_zscore = bct.module_degree_zscore(adj, group_community)
    eigenvector_centrality = bct.eigenvector_centrality_und(adj)
    edge_betweenness, node_betweenness = bct.edge_betweenness_wei(L)
    local_clustering = bct.clustering_coef_wu(adj)
    average_clustering = local_clustering.mean()
    local_assortativity, _ = bct.local_assortativity_wu_sign(adj)
    assortativity = bct.assortativity_wei(adj)
    spl, _, _ = bct.distance_wei_floyd(L)
    charpath, global_efficiency, nodal_eccentricity, radius, diameter = bct.charpath(spl)
    local_efficiency = bct.efficiency_wei(adj, "local")

    np.fill_diagonal(adj, np.nan)
    np.fill_diagonal(edge_betweenness, np.nan)
    np.fill_diagonal(spl, np.nan)
    edge_results = [adj, edge_betweenness, spl]
    node_results = [participation_coef, module_degree_zscore, eigenvector_centrality, node_betweenness, local_clustering, \
        local_assortativity, nodal_strengths, nodal_eccentricity, local_efficiency]
    network_results = [average_clustering, Q, assortativity, total_strengths, charpath, global_efficiency, radius, diameter]

    edge_level = {}
    for i, k in enumerate(edge_metrics):
        edge_level[k] = edge_results[i]
    node_level = {}
    for i, k in enumerate(node_metrics):
        node_level[k] = node_results[i]
    network_level = {}
    for i, k in enumerate(network_metrics):
        network_level[k] = network_results[i]
    output = {}
    output["edge_level"] = edge_level
    output["node_level"] = node_level
    output["network_level"] = network_level
    return output

# def summarize_graph_metrics(individual, group_average, triu_indices, system_label):
#     individual_edge = individual["edge_level"]
#     avg_edge = group_average["edge_level"]
#     individual_node = individual["node_level"]
#     avg_node = group_average["node_level"]
#     n_system = system_label.shape[0]
#     individual["system_edge"] = {}
#     individual["system_node"] = {}
#     for k,v in individual_edge.items():
#         individual["edge_level"][k] = stats.pearsonr(v[triu_indices], avg_edge[k][triu_indices]).statistic
#         system_stat = np.zeros((n_system,1))
#         for i in range(n_system):
#             system_value = np.nanmean(v[system_label[i], :], axis=0)
#             avg_value = np.nanmean(avg_edge[k][system_label[i], :], axis=0)
#             system_stat[i] = stats.pearsonr(system_value, avg_value).statistic
#         individual["system_edge"][f"{k}_7nets"] = system_stat
#     for k,v in individual_node.items():
#         individual["node_level"][k] = stats.pearsonr(v, avg_node[k]).statistic
#         system_stat = np.zeros((n_system,1))
#         for i in range(n_system):
#             system_stat[i] = v[system_label[i]].mean()
#         individual["system_node"][f"{k}_7nets"] = system_stat
#     return individual

def individual_graph_metrics(vector, n_region, group_average_metrics, system_label, triu_indices, group_community):
    adj = vector2adj(vector, n_region, triu_indices)
    individual_metrics = get_graph_metrics(adj, group_community)
    # individual_metrics = summarize_graph_metrics(individual_metrics, group_average_metrics, triu_indices, system_label)
    return individual_metrics

def group_average_graph_metrics(vector, n_region, group_community, triu_indices):
    adj = vector2adj(vector, n_region, triu_indices)
    t1 = time.time()
    avg_metrics = get_graph_metrics(adj, group_community)
    t2 = time.time()
    print(f"{t2-t1}s to compute metrics")
    return avg_metrics

def group_average_consensus_community(vector, n_region, triu_indices, n_rep=100):
    adj = vector2adj(vector, n_region, triu_indices)
    ci_all = np.zeros((n_region, n_rep))
    for i in range(n_rep):
        Mi, Qi = bct.community_louvain(adj, gamma=1)
        ci_all[:,i] = Mi
    numm = ci_all.max(axis=0)
    use = (numm > 1) & (numm < n_region)
    ci_all = ci_all[:,use]
    Aall = bct.agreement(ci_all) / ci_all.shape[1]
    anull = 0
    for i in range(ci_all.shape[1]):
        hist_counts, _ = np.histogram(ci_all[:, i])
        anull += np.sum((hist_counts / n_region) * ((hist_counts - 1) / (n_region - 1)))
    A = Aall - anull / ci_all.shape[1]
    tau = 0
    ciu = bct.consensus_und(A, tau, 10)
    return ciu

def group_graph_metrics(connectivity:np.ndarray, group_average_metrics, group_community, settings, system_label, triu_indices=None, is_parallel=True):
    if triu_indices is None:
        triu_indices = np.triu_indices(settings.n_reigon, k=1)
    assert len(connectivity.shape) == 2
    n = connectivity.shape[0]
    if is_parallel:
        vectors = [connectivity[i,:] for i in range(n)]
        n_regions = [settings.n_region] * n
        group_average_metrics_list = [group_average_metrics] * n
        triu_indices_list = [triu_indices] * n
        group_community_list = [group_community] * n
        system_label_list = [system_label] * n
        with PPE() as executor:
            results = list(executor.map(individual_graph_metrics, vectors, n_regions, group_average_metrics_list, system_label_list, triu_indices_list, group_community_list))
    else:
        results = []
        for i in range(n):
            results.append(individual_graph_metrics(connectivity[i,:], settings.n_region, group_average_metrics, system_label, triu_indices, group_community))
    # results is a list of dictionaries returned by individual_graph_metrics() function; *results is passed to merge_metrics_dict() function
    return results

def merge_metrics_dict(*dicts):
    for i, dic in enumerate(dicts):
        if i == 0:
            out_dict = {}
            for k, v in dic.items():
                out_dict[k] = {}
                for kk, _ in v.items():
                    out_dict[k][kk] = []
        for k, v in dic.items():
            for kk, vv in v.items():
                if isinstance(vv, list):
                    out_dict[k][kk].extend(vv)
                else:    
                    out_dict[k][kk].append(vv)
    return out_dict

def get_group_average_metrics(f_path, settings, connectivity_mu, triu_indices=None):
    if os.path.exists(f_path):
        with open(f_path, "rb") as file:
            group_metrics = pickle.load(file)
    else:
        if triu_indices is None:
            triu_indices = np.triu_indices(settings.n_region, k=1)
        group_community = group_average_consensus_community(connectivity_mu, settings.n_region, triu_indices)
        average_metrics = group_average_graph_metrics(connectivity_mu, settings.n_region, group_community, triu_indices)
        group_metrics = {}
        group_metrics["group_community"] = group_community
        group_metrics["average_metrics"] = average_metrics
        with open(f_path, "wb") as file:
            pickle.dump(group_metrics, file)
    return group_metrics

def paired_violin(arr1, arr2, arr1_name, arr2_name, violin_names, diffs, ps, savepath, figsize=(12,6), fs=30, inner=None, star_fs=30, theoretical_null=None, remove_text=False, sparse_tick=False):
    # plot splitted violin plots for two arrays, with ith violin visualize arr1[i,:] on the left and arr2[i,:] on the right
    # arrays are of shape (n_metrics, n_observations), violin_names is a vector of length n that notes the violin tick labels
    n_legend = 2
    n = arr1.shape[0]
    if n==1:
        diffs = [diffs]
        ps = [ps]
    n_samp = arr1.shape[1]
    data = []
    group = []
    category = []
    for i in range(n):
        data.extend(arr1[i,:])
        data.extend(arr2[i,:])
        group.extend([violin_names[i]] * n_samp * 2)
        category.extend([arr1_name] * n_samp + [arr2_name] * n_samp)
    df = pd.DataFrame({"value": data, "group": group, "category":category})

    plt.figure(figsize=figsize)
    ax = sns.violinplot(x="group", y="value", hue="category", data=df, split=True, density_norm="width", inner=inner)
    max_y = np.max((arr1.max(), arr2.max()))
    min_y = np.max((arr1.min(), arr2.min()))
    star_y = max_y + (max_y - min_y) * 0.1
    upper_y = max_y + (max_y - min_y) * 0.2
    for i,p in enumerate(ps):
        color = "red" if diffs[i] > 0 else "black"
        if p < 0.001:
            ax.text(i, star_y, "***", ha="center", fontsize=star_fs, color=color)
        elif p < 0.01:
            ax.text(i, star_y, "**", ha="center", fontsize=star_fs, color=color)
        elif p < 0.05:
            ax.text(i, star_y, "*", ha="center", fontsize=star_fs, color=color)
    if theoretical_null is not None:
        ax.axhline(y=theoretical_null, linewidth=2, color='black', linestyle='--', label='theoretical null')
        n_legend = 3
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if sparse_tick:
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], upper_y)
        yticks = ax.get_yticks()
        yticks_select = [yticks[0], yticks[-1]]
        if theoretical_null is not None:
            yticks_select.append(theoretical_null)
        ax.set_yticks(yticks_select)

    plt.xticks(rotation=45, fontsize=fs, ha="right")
    plt.yticks(fontsize=fs)
    plt.xlabel("")
    plt.ylabel("")
    plt.subplots_adjust(top=0.75)
    plt.legend(title="", fontsize=fs, bbox_to_anchor=(1, 1), loc='lower right', ncol=n_legend)

    if remove_text:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.get_legend().remove()

    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def permutation_test(arr1, arr2, n_perm=10000, alternative="two-sided"):
    '''
    permutation test to compare mean between independent groups without assuming distributions
    '''
    observed_diff = np.mean(arr1) - np.mean(arr2)
    combined = np.concatenate([arr1, arr2])
    diff_means = np.zeros(n_perm)
    
    for i in range(n_perm):
        np.random.shuffle(combined)
        diff_means[i] = np.mean(combined[:len(arr1)]) - np.mean(combined[len(arr1):])
    
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(diff_means) >= np.abs(observed_diff))
    elif alternative == 'less':
        p_value = np.mean(diff_means <= observed_diff)
    elif alternative == 'greater':
        p_value = np.mean(diff_means >= observed_diff)
    return observed_diff, p_value

# def permutation_test_rel(arr1, arr2, n_perm=1000):
#     '''
#     permutation test to compare mean between two groups of paired observations without assuming distributions
#     '''
#     pass

def subgroup_bar_plots(arr, subgroup_names, bar_names, savepath, err_bar=None, fs=20, figsize=(20,6), remove_text=False, sparse_tick=False):
    '''
    Create a barplot figure
    arr: shape [n_subgroup, n_bar] where n_bar is the number of barplots in each subgroup
    err_bar: if None, do not add error bar; else shape [n_subgroup, n_bar] (positive relative valeus from the bar height, low=high)
    '''
    n_subgroup, n_bar = arr.shape
    barwidth = 1 / (n_bar + 1)
    fig, ax = plt.subplots(figsize=figsize)
    xpos = np.arange(n_subgroup)
    cmap = plt.colormaps["Set3"]
    colors = [cmap(i/n_bar) for i in range(n_bar)]

    for i in range(n_bar):
        xpos_i = xpos + i * barwidth
        if err_bar is not None:
            ax.bar(xpos_i, arr[:,i], yerr = err_bar[:,i], color=colors[i], label=bar_names[i], width=barwidth)
        else:
            ax.bar(xpos_i, arr[:,i], color=colors[i], label=bar_names[i], width=barwidth)
    
    tick_pos = np.divide(xpos + xpos_i, 2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if sparse_tick:
        yticks = ax.get_yticks()
        yticks_select = [yticks[0], yticks[-1]]
        if 0 not in yticks_select:
            yticks_select.append(0)
        ax.set_yticks(yticks_select)

    plt.xticks(tick_pos, subgroup_names, rotation=45, fontsize=fs, ha="right")
    plt.yticks(fontsize=fs)
    plt.legend(title="", fontsize=fs, bbox_to_anchor=(1, 1), loc='lower right', ncol=len(bar_names))

    if remove_text:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.get_legend().remove()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

def violin_with_pairwise_test(data, labels, test_fn, save_path, figsize=(20,6), inner="box"):
    # data is n_observation by n_category
    n = data.shape[1]
    p_mat = np.ones((n,n))
    for i in range(n-1):
        for j in range(i+1,n):
            _,p = test_fn(data[:,i], data[:,j])
            p_mat[i,j] = p
    triu_indices = np.triu_indices(p_mat.shape[0], k=1)
    contrast = data.mean(axis=0).reshape(-1, 1) - data.mean(axis=0).reshape(1, -1)
    corrected = stats.false_discovery_control(p_mat[triu_indices])
    p_mat[triu_indices] = corrected

    not_significant = p_mat >= 0.05
    contrast = np.triu(contrast, k=1)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    sns.violinplot(data=data, ax=axes[0], inner=inner)
    axes[0].set_xticks(np.arange(n))
    axes[0].set_xticklabels(labels)

    max_abs = np.abs(contrast).max()
    cax = axes[1].imshow(contrast, cmap='bwr', vmin=-max_abs, vmax=max_abs)
    axes[1].set_xticks(np.arange(n))
    axes[1].set_yticks(np.arange(n))
    axes[1].set_xticklabels(labels)
    axes[1].set_yticklabels(labels)
    fig.colorbar(cax, ax=axes[1])

    mask_indices = np.where(not_significant == False)
    axes[1].scatter(mask_indices[1], mask_indices[0], color='black', marker='*', s=100, edgecolor='black')

    axes[0].tick_params(axis='both', which='major', labelsize=35)
    axes[1].tick_params(axis='both', which='major', labelsize=35)
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(axes[1].get_yticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def violin_with_pairwise_test_separate(data, labels, test_fn, save_path, figsize=(20,6), inner="box", sort=False, remove_text=False, color_p_by="contrast"):
    # same with violin_with_pairwise_test but save as separate files
    # save path becomes a list fo [violin_path, p_mat_path]
    # also make the p_mat plot into lower triangular matrix instead of upper triangular
    # figsize is for the violinplot; figsize of the p_mat plot is constant
    # include additional argument to sort the violins from smallest to largest (p_mat also in the same order)
    data_cp = np.copy(data)
    n = data_cp.shape[1]
    p_mat = np.ones((n,n))
    alternative="two-sided"
    palette = sns.color_palette('deep')
    palette = palette[:len(labels)]
    idx = np.arange(data_cp.shape[1])
    if sort:
        idx = np.argsort(data_cp.mean(axis=0))
        data_cp = data_cp[:,idx]
        labels = labels[idx]
        # alternative = "less"
        palette = np.asarray(palette)[idx].tolist()
    for i in range(n-1):
        for j in range(i+1,n):
            _,p = test_fn(data_cp[:,i], data_cp[:,j], alternative=alternative)
            p_mat[j,i] = p
    tril_indices = np.tril_indices(p_mat.shape[0], k=-1)
    contrast = data_cp.mean(axis=0).reshape(1, -1) - data_cp.mean(axis=0).reshape(-1, 1)
    corrected = stats.false_discovery_control(p_mat[tril_indices])
    p_mat[tril_indices] = corrected

    not_significant = p_mat >= 0.05
    contrast = np.tril(contrast, k=-1)

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=data_cp, ax=ax, inner=inner, palette=palette)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=35)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if remove_text:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(save_path[0])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,6))
    max_abs = np.abs(contrast).max()

    if color_p_by == "contrast":
        if sort:
            cax = ax.imshow(-contrast, cmap='Blues', vmin=0, vmax=max_abs)
        else:
            cax = ax.imshow(contrast, cmap='bwr', vmin=-max_abs, vmax=max_abs)
    elif color_p_by == "p":
        cax = ax.imshow(p_mat, cmap='Reds_r')
    else:
        raise NotImplementedError("parameter 'color_p_by' must be chosen from ['contrast','p']")
        
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    cbar = fig.colorbar(cax, ax=ax)

    mask_indices = np.where(not_significant == False)
    ax.scatter(mask_indices[1], mask_indices[0], color='black', marker='*', s=100, edgecolor='black')

    ax.tick_params(axis='both', which='major', labelsize=35)
    ax.tick_params(axis='both', which='major', labelsize=35)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if remove_text:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        cbar.ax.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(save_path[1])
    plt.close(fig)
    return idx # sort index

def matrix_plot_contrast_with_significance(contrasts, significant, subgroup_names, pixel_names, savepath, n_rows = 2, fs=20, figsize=(20,6)):
    n_plots = contrasts.shape[0]
    n_pixels = contrasts.shape[-1]
    n_cols = (n_plots + 1) // n_rows
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        if i >= n_plots:
            fig.delaxes(ax)
        else:
            contrast_i = contrasts[i,:,:]
            max_abs = np.abs(contrast_i).max()
            cax = ax.imshow(contrasts[i,:,:], cmap='bwr', vmin=-max_abs, vmax=max_abs)
            ax.set_xticks(np.arange(n_pixels))
            ax.set_yticks(np.arange(n_pixels))
            ax.set_xticklabels(pixel_names)
            ax.set_yticklabels(pixel_names)
            fig.colorbar(cax, ax=ax)

            mask_indices = np.where(significant[i,:,:] == True)
            ax.scatter(mask_indices[1], mask_indices[0], color='black', marker='*', s=100, edgecolor='black')
            ax.tick_params(axis='both', which='major', labelsize=fs)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            plt.setp(ax.get_yticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            ax.set_title(subgroup_names[i],fontsize=fs)

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close(fig)

def composition_distance(g1,g2):
    n = g1.shape[0]
    d1 = pdist(g1)
    d2 = pdist(g2)
    g = np.concatenate((g1,g2),axis=0)
    d12 = squareform(pdist(g))
    d12 = d12[np.arange(n),:][:,np.arange(n, 2*n)]

    diff = d12.mean() - np.concatenate((d1,d2),axis=0).mean()
    return diff

def compare_composition_paired(g1, g2, n_perm=10000):
    n = g1.shape[0]
    assert g2.shape[0] == n
    diff = composition_distance(g1,g2)

    perm_diff = np.zeros((n_perm,))
    for i in range(n_perm):
        select_self = np.random.choice(2, size=g1.shape[0]).astype(bool)
        select_other = np.logical_not(select_self)
        g1_perm = np.zeros(g1.shape)
        g2_perm = np.zeros(g2.shape)
        g1_perm[select_self,:] = g1[select_self,:]
        g2_perm[select_self,:] = g2[select_self,:]
        g1_perm[select_other,:] = g2[select_other,:]
        g2_perm[select_other,:] = g1[select_other,:]
        perm_diff[i] = composition_distance(g1_perm,g2_perm)
    p = (diff < perm_diff).sum() / perm_diff.shape[0]
    return diff, p

def compare_composition_all_pairs(compositions):
    n_group = compositions.shape[1]
    pmat = np.ones((n_group, n_group))
    diff_mat = np.zeros((n_group, n_group))
    diff_vec = []
    pvec = []
    cols = []
    rows = []
    for j in range(n_group-1):
        for i in range(j+1, n_group):
            data_j = compositions[:,j,:]
            data_i = compositions[:,i,:]
            cols.append(j)
            rows.append(i)
            diff, p = compare_composition_paired(data_i, data_j)
            pvec.append(p)
            diff_vec.append(diff)
    
    pvec = np.asarray(pvec)
    p_corrected = stats.false_discovery_control(pvec)
    cols = np.asarray(cols)
    rows = np.asarray(rows)
    pmat[rows,cols] = p_corrected
    diff_mat[rows,cols] = np.asarray(diff_vec)
    return diff_mat, pmat

def scatter_embedding(savepath, embedding, labels, label_color, s=50, figsize=(8,6)):
    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = np.unique(labels)
    for label, color in zip(unique_labels, label_color):
        indices = labels == label
        ax.scatter(embedding[indices,0], embedding[indices,1], c=[color], s=s, label=label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close(fig)




