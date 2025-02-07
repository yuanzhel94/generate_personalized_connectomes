import numpy as np
from tqdm import tqdm
import scipy.stats as stats

from eval import *
from utils import *
from tmp import *
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

NETWORK_MEASURE_SELECTED = ['connectivity','edge betweenness centrality',\
    'participation coefficient','local clustering', 'nodal strengths',\
        'modularity', 'network total strengths', 'charpath']

def model_versus_null(selected_only, parc='Schaefer7n200p'): # section 2
    if parc=='Schaefer7n200p':
        figpath = "/data/gpfs/projects/punim1278/projects/vaegm/figures/section2" if not selected_only else "/data/gpfs/projects/punim1278/projects/vaegm/figures/section2_selected"
    else:
        figpath = f"/data/gpfs/projects/punim1278/projects/vaegm/figures/{parc}/section2" if not selected_only else f"/data/gpfs/projects/punim1278/projects/vaegm/figures/{parc}/section2_selected"

    ablation_indices = [0, -1] # 0 for model and -1 for null
    values, metrics = get_system_variations(ablation_indices,parc) #shape (metrics, 1000, ablations) for each metric type (level); netrics: key - metric type(level), value - list of metric names correspond to axis 0 in variable values
    
    # prepare identifiability data
    diff_iden = [[], []]
    others_iden = [[], []]
    type_list = ["corr", "demean", "mse"]
    others_iden_names = None
    diff_iden_diff = []
    diff_iden_ps = []
    for metric_type in type_list: #e.g., corr, [diff_iden, hungarian_acc, ...]
        diff_iden_idx = np.asarray([metric == "diff_iden" for metric in metrics[metric_type]])
        others_idx = np.logical_not(diff_iden_idx)
        diff_iden[0].append(values[metric_type][diff_iden_idx,:,0]) # append (1,1000)
        diff_iden[1].append(values[metric_type][diff_iden_idx,:,1]) # append (1,1000)
        
        others_iden[0].append(values[metric_type][others_idx,:,0]) # append (3,1000)
        others_iden[1].append(values[metric_type][others_idx,:,1]) # append (3,1000)
        if others_iden_names is None:
            others_iden_names = []
            for i,b in enumerate(others_idx):
                if b:
                    others_iden_names.append(metrics[metric_type][i])
    others_iden = np.asarray(others_iden)
    
    graph_levels = ["edge", "node", "network"]

    # get permutation results
    mean_diffs = []
    ps = []
    for i, metric_type in enumerate(type_list): # 3 diff_iden
        mean_diff, p = permutation_test(diff_iden[0][i].squeeze(), diff_iden[1][i].squeeze())
        mean_diffs.append(mean_diff)
        ps.append(p)
    for i, name in enumerate(others_iden_names): # 3 other identifiabilities, each consists 3 tests (corr, demean, mse)
        n_test = others_iden.shape[1]
        for j in range(n_test):
            mean_diff, p = permutation_test(others_iden[0,j,i,:], others_iden[1,j,i,:])
            mean_diffs.append(mean_diff)
            ps.append(p)
    for lvl in graph_levels:
        n_test = values[lvl].shape[0]
        for j in range(n_test):
            mean_diff, p = permutation_test(values[lvl][j,:,0], values[lvl][j,:,1])
            mean_diffs.append(mean_diff)
            ps.append(p)

    mean_diffs = np.asarray(mean_diffs)
    ps = np.asarray(ps)
    ps_corrected = stats.false_discovery_control(ps)

    np.save(f'/data/gpfs/projects/punim1278/projects/vaegm/data/SC_{parc}/mean_diffs.npy', mean_diffs)

    # plot differential identifiability
    iden_path = os.path.join(figpath, "identifiability")
    os.makedirs(iden_path, exist_ok=True)
    count = 0
    for i, metric_type in enumerate(type_list):
        savepath = os.path.join(iden_path, f"{metric_type}_diff_iden.png")
        savepath1 = os.path.join(iden_path, f"{metric_type}_diff_iden_notxt.png")
        paired_violin(diff_iden[0][i], diff_iden[1][i], "model", "null", ["diff_iden"], mean_diffs[count], ps_corrected[count], savepath, figsize=(18,12), fs=30)
        paired_violin(diff_iden[0][i], diff_iden[1][i], "model", "null", ["diff_iden"], mean_diffs[count], ps_corrected[count], savepath1, figsize=(18,12), fs=30, remove_text=True)
        print(f"{metric_type} differential identifiability: {diff_iden[0][i].mean()}; std: {diff_iden[0][i].std()}; p {ps_corrected[count]}; null: {diff_iden[1][i].mean()}; std: {diff_iden[1][i].std()}")
        count +=1
        
        
    # plot other identifiability
    for i, name in enumerate(others_iden_names):
        savepath = os.path.join(iden_path, f"{name}.png")
        savepath1 = os.path.join(iden_path, f"{name}_notxt.png")
        n_test = others_iden.shape[1]
        if name == "rank_prc":
            theoretical_null = 0.5
        else:
            theoretical_null = 0.005
        if selected_only:
            theoretical_null = None
        theoretical_null = None
        paired_violin(others_iden[0,:,i,:], others_iden[1,:,i,:], "model", "null", type_list, mean_diffs[count:count+n_test], ps_corrected[count:count+n_test], savepath, figsize=(18,12), fs=30, theoretical_null=theoretical_null)
        paired_violin(others_iden[0,:,i,:], others_iden[1,:,i,:], "model", "null", type_list, mean_diffs[count:count+n_test], ps_corrected[count:count+n_test], savepath1, figsize=(18,12), fs=30, theoretical_null=theoretical_null, remove_text=True)
        print(f"{name} for {type_list} are {others_iden[0,:,i,:].mean(axis=1)}; std {others_iden[0,:,i,:].std(axis=1)}; null are {others_iden[1,:,i,:].mean(axis=1)}; std {others_iden[1,:,i,:].std(axis=1)}; p {ps_corrected[count:count+n_test]}")
        count += n_test
        


    # plot other identifiability individually
    count = len(type_list)
    for i, name in enumerate(others_iden_names):
        if name == "rank_prc":
            theoretical_null = 0.5
        else:
            theoretical_null = 0.005
        if selected_only:
            theoretical_null = None
        theoretical_null = None
        n_test = others_iden.shape[1]
        for j, typej in enumerate(type_list):
            path_j = os.path.join(iden_path, typej)

            os.makedirs(path_j,exist_ok=True)
            savepath = os.path.join(path_j, f"{typej}_{name}.png")
            savepath1 = os.path.join(path_j, f"{typej}_{name}_notxt.png")

            paired_violin(others_iden[0,j,i,:][None,:], others_iden[1,j,i,:][None,:], "model", "null", [typej], mean_diffs[count], ps_corrected[count], savepath, figsize=(18,12), fs=30, theoretical_null=theoretical_null)
            paired_violin(others_iden[0,j,i,:][None,:], others_iden[1,j,i,:][None,:], "model", "null", [typej], mean_diffs[count], ps_corrected[count], savepath1, figsize=(18,12), fs=30, theoretical_null=theoretical_null, remove_text=True)
            count += 1


    # plot graph metrics
    graph_metric_path = os.path.join(figpath, "graph_metrics")
    os.makedirs(graph_metric_path, exist_ok=True)
    graph_sum = 0
    graph_count = 0
    graph_max = -np.inf
    graph_max_name = None
    graph_prc_var = np.asarray([])
    null_prc_var = np.asarray([])
    for lvl in graph_levels:
        metric_names = metrics[lvl]
        if not selected_only:
            selected = np.ones((len(metric_names),), dtype=bool)
        else:
            selected = np.zeros((len(metric_names),), dtype=bool)
            for idx in range(selected.shape[0]):
                selected[idx] = True if metric_names[idx] in NETWORK_MEASURE_SELECTED else False
        savepath = os.path.join(graph_metric_path, f"{lvl}.png")
        savepath1 = os.path.join(graph_metric_path, f"{lvl}_notxt.png")
        n_test = values[lvl].shape[0]
        if lvl == "edge":
            figsize = (18,12)
        else:
            figsize = (36,12)
        if selected_only:
            figsize = (18,12)
        count_idx = np.where(selected)[0]
        theoretical_null = 0 if not selected_only else None
        theoretical_null = None
        # paired_violin(values[lvl][selected,:,0], values[lvl][selected,:,1], "model", "null", metric_names[selected], mean_diffs[count:count+n_test], ps_corrected[count:count+n_test], savepath, figsize=figsize, fs=10, theoretical_null=0)
        # paired_violin(values[lvl][selected,:,0], values[lvl][selected,:,1], "model", "null", metric_names[selected], mean_diffs[count:count+n_test], ps_corrected[count:count+n_test], savepath1, figsize=figsize, fs=10, theoretical_null=0, remove_text=True)
        paired_violin(values[lvl][selected,:,0], values[lvl][selected,:,1], "model", "null", metric_names[selected], mean_diffs[count+count_idx], ps_corrected[count+count_idx], savepath, figsize=figsize, fs=10, theoretical_null=theoretical_null)
        paired_violin(values[lvl][selected,:,0], values[lvl][selected,:,1], "model", "null", metric_names[selected], mean_diffs[count+count_idx], ps_corrected[count+count_idx], savepath1, figsize=figsize, fs=10, theoretical_null=theoretical_null, remove_text=True)
        print(f"p {ps_corrected[count+count_idx]}")
        
        count += n_test
        graph_count += n_test
        graph_sum += values[lvl][:,:,0].mean(axis=1).sum()
        print(f"variance explained for {metric_names} are {values[lvl][:,:,0].mean(axis=1)}, average {values[lvl][:,:,0].mean()}")
        
        graph_prc_var = np.append(graph_prc_var, values[lvl][:,:,0].mean(axis=1))
        null_prc_var = np.append(null_prc_var, values[lvl][:,:,1].mean(axis=1))
        if graph_max < values[lvl][:,:,0].mean(axis=1).max():
            graph_max = values[lvl][:,:,0].mean(axis=1).max()
            graph_max_name = metric_names[np.argmax(values[lvl][:,:,0].mean(axis=1))]
    print(f"graph metrics average variance explained {graph_sum / graph_count}")
    print(f"graph metrics max variance explained {graph_max}; max found in {graph_max_name}")
    print(f"model mean variance explained {graph_prc_var.mean()}, std {graph_prc_var.std()}, fisherz mean {fisherz_avg_r(graph_prc_var)}")
    print(f"null mean variance explained {null_prc_var.mean()}, std {null_prc_var.std()}, fisherz mean {fisherz_avg_r(null_prc_var)}")


def impact_of_traits(selected_only, parc='Schaefer7n200p'): # section 3
    if parc=='Schaefer7n200p':
        figpath = "/data/gpfs/projects/punim1278/projects/vaegm/figures/section3" if not selected_only else "/data/gpfs/projects/punim1278/projects/vaegm/figures/section3_selected"
    else:
        figpath = f"/data/gpfs/projects/punim1278/projects/vaegm/figures/{parc}/section3" if not selected_only else f"/data/gpfs/projects/punim1278/projects/vaegm/figures/{parc}/section3_selected"
    
    mean_diffs = np.load(f'/data/gpfs/projects/punim1278/projects/vaegm/data/SC_{parc}/mean_diffs.npy')

    ablation_indices = np.arange(6) # 0 for model, 1-4 for ablation on age_sex, lifestyle factors, body phenotypes, and cognitive measures, repsectively. 5 for null.
    values, metrics = get_system_variations(ablation_indices,parc) #shape (metrics, 1000, ablations) for each metric type (level); netrics: key - metric type(level), value - list of metric names correspond to axis 0 in variable values

    lvls = []
    metric_names = []
    prcs = []
    ps = []
    contrasts = []
    err_bars = []
    
    for key,v in values.items(): # v of shape [n_metric, 1000, n_ablations]
        for i, metric in tqdm(enumerate(metrics[key])):
            data = v[i,:,:] # (1000, n_ablations)
            mu = data.mean(axis=0) # (n_ablations,)
            std = data.std(axis=0) # (n_ablations,)
            model_diff = mu[0] - mu[-1] # float
            trait_diff = mu[0] - mu[1:-1] # (n_traits,)
            prc = np.divide(trait_diff, model_diff) # (n_traits,)
            err_bar = np.abs(np.divide(std[1:-1], model_diff)) # (n_traits,)
            p_mat = np.zeros((prc.shape[0], prc.shape[0])) # (n_traits, n_traits): diagonal denotes if significantly different from model; triu (k=1) denotes significant difference between traits
            contrast_mat = np.zeros((prc.shape[0], prc.shape[0]))
            for j in range(prc.shape[0]):
                for k in range(j, prc.shape[0]):
                    if j==k:
                        _, p = permutation_test(data[:,0], data[:,j+1])
                        contrast = prc[j] - 1
                    else:
                        _, p = permutation_test(data[:,j+1], data[:,k+1])
                        contrast = prc[j] - prc[k]
                    p_mat[j,k] = p
                    contrast_mat[j,k] = contrast
                     
            lvls.append(key)
            metric_names.append(metric)
            prcs.append(prc)
            ps.append(p_mat)
            contrasts.append(contrast_mat)
            err_bars.append(err_bar)

    prcs = np.asarray(prcs) # (n_metrics, n_traits)
    ps = np.asarray(ps) # (n_metrics, n_traits, n_traits)
    contrasts = np.asarray(contrasts) # (n_metrics, n_traits, n_traits)
    err_bars = np.asarray(err_bars) # (n_metrics, n_traits)
    
    #correct ps
    triu_indices = np.triu_indices(ps.shape[-1], k=0)
    ps[:,triu_indices[0], triu_indices[1]] = stats.false_discovery_control(ps[:,triu_indices[0], triu_indices[1]])
    significant = np.zeros(ps.shape, dtype=bool)
    significant[:,triu_indices[0], triu_indices[1]] = ps[:,triu_indices[0], triu_indices[1]] < 0.05

    iden_lvls = ["corr", "demean", "mse"]
    is_iden = np.asarray([lvl in iden_lvls for lvl in lvls], dtype=bool)
    is_rankprc = np.asarray([metric_name == "rank_prc" for metric_name in metric_names], dtype=bool)
    is_top1 = np.asarray([metric_name == "top1_acc" for metric_name in metric_names], dtype=bool)
    is_hungarian = np.asarray([metric_name == "hungarian_acc" for metric_name in metric_names], dtype=bool)
    is_diff = np.asarray([metric_name == "diff_iden" for metric_name in metric_names], dtype=bool)

    select = np.logical_or(np.logical_not(is_iden), is_rankprc)
    selected_metrics = []
    for i, metric in enumerate(metric_names):
        if select[i] and is_iden[i]:
            selected_metrics.append(f"{lvls[i]}_iden")
        elif select[i]:
            selected_metrics.append(metric)
    
    traits = ["age_sex", "lifestyle", "body", "cognitive"]
    prc_path = os.path.join(figpath, "prc")
    os.makedirs(prc_path, exist_ok=True)

    # subgroup_bar_plots(prcs[select,:], selected_metrics, traits, os.path.join(prc_path, "prc.png"), err_bar=err_bars[select,:], figsize=(20,6), fs=20)
    subgroup_bar_plots(prcs[select,:], selected_metrics, traits, os.path.join(prc_path, "prc.png"), err_bar=None, figsize=(20,6), fs=20)
    subgroup_bar_plots(prcs[select,:], selected_metrics, traits, os.path.join(prc_path, "prc_notxt.png"), err_bar=None, figsize=(20,6), fs=20, remove_text=True)

    for key, _ in values.items():
        this_select = np.asarray([lvl == key for lvl in lvls])
        if key in ["edge", "node", "network"] and selected_only:
            in_measures = np.asarray([metric_name in NETWORK_MEASURE_SELECTED for metric_name in metric_names])
            this_select = np.logical_and(this_select,in_measures)
        this_metrics = np.asarray(metric_names)[this_select]
        this_path = os.path.join(figpath, key)
        os.makedirs(this_path, exist_ok=True)
        bar_figsize = (36,12) if key in ["node", "network"] else (18,12)
        if selected_only:
            bar_figsize = (18,12)
        subgroup_bar_plots(prcs[this_select,:], this_metrics, traits, os.path.join(this_path, f"{key}_prc.png"), err_bar=None, figsize=bar_figsize, fs=20)
        subgroup_bar_plots(prcs[this_select,:], this_metrics, traits, os.path.join(this_path, f"{key}_prc_notxt.png"), err_bar=None, figsize=bar_figsize, fs=20, remove_text=True)
        
        contrast_figsize = (20,6) if key in ["node", "network"] else (18,6)
        matrix_plot_contrast_with_significance(contrasts[this_select,:,:], significant[this_select,:,:], this_metrics, traits, os.path.join(this_path, f"{key}_significance.png"), n_rows = 2 if key in ["node", "network"] else 1, fs=20, figsize=contrast_figsize)
    # print((ps[:,triu_indices[0], triu_indices[1]] < 0.05).sum(), ps[:,triu_indices[0], triu_indices[1]].shape)

    # plot for three rank_prc
    rankprc_path = os.path.join(figpath, "rankprc")
    os.makedirs(rankprc_path, exist_ok=True)
    subgroup_bar_plots(prcs[is_rankprc,:], ["corr", "demean", "mse"], traits, os.path.join(rankprc_path, "rank_prc.png"), err_bar=None, figsize=(18,12), fs=20)
    subgroup_bar_plots(prcs[is_rankprc,:], ["corr", "demean", "mse"], traits, os.path.join(rankprc_path, "rank_prc_notxt.png"), err_bar=None, figsize=(18,12), fs=20, remove_text=True)
    matrix_plot_contrast_with_significance(contrasts[is_rankprc,:,:], significant[is_rankprc,:,:], ["corr", "demean", "mse"], traits, os.path.join(rankprc_path, "rank_prc_significance.png"), n_rows = 1, fs=20, figsize=(18,6))

    # plot for three top1 accuracy
    top1_path = os.path.join(figpath, "top1")
    os.makedirs(top1_path, exist_ok=True)
    subgroup_bar_plots(prcs[is_top1,:], ["corr", "demean", "mse"], traits, os.path.join(top1_path, "top1.png"), err_bar=None, figsize=(18,12), fs=20)
    subgroup_bar_plots(prcs[is_top1,:], ["corr", "demean", "mse"], traits, os.path.join(top1_path, "top1_notxt.png"), err_bar=None, figsize=(18,12), fs=20, remove_text=True)
    matrix_plot_contrast_with_significance(contrasts[is_top1,:,:], significant[is_top1,:,:], ["corr", "demean", "mse"], traits, os.path.join(top1_path, "top1_significance.png"), n_rows = 1, fs=20, figsize=(18,6))

    # plot for three hungarian accuracy
    hun_path = os.path.join(figpath, "hun")
    os.makedirs(hun_path, exist_ok=True)
    subgroup_bar_plots(prcs[is_hungarian,:], ["corr", "demean", "mse"], traits, os.path.join(hun_path, "hun.png"), err_bar=None, figsize=(18,12), fs=20)
    subgroup_bar_plots(prcs[is_hungarian,:], ["corr", "demean", "mse"], traits, os.path.join(hun_path, "hun_notxt.png"), err_bar=None, figsize=(18,12), fs=20, remove_text=True)
    matrix_plot_contrast_with_significance(contrasts[is_hungarian,:,:], significant[is_hungarian,:,:], ["corr", "demean", "mse"], traits, os.path.join(hun_path, "hun_significance.png"), n_rows = 1, fs=20, figsize=(18,6))

    # plot for three differential identifiability
    diff_path = os.path.join(figpath, "diff")
    os.makedirs(diff_path, exist_ok=True)
    subgroup_bar_plots(prcs[is_diff,:], ["corr", "demean", "mse"], traits, os.path.join(diff_path, "diff.png"), err_bar=None, figsize=(18,12), fs=20)
    subgroup_bar_plots(prcs[is_diff,:], ["corr", "demean", "mse"], traits, os.path.join(diff_path, "diff_notxt.png"), err_bar=None, figsize=(18,12), fs=20, remove_text=True)
    matrix_plot_contrast_with_significance(contrasts[is_diff,:,:], significant[is_diff,:,:], ["corr", "demean", "mse"], traits, os.path.join(diff_path, "diff_significance.png"), n_rows = 1, fs=20, figsize=(18,6))

    # plot for all identifiability individually (i.e., 3x4=12 figures)
    all_iden_path = os.path.join(figpath, "all_iden")
    os.makedirs(all_iden_path, exist_ok=True)
    iden_types = ["rankprc","top1","hun","diff"]
    iden_idx = [is_rankprc,is_top1,is_hungarian,is_diff]
    similarity_types = ["corr", "demean", "mse"]
    for i, iden_type in enumerate(iden_types):
        for j, similarity_type in enumerate(similarity_types):
            subgroup_bar_plots(prcs[iden_idx[i],:][j,:][None,:], [similarity_type], traits, os.path.join(all_iden_path, f"{iden_type}_{similarity_type}.png"), err_bar=None, figsize=(18,12), fs=20)
            subgroup_bar_plots(prcs[iden_idx[i],:][j,:][None,:], [similarity_type], traits, os.path.join(all_iden_path, f"{iden_type}_{similarity_type}_notxt.png"), err_bar=None, figsize=(18,12), fs=20, remove_text=True)


    # plot for pie chart
    cmap = plt.colormaps["Set3"]
    n_traits = prcs.shape[-1]
    colors = [cmap(i/n_traits) for i in range(n_traits)]
    fig, ax = plt.subplots()
    if np.all(prcs.mean(axis=0) > 0):
        ax.pie(prcs.mean(axis=0), labels=traits, colors=colors, autopct='%1.1f%%', explode=[0.05]*4, shadow=True)
        plt.savefig(os.path.join(figpath, "mean_percentage_contribution_pie.png"))
        for text in ax.texts:
            text.remove()
        plt.savefig(os.path.join(figpath, "mean_percentage_contribution_pie_notxt.png"))
        
    else:
        subgroup_bar_plots(prcs.mean(axis=0)[None,:], [''], traits, os.path.join(figpath, "mean_percentage_contribution_bar.png"), err_bar=None, figsize=(18,12), fs=20)
        subgroup_bar_plots(prcs.mean(axis=0)[None,:], [''], traits, os.path.join(figpath, "mean_percentage_contribution_bar_notxt.png"), err_bar=None, figsize=(18,12), fs=20, remove_text=True)

    plt.close()

    # if reverse the percentage for negative measures
    reverse_sign = np.sign(mean_diffs)
    prcs = np.multiply(prcs, reverse_sign[:,None])
    for key, _ in values.items():
        if key == "network":
            this_select = np.asarray([lvl == key for lvl in lvls])
            if key in ["edge", "node", "network"] and selected_only:
                in_measures = np.asarray([metric_name in NETWORK_MEASURE_SELECTED for metric_name in metric_names])
                this_select = np.logical_and(this_select,in_measures)
            this_metrics = np.asarray(metric_names)[this_select]
            this_path = os.path.join(figpath, key)
            os.makedirs(this_path, exist_ok=True)
            bar_figsize = (36,12) if key in ["node", "network"] else (18,12)
            if selected_only:
                bar_figsize = (18,12)
            subgroup_bar_plots(prcs[this_select,:], this_metrics, traits, os.path.join(this_path, f"{key}_prc_reversed.png"), err_bar=None, figsize=bar_figsize, fs=20)
            subgroup_bar_plots(prcs[this_select,:], this_metrics, traits, os.path.join(this_path, f"{key}_prc_reversed_notxt.png"), err_bar=None, figsize=bar_figsize, fs=20, remove_text=True)
            
            contrast_figsize = (20,6) if key in ["node", "network"] else (18,6)
            matrix_plot_contrast_with_significance(contrasts[this_select,:,:], significant[this_select,:,:], this_metrics, traits, os.path.join(this_path, f"{key}_significance_reversed.png"), n_rows = 2 if ["node", "network"] else 1, fs=20, figsize=contrast_figsize)
    # pie chart
    cmap = plt.colormaps["Set3"]
    n_traits = prcs.shape[-1]
    colors = [cmap(i/n_traits) for i in range(n_traits)]
    fig, ax = plt.subplots()
    if np.all(prcs.mean(axis=0) > 0):
        ax.pie(prcs.mean(axis=0), labels=traits, colors=colors, autopct='%1.1f%%', explode=[0.05]*4, shadow=True)
        plt.savefig(os.path.join(figpath, "mean_percentage_contribution_pie_reversed.png"))
        for text in ax.texts:
            text.remove()
        plt.savefig(os.path.join(figpath, "mean_percentage_contribution_pie_reversed_notxt.png"))
    else:
        subgroup_bar_plots(prcs.mean(axis=0)[None,:], [''], traits, os.path.join(figpath, "mean_percentage_contribution_bar.png"), err_bar=None, figsize=(18,12), fs=20)
        subgroup_bar_plots(prcs.mean(axis=0)[None,:], [''], traits, os.path.join(figpath, "mean_percentage_contribution_bar_notxt.png"), err_bar=None, figsize=(18,12), fs=20, remove_text=True)

    plt.close()




def evaluate_subsystems(): # section 4
    figpath = "/data/gpfs/projects/punim1278/projects/vaegm/figures/section4"

    # Q1: How different metrics' captured inter-individual variations (i.e., identifiability and graph metric variations) differ between subsystems?
    figpath_q1 = os.path.join(figpath, "q1")
    ablation_indices = 0
    values, metrics, system_names = get_subsystem_variations(ablation_indices) # shape (7nets, metrics, 1000) for each metric type (level) because only 1 ablation ("none") selected; metrics: key - metric type(level), value - list of metric names correspond to axis 1 in variable values
    improve_prc = []
    improve_metric = []
    iden_metric = "diff_iden"
    iden_base = 0
    iden_lvls = ["corr", "demean", "mse"]
    graph_lvls = ["edge", "node"]
    filters = []
    for metric_lvl, metric_names in metrics.items():
        for i, metric in tqdm(enumerate(metric_names)):
            savepath = os.path.join(figpath_q1, metric_lvl)
            os.makedirs(savepath, exist_ok=True)
            violin_with_pairwise_test_separate(values[metric_lvl][:,i,:].T, system_names, permutation_test, [os.path.join(savepath, f"{metric}_violin.png"),os.path.join(savepath, f"{metric}_pmat.png")], figsize=(18,12), inner="box", sort=True)
            violin_with_pairwise_test_separate(values[metric_lvl][:,i,:].T, system_names, permutation_test, [os.path.join(savepath, f"{metric}_violin_notxt.png"),os.path.join(savepath, f"{metric}_pmat_notxt.png")], figsize=(18,12), inner="box", sort=True, remove_text=True)
            if (metric_lvl in iden_lvls) and (metric == iden_metric):
                improve_i = values[metric_lvl][:,i,:].mean(axis=-1) - iden_base
                improve_metric.append(f"{metric_lvl}_{metric}")
                improve_prc.append(np.divide(improve_i, improve_i.sum()))
                filters.append(True)
            elif metric_lvl in graph_lvls:
                improve_i = values[metric_lvl][:,i,:].mean(axis=-1)
                improve_metric.append(metric)
                improve_prc.append(np.divide(improve_i, improve_i.sum()))
                if metric in ["local clustering", "local efficiency"]:
                    filters.append(False)
                else:
                    filters.append(True)
            

    improve_prc = np.asarray(improve_prc)
    fig, ax = plt.subplots(figsize=(30,20))
    cax = ax.imshow(improve_prc, cmap='Reds')
    ax.set_xticks(np.arange(improve_prc.shape[-1]))
    ax.set_yticks(np.arange(improve_prc.shape[0]))
    ax.set_xticklabels(system_names)
    ax.set_yticklabels(improve_metric)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    cbar = fig.colorbar(cax, ax=ax)
    cbar.ax.tick_params(labelsize=35)
    ax.tick_params(axis='both', which='major', labelsize=35)
    plt.savefig(os.path.join(figpath_q1, "percentage_improvement.png"))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    cbar.ax.set_yticklabels([])
    plt.savefig(os.path.join(figpath_q1, "percentage_improvement_notxt.png"))
    plt.close(fig)

    violin_with_pairwise_test_separate(np.log(improve_prc), system_names, stats.wilcoxon, [os.path.join(figpath_q1, "improvement_distribution_violin_log.png"), os.path.join(figpath_q1, "improvement_distribution_pmat.png")], figsize=(18,12), inner="points", sort=True, color_p_by='p') # not remove clocal clustering and local efficiency
    violin_with_pairwise_test_separate(improve_prc, system_names, stats.wilcoxon, [os.path.join(figpath_q1, "improvement_distribution_violin.png"), os.path.join(figpath_q1, "improvement_distribution_pmat.png")], figsize=(18,12), inner="points", sort=True, color_p_by='p') # not remove clocal clustering and local efficiency
    violin_with_pairwise_test_separate(np.log(improve_prc), system_names, stats.wilcoxon, [os.path.join(figpath_q1, "improvement_distribution_violin_log_notxt.png"), os.path.join(figpath_q1, "improvement_distribution_pmat_notxt.png")], figsize=(18,12), inner="points", sort=True, remove_text=True, color_p_by='p')
    violin_with_pairwise_test_separate(improve_prc, system_names, stats.wilcoxon, [os.path.join(figpath_q1, "improvement_distribution_violin_notxt.png"), os.path.join(figpath_q1, "improvement_distribution_pmat_notxt.png")], figsize=(18,12), inner="points", sort=True, remove_text=True, color_p_by='p')
    filters = np.asarray(filters)
    violin_with_pairwise_test_separate(np.log(improve_prc[filters,:]), system_names, stats.wilcoxon, [os.path.join(figpath_q1, "filtered_improvement_distribution_violin_log.png"), os.path.join(figpath_q1, "filtered_improvement_distribution_pmat.png")], figsize=(18,12), inner="points", sort=True, color_p_by='p') # not remove clocal clustering and local efficiency
    violin_with_pairwise_test_separate(improve_prc[filters,:], system_names, stats.wilcoxon, [os.path.join(figpath_q1, "filtered_improvement_distribution_violin.png"), os.path.join(figpath_q1, "filtered_improvement_distribution_pmat.png")], figsize=(18,12), inner="points", sort=True, color_p_by='p') # not remove clocal clustering and local efficiency
    subsystem_idx = violin_with_pairwise_test_separate(np.log(improve_prc[filters,:]), system_names, stats.wilcoxon, [os.path.join(figpath_q1, "filtered_improvement_distribution_violin_log_notxt.png"), os.path.join(figpath_q1, "filtered_improvement_distribution_pmat_notxt.png")], figsize=(18,12), inner="points", sort=True, remove_text=True, color_p_by='p')
    subsystem_idx = violin_with_pairwise_test_separate(improve_prc[filters,:], system_names, stats.wilcoxon, [os.path.join(figpath_q1, "filtered_improvement_distribution_violin_notxt.png"), os.path.join(figpath_q1, "filtered_improvement_distribution_pmat_notxt.png")], figsize=(18,12), inner="points", sort=True, remove_text=True, color_p_by='p')
    
    # load hcp test_retest and compare differences by paired permutation test for composition
    hcp = np.load('hcp_test_retest_captured.npz')
    hcp_results = hcp['results']
    hcp_filtered = hcp['results_filtered']
    hcp_diff, hcp_p = compare_composition_paired(hcp_results, improve_prc)
    hcp_diff_filtered, hcp_p_filtered = compare_composition_paired(hcp_results, improve_prc)
    print(f'unfiltered p: {hcp_p}, filtered p: {hcp_p_filtered}')
    
    # Q2. How impacts of traits differ between subsystems? 
    figpath_q2 = os.path.join(figpath, "q2")
    ablation_indices = np.arange(6)
    values, metrics, system_names = get_subsystem_variations(ablation_indices) # shape (7nets, metrics, 1000, ablations) for each metric type (level); metrics: key - metric type(level), value - list of metric names correspond to axis 1 in variable values; system_names
    all_results = []
    ablated_traits = ["age_sex", "lifestyle", "body", "cognitive"]
    metric_mask = []
    metric_name = []

    for metric_lvl, metric_names in metrics.items():
        for i, metric in enumerate(metric_names):
            metric_mu = values[metric_lvl][:,i,:,:].squeeze().mean(axis=1) #shape: (n_sys, ablations)
            all_diff = metric_mu[:,0] - metric_mu[:,-1] # (n_sys)
            # all_diff = np.clip(metric_mu[:,0] - metric_mu[:,-1], a_min=0, a_max=None)
            metric_result = np.divide(metric_mu[:,0][:,None] - metric_mu[:,1:-1], all_diff[:,None]) # shape: (n_sys, 4)
            # metric_result = np.clip(np.divide(np.clip(metric_mu[:,0][:,None] - metric_mu[:,1:-1], a_min=0, a_max=None), all_diff[:,None]), a_min=0, a_max=1) # shape: (n_sys, 4)
            
            fig, ax = plt.subplots(figsize=(20,20))
            cax = ax.imshow(metric_result, cmap='Reds', vmin=0, vmax=1)
            ax.set_xticks(np.arange(metric_result.shape[-1]))
            ax.set_yticks(np.arange(metric_result.shape[0]))
            ax.set_xticklabels(ablated_traits)
            ax.set_yticklabels(system_names)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            cbar = fig.colorbar(cax, ax=ax)
            cbar.ax.tick_params(labelsize=35)
            ax.tick_params(axis='both', which='major', labelsize=35)
            os.makedirs(os.path.join(figpath_q2, metric_lvl), exist_ok=True)
            plt.savefig(os.path.join(figpath_q2, metric_lvl, f"{metric}_interaction.png"))
            plt.close(fig)

            if (metric_lvl in iden_lvls) and (metric == iden_metric):
                all_results.append(metric_result)
                metric_mask.append(True)
                metric_name.append(metric_lvl)
            elif metric_lvl in graph_lvls:
                all_results.append(metric_result)
                if metric in ["local clustering", "local efficiency"]:
                    metric_mask.append(False)
                else:
                    metric_mask.append(True)
                metric_name.append(metric)

    all_results = np.asarray(all_results) # shape n_metrics, n_sys, 4
    print(all_results.sum(axis=-1))
    print(all_results.shape)
    print(np.asarray(metric_mask).sum())

    for i, trait in enumerate(ablated_traits):
        # if trait == "body":
        #     print((all_results[metric_mask,:, i] < 0).sum())
        violin_with_pairwise_test_separate(all_results[metric_mask,:, i], system_names, stats.wilcoxon, [os.path.join(figpath_q2, f"{trait}_interaction_masked_violin.png"), os.path.join(figpath_q2, f"{trait}_interaction_masked_pmat.png")], inner="points", figsize=(18,12), sort=True)
        violin_with_pairwise_test_separate(all_results[metric_mask,:, i], system_names, stats.wilcoxon, [os.path.join(figpath_q2, f"{trait}_interaction_masked_violin_notxt.png"), os.path.join(figpath_q2, f"{trait}_interaction_masked_pmat_notxt.png")], inner="points", figsize=(18,12), sort=True, remove_text=True)
        violin_with_pairwise_test_separate(all_results[:,:, i], system_names, stats.wilcoxon, [os.path.join(figpath_q2, f"{trait}_interaction_violin.png"), os.path.join(figpath_q2, f"{trait}_interaction_pmat.png")], inner="points", figsize=(18,12), sort=True)
        violin_with_pairwise_test_separate(all_results[:,:, i], system_names, stats.wilcoxon, [os.path.join(figpath_q2, f"{trait}_interaction_violin_notxt.png"), os.path.join(figpath_q2, f"{trait}_interaction_pmat_notxt.png")], inner="points", figsize=(18,12), sort=True, remove_text=True)

    test_results = all_results
    # test_results = np.divide(all_results, all_results.sum(axis=-1, keepdims=True))
    # reorder test results before compute pmat
    reordered_test_results = test_results[:,subsystem_idx,:]
    diff_mat, pmat = compare_composition_all_pairs(reordered_test_results[metric_mask,:, :])
    print(diff_mat)
    print(pmat)
    tril_indices = np.tril_indices(pmat.shape[0],k=-1)
    print(pmat[tril_indices])
    print((pmat[tril_indices]<0.05).sum())
    print(diff_mat[tril_indices][pmat[tril_indices]<0.05])
    print((diff_mat[tril_indices][pmat[tril_indices]<0.05] < 0).sum())

    # # visualize using line plots (with box plots) + significance matrix
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(test_results.shape[-1]):
        data_i = test_results[metric_mask,:,i]
        ax.plot(np.arange(data_i.shape[1]), data_i.mean(axis=0))
        ax.boxplot(data_i, positions=np.arange(data_i.shape[1]))
    plt.tight_layout()
    plt.savefig(os.path.join(figpath_q2, "line_plot_with_boxes.png"))
    # plt.close(fig)

    # visualize using barplots + significance matrix'
    # barplots
    bar_cmap = plt.colormaps["Set3"]
    n_traits = test_results.shape[-1]
    bar_colors = [bar_cmap(i/n_traits) for i in range(n_traits)]
    fig, ax = plt.subplots(figsize=(18,12))
    idx = np.arange(test_results.shape[1])
    bottom = np.zeros(idx.shape)
    normalized_results = np.divide(test_results, test_results.sum(axis=-1, keepdims=True))
    # reorder subsystems to align with q1 violin plot
    normalized_results = normalized_results[:,subsystem_idx,:]
    for i in range(n_traits):
        data_i = normalized_results[metric_mask,:,i]
        ax.bar(idx, data_i.mean(axis=0), bottom=bottom, label=ablated_traits[i], color=[bar_colors[i]])
        bottom += data_i.mean(axis=0)

    plt.legend(title="", bbox_to_anchor=(1, 1), loc='lower right', ncol=len(ablated_traits))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(idx)
    ax.set_xticklabels(np.asarray(system_names)[subsystem_idx])
    plt.tight_layout()
    plt.savefig(os.path.join(figpath_q2, "stacked_bar.png"))

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.get_legend().remove()
    plt.savefig(os.path.join(figpath_q2, "stacked_bar_notxt.png"))
    plt.close(fig)
    # significance matrix
    fig, ax = plt.subplots(figsize=(6,6))
    cax = ax.imshow(pmat, cmap='Reds_r')
    ax.set_xticks(np.arange(pmat.shape[0]))
    ax.set_yticks(np.arange(pmat.shape[0]))
    ax.set_xticklabels(np.asarray(system_names)[subsystem_idx])
    ax.set_yticklabels(np.asarray(system_names)[subsystem_idx])
    cbar = fig.colorbar(cax, ax=ax)

    mask_indices = np.where(pmat <= 0.05)
    ax.scatter(mask_indices[1], mask_indices[0], color='black', marker='*', s=100, edgecolor='black')
    ax.tick_params(axis='both', which='major', labelsize=35)
    ax.tick_params(axis='both', which='major', labelsize=35)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(figpath_q2, "subsystem_significance_mat.png"))

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    cbar.ax.set_yticklabels([])
    plt.savefig(os.path.join(figpath_q2, "subsystem_significance_mat_notxt.png"))
    plt.close(fig)


if __name__ == "__main__":
    # model_versus_null(False,'Schaefer7n200p')
    impact_of_traits(False,'Schaefer7n200p')
    # evaluate_subsystems()

    #aparc
    # model_versus_null(False,'aparc')
    # impact_of_traits(False, 'aparc')
    


