import json
import numpy as np
import os
from eval import *
import time
import pandas as pd
import argparse
import warnings
import pickle
from tmp import eval_correlation, eval_identifiability_hcp
import scipy.stats as stats


def load_cmat_7net_assignment(data_path, sbj_idx):
    npy_f = os.path.join(data_path, sbj_idx, f"{sbj_idx}_schaefer200p7n_atlas_probabilistic_structural_connectivity_with_subcortex.npy")
    json_f = os.path.join(data_path, sbj_idx, "schaefer200p7n_atlas_label_names_ordered_list.json")
    c = np.load(npy_f)
    df_7net, idx_select = get_7nets_hcp(json_f)
    c = c[idx_select,:][:,idx_select]
    return c, df_7net

def get_all_c(dirpath):
    sbj_indices = [name for name in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, name))]
    all_c = []
    for i, sbj_idx in enumerate(sbj_indices):
        c, df_7net = load_cmat_7net_assignment(dirpath, sbj_idx)
        all_c.append(c)
    all_c = np.asarray(all_c)
    all_c = np.log10(all_c + 1)
    return all_c, df_7net, sbj_indices

def get_7nets_hcp(json_f):
    systems = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    names = ["VIS", "SMN", "DAN", "VAN", "LIM", "ECN", "DMN"]
    with open(json_f) as f:
        regions = json.load(f)
    assignment = np.zeros((len(systems), len(regions)), dtype=bool)
    for i, system in enumerate(systems):
        result = [system in region for region in regions]
        assignment[i,:] = np.asarray(result)
    idx_select = assignment.sum(axis=0) != 0
    assignment = assignment[:,idx_select]
    df = pd.concat([pd.DataFrame(names, columns=["7nets"]), pd.DataFrame(assignment, columns=np.asarray(regions)[idx_select].tolist())], axis=1)
    return df, idx_select

def evaluate_test_retest(sbj_idx):
    t1 = time.time()
    test_path = "/data/gpfs/projects/punim1278/projects/vaegm/data/hcp_test_retest/HCP_test"
    retest_path = "/data/gpfs/projects/punim1278/projects/vaegm/data/hcp_test_retest/HCP_retest"
    others_path = "/data/gpfs/projects/punim1278/projects/vaegm/data/hcp_test_retest/HCP_others"

    test_c, test_7net, test_sbjs = get_all_c(test_path)
    retest_c, retest_7net, retest_sbjs = get_all_c(retest_path)
    others_c, others_7net, others_sbjs = get_all_c(others_path)
    c_mu = others_c.mean(axis=0)

    assert (np.array_equal(test_7net.values[:,1:], retest_7net.values[:,1:]) and np.array_equal(test_7net.values[:,1:], others_7net.values[:,1:]))
    system_info = test_7net
    system_label = system_info.values[:,1:].astype(bool)
    system_name = system_info.values[:,0]

    print(test_sbjs == retest_sbjs)

    class DummySettings:
        pass
    dummy_settings = DummySettings()
    n_region = c_mu.shape[-1]
    setattr(dummy_settings, 'n_region', n_region)

    triu_indices = np.triu_indices(c_mu.shape[-1], k=1)
    rows, cols = triu_indices
    test_vec = test_c[:,rows,cols]
    retest_vec = retest_c[:,rows,cols]
    mu_vec = c_mu[rows,cols]

    group_average_f = "/data/gpfs/projects/punim1278/projects/vaegm/data/hcp_test_retest/group_average.pkl"
    group_metrics = get_group_average_metrics(group_average_f, dummy_settings, mu_vec, triu_indices)
    group_community = group_metrics["group_community"]
    average_metrics = group_metrics["average_metrics"]
    
    t2 = time.time()
    test_result = group_graph_metrics(test_vec[sbj_idx,:][None,:], average_metrics, group_community, dummy_settings, system_label, triu_indices=triu_indices, is_parallel=False)[0]
    retest_result = group_graph_metrics(retest_vec[sbj_idx,:][None,:], average_metrics, group_community, dummy_settings, system_label, triu_indices=triu_indices, is_parallel=False)[0]
    t3 = time.time()
    print(test_result)
    print(f"{t2-t1} s", f"{t3-t2} s")
    test_outpath = "/data/gpfs/projects/punim1278/projects/vaegm/data/hcp_test_retest/test_result"
    retest_outpath = "/data/gpfs/projects/punim1278/projects/vaegm/data/hcp_test_retest/retest_result"
    with open(os.path.join(test_outpath, f"sbj_{sbj_idx}.pkl"), "wb") as file:
        pickle.dump(test_result, file)
    with open(os.path.join(retest_outpath, f"sbj_{sbj_idx}.pkl"), "wb") as file:
        pickle.dump(retest_result, file)

def parse_hcp_arguments():
    parser = argparse.ArgumentParser(prog="hcp", description="test retest captured variations in subsystems")
    parser.add_argument("--sbjidx", dest="sbj_idx", required=True, type=int)
    return parser.parse_args()

def merge_metrics():
    test_outpath = "/data/gpfs/projects/punim1278/projects/vaegm/data/hcp_test_retest/test_result"
    retest_outpath = "/data/gpfs/projects/punim1278/projects/vaegm/data/hcp_test_retest/retest_result"
    n = 42
    test_data = None
    retest_data = None
    for sbj_idx in range(n):
        test_f = os.path.join(test_outpath, f"sbj_{sbj_idx}.pkl")
        retest_f = os.path.join(retest_outpath, f"sbj_{sbj_idx}.pkl")
        with open(test_f, "rb") as file:
            testi = pickle.load(file)
            test_data = merge_metrics_dict(testi) if test_data is None else merge_metrics_dict(test_data, testi)
        with open(retest_f, "rb") as file:
            retesti = pickle.load(file)
            retest_data = merge_metrics_dict(retesti) if retest_data is None else merge_metrics_dict(retest_data, retesti)
    for k,v in test_data.items():
        for kk, vv in v.items():
            test_data[k][kk] = np.asarray(vv)
            retest_data[k][kk] = np.asarray(retest_data[k][kk])
    return test_data, retest_data

def analyze_test_retest_variations(test_data, retest_data):
    others_path = "/data/gpfs/projects/punim1278/projects/vaegm/data/hcp_test_retest/HCP_others"
    others_c, others_7net, others_sbjs = get_all_c(others_path)
    c_mu = others_c.mean(axis=0)
    system_info_f = "/data/gpfs/projects/punim1278/projects/vaegm/data/hcp_test_retest/system_info.csv"
    if not os.path.exists(system_info_f):
        parc_json = "/data/gpfs/projects/punim1278/projects/vaegm/data/hcp_test_retest/HCP_test/103818/schaefer200p7n_atlas_label_names_ordered_list.json"
        system_info, _ = get_7nets_hcp(parc_json)
        system_info.to_csv("/data/gpfs/projects/punim1278/projects/vaegm/data/hcp_test_retest/system_info.csv")
    network_df, system_df = eval_correlation(test_data, retest_data, system_info_f)
    iden = eval_identifiability_hcp(test_data, retest_data, system_info_f, c_mu)
    system_names = pd.read_csv(system_info_f, index_col=0).iloc[:,0].values
    edge_metrics = test_data['edge_level'].keys()
    node_metrics = test_data['node_level'].keys()
    iden_types = ["corr", "demean_corr","mse"]
    n_metrics = len(edge_metrics) + len(node_metrics) + len(iden_types)
    results = np.zeros((n_metrics, len(system_names)))
    metric_filter = []
    filtered_metrics = ["local clustering", "local efficiency"]
    metric_names = []
    for i, name in enumerate(system_names):
        count = 0
        for iden_type in iden_types:
            results[count,i] = iden[iden_type][name][2] #diff_iden
            if i==0:
                metric_filter.append(iden_type in filtered_metrics)
                metric_names.append(iden_type)
            count += 1
        for metric in edge_metrics:
            results[count,i] = system_df.loc[system_df["metrics"] == f"{name}_{metric}", "gen_test_r"]
            if i==0:
                metric_filter.append(metric in filtered_metrics)
                metric_names.append(metric)
            count += 1
        for metric in node_metrics:
            results[count,i] = system_df.loc[system_df["metrics"] == f"{name}_{metric}", "gen_test_r"]
            if i==0:
                metric_filter.append(metric in filtered_metrics)
                metric_names.append(metric)
            count += 1
        
    metric_filter = np.logical_not(np.asarray(metric_filter))
    prc_results = np.divide(results,results.sum(axis=1,keepdims=True))
    metric_names = np.asarray(metric_names)

    figpath = "/data/gpfs/projects/punim1278/projects/vaegm/figures/hcp_test_retest"
    os.makedirs(figpath, exist_ok=True)
    fig, ax = plt.subplots(figsize=(30,20))
    cax = ax.imshow(prc_results, cmap='Reds')
    ax.set_xticks(np.arange(prc_results.shape[-1]))
    ax.set_yticks(np.arange(prc_results.shape[0]))
    ax.set_xticklabels(system_names)
    ax.set_yticklabels(metric_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    cbar = fig.colorbar(cax, ax=ax)
    cbar.ax.tick_params(labelsize=35)
    ax.tick_params(axis='both', which='major', labelsize=35)
    plt.savefig(os.path.join(figpath, "percentage_improvement.png"))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    cbar.ax.set_yticklabels([])
    plt.savefig(os.path.join(figpath, "percentage_improvement_notxt.png"))
    plt.close(fig)

    violin_with_pairwise_test_separate(prc_results, system_names, stats.wilcoxon, [os.path.join(figpath, "improvement_distribution_violin.png"), os.path.join(figpath, "improvement_distribution_pmat.png")], figsize=(18,12), inner="points", sort=True, color_p_by='p') # not remove clocal clustering and local efficiency
    violin_with_pairwise_test_separate(prc_results, system_names, stats.wilcoxon, [os.path.join(figpath, "improvement_distribution_violin_notxt.png"), os.path.join(figpath, "improvement_distribution_pmat_notxt.png")], figsize=(18,12), inner="points", sort=True, remove_text=True, color_p_by='p')

    violin_with_pairwise_test_separate(prc_results[metric_filter,:], system_names, stats.wilcoxon, [os.path.join(figpath, "filtered_improvement_distribution_violin.png"), os.path.join(figpath, "filtered_improvement_distribution_pmat.png")], figsize=(18,12), inner="points", sort=True, color_p_by='p') # not remove clocal clustering and local efficiency
    violin_with_pairwise_test_separate(prc_results[metric_filter,:], system_names, stats.wilcoxon, [os.path.join(figpath, "filtered_improvement_distribution_violin_notxt.png"), os.path.join(figpath, "filtered_improvement_distribution_pmat_notxt.png")], figsize=(18,12), inner="points", sort=True, remove_text=True, color_p_by='p')

    return prc_results, prc_results[metric_filter,:]

if __name__ == "__main__":
    # warnings.filterwarnings("ignore", category=RuntimeWarning)
    # pyargs = parse_hcp_arguments()
    # evaluate_test_retest(pyargs.sbj_idx)

    test_data, retest_data = merge_metrics()
    results, results_filtered = analyze_test_retest_variations(test_data, retest_data)
    np.savez('hcp_test_retest_captured.npz', results=results, results_filtered=results_filtered)
    print(results.shape, results_filtered.shape)
    print(results.sum(axis=1))
    
