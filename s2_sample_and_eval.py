import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import *
from loader import *
from eval import *
from utils import *
from train import *
from tqdm import tqdm

# for graph metrics
import scipy.stats as stats
from concurrent.futures import ProcessPoolExecutor as PPE
import os
import pickle
import pandas as pd
import statsmodels.stats.multitest as multitest
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

def group_features():
    pyargs, settings = parse_args_setting() # get parser and yaml setting details
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # specify device for training/inference
    trait_preprocess = load_trait_preprocessor() # get sklearn trait preprocessor models pretrained on training data
    connectivity_preprocess = load_connectivity_preprocessor() # get sklearn connectivity preprocessor models pretrained on training data

    test_dataset = get_datasets(settings, trait_preprocess, "test_traits.csv")
    test_loader = get_loaders(settings, False, test_dataset)

    # get group average metrics, i.e., group consensus community, and group average graph metrics, if the data storage path does not exist compute from data
    connectivity_mu = connectivity_preprocess["connectivity_scaler"].mean_
    triu_indices = np.triu_indices(settings.n_region, k=1)
    group_metrics_f = os.path.join(settings.data_path, "group_metrics.pkl")
    group_metrics = get_group_average_metrics(group_metrics_f, settings, connectivity_mu, triu_indices)
    group_community = group_metrics["group_community"]
    average_metrics = group_metrics["average_metrics"]
    system_info = pd.read_csv(os.path.join(settings.data_path, "seven_net.csv"), index_col=0)
    system_label = system_info.values[:,1:].astype(bool)
    system_name = system_info.values[:,0]

    vec0 = test_dataset[0]["connectome"]
    vec1 = test_dataset[1]["connectome"]
    vec = np.concatenate((vec0[None,:], vec1[None,:]),axis=0)
    t1 = time.time()
    results = group_graph_metrics(vec, average_metrics, group_community, settings, system_label, triu_indices, is_parallel=True)
    t2 = time.time()
    result0 = results[0]
    for k,v in result0.items():
        for kk, vv in v.items():
            print(kk, vv, np.isnan(vv).sum())
 

    merged = merge_metrics_dict(*results)
    for k,v in merged.items():
        for kk, vv in v.items():
            if (k == "network_level") or "system" in k:
                print(kk, vv, np.isnan(vv).sum())
            else:
                print(kk, np.isnan(vv).sum())
    print(f"{t2-t1}s")
    with open("subject_metrics.pkl", "wb") as file:
        pickle.dump(result0, file)

def merge_metrics(folder_path, n_sbj):
    avail = np.zeros(n_sbj, dtype=bool)
    collected = None
    for i in tqdm(range(n_sbj)):
        f = os.path.join(folder_path, f"sbj_{i}.pkl")
        if os.path.exists(f):
            with open(f, "rb") as file:
                metrics_dict = pickle.load(file)
                collected = merge_metrics_dict(metrics_dict) if collected is None else merge_metrics_dict(collected, metrics_dict)
                avail[i] = True
    
    for k,v in collected.items():
        for kk, vv in v.items():
            collected[k][kk] = np.asarray(vv)
            print(kk, collected[k][kk].shape, np.isnan(collected[k][kk]).sum())

    data = {}
    data["avail"] = avail
    data["collected"] = collected
    print(f"{avail.sum() / avail.shape[0]} subjects collected")
    print(np.where(np.logical_not(avail)))
    with open(os.path.join(folder_path, "collected.pkl"), "wb") as file:
        pickle.dump(data, file)
    
def masked_pearsonr(arr1, arr2, **kwargs):
    mask1 = np.isnan(arr1)
    mask2 = np.isnan(arr2)
    mask3 = np.isinf(arr1)
    mask4 = np.isinf(arr2)
    mask = ~(mask1 | mask2 | mask3 | mask4)
    if mask.sum() > 1:
        return stats.pearsonr(arr1[mask], arr2[mask], **kwargs)
    else:
        return [np.nan, np.nan]

def sample_and_eval(n_sbjs, test_path, gen_path, system_label_f, out_path, ablation, eval_idx, data_path, transform='fisher'):
    f = f"{data_path}/connectivity_preprocess.pkl"
    with open(f, "rb") as file:
        connectivity_preprocess = pickle.load(file)

    idx = np.random.randint(1, 8, size=(n_sbjs,))
    test_data = None
    gen_data = None
    for i, repi in tqdm(enumerate(idx)):
        test_f = os.path.join(test_path, f"sbj_{i}.pkl")
        gen_f = os.path.join(gen_path, f"rep{repi}", f"sbj_{i}.pkl")
        with open(test_f, "rb") as file:
            testi = pickle.load(file)
            test_data = merge_metrics_dict(testi) if test_data is None else merge_metrics_dict(test_data, testi)
        with open(gen_f, "rb") as file:
            geni = pickle.load(file)[ablation]
            gen_data = merge_metrics_dict(geni) if gen_data is None else merge_metrics_dict(gen_data, geni)
    for k,v in test_data.items():
        for kk, vv in v.items():
            test_data[k][kk] = np.asarray(vv)
            gen_data[k][kk] = np.asarray(gen_data[k][kk])
    
    if system_label_f is not None:
        network_df, system_df = eval_correlation(test_data, gen_data, system_label_f,transform)
    else:
        network_df = eval_correlation(test_data, gen_data, system_label_f,transform)
    ablation_path = os.path.join(out_path, ablation)
    os.makedirs(ablation_path, exist_ok=True)
    network_df.to_csv(os.path.join(ablation_path, f"network_eval{eval_idx}.csv"))
    if system_label_f is not None:
        system_df.to_csv(os.path.join(ablation_path, f"system_eval{eval_idx}.csv"))
        system_sorted = system_df.sort_values(by="metrics")
        system_sorted.to_csv(os.path.join(ablation_path, f"system_sorted_eval{eval_idx}.csv"))

    # corr_iden, demean_iden, mse_iden = eval_identifiability(test_data, gen_data, connectivity_preprocess)
    iden = eval_identifiability(test_data, gen_data, system_label_f, connectivity_preprocess)
    cols = ["system", "hungarian_acc", "top1_acc", "diff_iden", "t", "p", "iden_self", "iden_others", "rank_prc"]
    for k,v in iden.items():
        k_df = pd.DataFrame(columns=cols)
        count = 0
        for kk, vv in v.items():
            li = [kk]
            li.extend(list(vv))
            k_df.loc[count] = li
            count += 1
        k_df.to_csv(os.path.join(ablation_path, f"{k}_eval{eval_idx}.csv"))
        
def eval_correlation(test_data, gen_data, system_label_f,transform='fisher'):
    if system_label_f is not None:
        system_names = pd.read_csv(system_label_f, index_col=0).iloc[:,0].values
        system_labels = pd.read_csv(system_label_f, index_col=0).iloc[:,1:].values
        system_df = pd.DataFrame(columns=["metrics_level","metrics", "gen_test_r", "gen_test_p"])
        system_count = 0
    network_df = pd.DataFrame(columns=["metrics_level","metrics", "gen_test_r", "gen_test_p"])
    network_count = 0
    n_region = test_data["edge_level"]["connectivity"].shape[-1]
    triu_indices = np.triu_indices(n_region, k=1)
    row_idx, col_idx = triu_indices

    for level_k, level_v in test_data.items():
        for k, v in level_v.items():
            v_gen = gen_data[level_k][k]
            match level_k:
                case "edge_level":
                    r = np.zeros((n_region, n_region))
                    for i, row in tqdm(enumerate(row_idx)):
                        r[row, col_idx[i]] = masked_pearsonr(v[:,row, col_idx[i]], v_gen[:, row, col_idx[i]])[0]
                    r = r + r.T
                    np.fill_diagonal(r, np.nan)
                    if transform == 'fisher':
                        avg_r = fisherz_avg_r(r)
                    else:
                        signed_square = np.multiply(np.square(r), np.sign(r))
                        rsquared_network = np.nanmean(signed_square)
                        avg_r = np.multiply(np.sign(rsquared_network), np.sqrt(np.abs(rsquared_network)))

                    p_network = stats.ttest_1samp(r[~np.isnan(r)], 0, nan_policy="omit",alternative="greater").pvalue
                    network_df.loc[network_count] = [level_k, k, avg_r, p_network]
                    network_count += 1
                    if system_label_f is not None:
                        for j,system in enumerate(system_names):
                            system_r = r[system_labels[j,:],:]
                            if transform == 'fisher':
                                system_avg_r = fisherz_avg_r(system_r)
                            else:
                                system_signed_square = signed_square[system_labels[j,:],:]
                                rsquared_j = np.nanmean(system_signed_square)
                                system_avg_r = np.multiply(np.sign(rsquared_j), np.sqrt(np.abs(rsquared_j)))

                            p_j = stats.ttest_1samp(system_r[~np.isnan(system_r)], 0, nan_policy="omit",alternative="greater").pvalue
                            system_df.loc[system_count] = [level_k, f"{system}_{k}", system_avg_r, p_j]
                            system_count += 1
                        
                case "node_level":
                    r = np.zeros((n_region,))
                    for i in range(n_region):
                        r[i] = masked_pearsonr(v[:,i], v_gen[:, i])[0]
                    if transform == 'fisher':
                        avg_r = fisherz_avg_r(r)
                    else:
                        signed_square = np.multiply(np.square(r), np.sign(r))
                        rsquared_network = np.nanmean(signed_square)
                        avg_r = np.multiply(np.sign(rsquared_network), np.sqrt(np.abs(rsquared_network)))
 
                    p_network = stats.ttest_1samp(r, 0, nan_policy="omit",alternative="greater").pvalue
                    network_df.loc[network_count] = [level_k, k, avg_r, p_network]
                    network_count += 1
                    if system_label_f is not None:
                        for j,system in enumerate(system_names):
                            system_r = r[system_labels[j,:]]
                            if transform == 'fisher':
                                system_avg_r = fisherz_avg_r(system_r)
                            else:
                                system_signed_square = signed_square[system_labels[j,:]]
                                rsquared_j = np.nanmean(system_signed_square)
                                system_avg_r = np.multiply(np.sign(rsquared_j), np.sqrt(np.abs(rsquared_j)))

                            p_j = stats.ttest_1samp(system_r, 0, nan_policy="omit",alternative="greater").pvalue
                            system_df.loc[system_count] = [level_k, f"{system}_{k}", system_avg_r, p_j]
                            system_count += 1

                case "network_level":
                    r_network, p_network = masked_pearsonr(v, v_gen, alternative="greater")
                    network_df.loc[network_count] = [level_k, k, r_network, p_network]
                    network_count += 1
    
    n_network_metrics = network_df.shape[0]
    if system_label_f is not None:
        n_system_metrics = system_df.shape[0]
        all_p = np.concatenate((network_df["gen_test_r"].values, system_df["gen_test_p"].values))
    else:
        all_p = network_df["gen_test_p"].values

    fdr_result = multitest.fdrcorrection(all_p)[0]
    print(f"{fdr_result.sum()} out of {fdr_result.shape[0]} passed FDR")
    fdr_network = pd.DataFrame(fdr_result[:n_network_metrics], columns=["pass_fdr"])
    network_df = pd.concat((network_df, fdr_network), axis=1)
    if system_label_f is not None:
        fdr_system = pd.DataFrame(fdr_result[n_network_metrics:], columns=["pass_fdr"])
        system_df = pd.concat((system_df, fdr_system), axis=1)
        return network_df, system_df
    else:
        return network_df

def eval_identifiability(test_data, gen_data, system_label_f, connectivity_preprocess, n_samp=200):
    if system_label_f is not None:
        system_names = pd.read_csv(system_label_f, index_col=0).iloc[:,0].values
        system_labels = pd.read_csv(system_label_f, index_col=0).iloc[:,1:].values
    c1 = test_data["edge_level"]["connectivity"]
    c2 = gen_data["edge_level"]["connectivity"]
    selected_sbjs = np.random.choice(c1.shape[0], size=(n_samp,), replace=False)
    c1 = c1[selected_sbjs,:,:]
    c2 = c2[selected_sbjs,:,:]
    n_region = c1.shape[-1]
    triu_indices = np.triu_indices(n_region, k=1)
    c_mu = np.zeros((n_region, n_region))
    c_mu[triu_indices] = connectivity_preprocess["connectivity_scaler"].mean_
    c_mu = c_mu + c_mu.T
    np.fill_diagonal(c_mu, np.nan)

    measures = ["corr", "demean_corr", "mse"]
    iden = {}
    for i, measure in enumerate(measures):
        iden[measure] = {}
    matrices = get_three_matrices(c1, c2, c_mu)
    computed_iden = get_three_identifiability(*matrices)
    for i, measure in enumerate(measures):
        iden[measure]["network"] = computed_iden[i]
    if system_label_f is not None:
        for i, system in enumerate(system_names):
            labeli = system_labels[i,:]
            matrices = get_three_matrices(c1, c2, c_mu, labeli)
            computed_iden = get_three_identifiability(*matrices)
            for j, measure in enumerate(measures):
                iden[measure][system] = computed_iden[j]
    return iden

def eval_identifiability_hcp(test_data, gen_data, system_label_f, connectivity_mu):
    system_names = pd.read_csv(system_label_f, index_col=0).iloc[:,0].values
    system_labels = pd.read_csv(system_label_f, index_col=0).iloc[:,1:].values
    c1 = test_data["edge_level"]["connectivity"]
    c2 = gen_data["edge_level"]["connectivity"]
    n_region = c1.shape[-1]
    triu_indices = np.triu_indices(n_region, k=1)
    c_mu = connectivity_mu
    np.fill_diagonal(c_mu, np.nan)

    measures = ["corr", "demean_corr", "mse"]
    iden = {}
    for i, measure in enumerate(measures):
        iden[measure] = {}
    matrices = get_three_matrices(c1, c2, c_mu)
    computed_iden = get_three_identifiability(*matrices)
    for i, measure in enumerate(measures):
        iden[measure]["network"] = computed_iden[i]
    for i, system in enumerate(system_names):
        labeli = system_labels[i,:]
        matrices = get_three_matrices(c1, c2, c_mu, labeli)
        computed_iden = get_three_identifiability(*matrices)
        for j, measure in enumerate(measures):
            iden[measure][system] = computed_iden[j]
    return iden

def compute_identifiability(inputs, mode="cost"):
    '''
    Evaluate a given metric with linear assignment
    inputs: N-by-N matrix where rows are generated and columns are empirical networks; values are either "cost" or "similarity" measurement
    mode: if "cost", smaller value implies better fit; if "similarity", larger value implies better fit
    '''
    match mode:
        case "cost":
            cost = inputs
            similarity = -inputs
        case "similarity":
            cost = 1 - inputs
            similarity = inputs
        case _:
            raise NotImplementedError("mode can only be either 'cost' or 'similarity'")
    
    if np.isnan(cost).any():
        print("Matrix contains NaN values")
    if np.isinf(cost).any():
        print("Matrix contains infinite values")
    row_idx, col_idx = linear_sum_assignment(cost)
    n = inputs.shape[0]
    hungarian_acc = np.equal(col_idx, np.arange(n)).sum() / n 
    top1_acc = np.equal(np.argmin(cost, axis=-1), np.arange(n)).sum() / n
    iden_self = np.diagonal(similarity)
    iden_others = similarity[~np.eye(similarity.shape[0],dtype=bool)]
    diff_iden = iden_self.mean() - iden_others.mean()
    t, p = stats.ttest_ind(iden_self, iden_others, alternative="greater")
    rank_prc = np.mean(similarity <= iden_self[:,None])
    return hungarian_acc, top1_acc, diff_iden, t, p, iden_self.mean(), iden_others.mean(), rank_prc

def get_three_identifiability(corr, demean_corr, mse):
    corr_iden = compute_identifiability(corr, mode="similarity")
    demean_iden = compute_identifiability(demean_corr, mode="similarity")
    mse_iden = compute_identifiability(mse, mode="cost")
    return corr_iden, demean_iden, mse_iden

def get_three_matrices(c1, c2, c_mu, system_label=None):
    c1_demean = c1 - c_mu[None,:,:]
    c2_demean = c2 - c_mu[None,:,:]
    if system_label is None:
        n_region = c1.shape[-1]
        triu_indices = np.triu_indices(n_region, k=1)
        v1 = c1[:,triu_indices[0], triu_indices[1]]
        v2 = c2[:,triu_indices[0], triu_indices[1]]
        v1_demean = c1_demean[:,triu_indices[0], triu_indices[1]]
        v2_demean = c2_demean[:,triu_indices[0], triu_indices[1]]
    else:
        v1 = np.nanmean(c1[:,system_label,:], axis=1)
        v2 = np.nanmean(c2[:,system_label,:], axis=1)
        v1_demean = np.nanmean(c1_demean[:,system_label,:], axis=1)
        v2_demean = np.nanmean(c2_demean[:,system_label,:], axis=1)
    corr = crosscorr_np(v1, v2)
    demean_corr = crosscorr_np(v1_demean, v2_demean)
    mse = v1[:, np.newaxis, :] - v2[np.newaxis, :, :]
    mse = np.square(mse).mean(axis=-1)
    return corr, demean_corr, mse


def parse_tmp_arguments():
    parser = argparse.ArgumentParser(prog="tmp", description="yaml and checkpoint parser")
    parser.add_argument("--eval", dest="eval", required=True, type=int)
    return parser.parse_args()

def collect_network_data(output_path, n_test=1000):
    ablations = ["none", "age_sex", "lifestyle", "body", "cognitive", "all"]
    f_prefix = ["corr", "demean_corr", "mse", "network"]
    iden_metrics = ["hungarian_acc", "top1_acc", "diff_iden", "rank_prc"]
    n_ablations = len(ablations)
    n_iden = len(iden_metrics)
    edge_metrics = None
    node_metrics = None
    network_metrics = None
    edge_values = None
    node_values = None
    network_values = None
    corr_values = np.zeros((n_iden, n_test, n_ablations))
    demean_values = np.zeros((n_iden, n_test, n_ablations))
    mse_values = np.zeros((n_iden, n_test, n_ablations))

    for i, ablation in tqdm(enumerate(ablations)):
        folder_path = os.path.join(output_path, ablation)
        for j, prefix in enumerate(f_prefix):
            for k in range(n_test):
                f = os.path.join(folder_path, f"{prefix}_eval{k}.csv")
                df = pd.read_csv(f, index_col=0)
                match prefix:
                    case "network":
                        edge_rows = df["metrics_level"] == "edge_level"
                        node_rows = df["metrics_level"] == "node_level"
                        network_rows = df["metrics_level"] == "network_level"
                        if (edge_metrics is None) and (edge_values is None):
                            edge_metrics = df.loc[edge_rows,"metrics"].values
                            edge_values = np.zeros((len(edge_metrics), n_test, n_ablations))
                        if (node_metrics is None) and (node_values is None):
                            node_metrics = df.loc[node_rows,"metrics"].values
                            node_values = np.zeros((len(node_metrics), n_test, n_ablations))
                        if (network_metrics is None) and (network_values is None):
                            network_metrics = df.loc[network_rows,"metrics"].values
                            network_values = np.zeros((len(network_metrics), n_test, n_ablations))
                        edge_values[:,k,i] = df.loc[edge_rows,"gen_test_r"].values
                        node_values[:,k,i] = df.loc[node_rows,"gen_test_r"].values
                        network_values[:,k,i] = df.loc[network_rows,"gen_test_r"].values
                    case "corr":
                        corr_values[:,k,i] = df.loc[0,iden_metrics].values
                    case "demean_corr":
                        demean_values[:,k,i] = df.loc[0,iden_metrics].values
                    case "mse":
                        mse_values[:,k,i] = df.loc[0,iden_metrics].values
    results = {}
    results["ablations"] = ablations
    results["iden_metrics"] = iden_metrics
    results["edge_metrics"] = edge_metrics
    results["node_metrics"] = node_metrics
    results["network_metrics"] = network_metrics
    results["corr_values"] = corr_values
    results["demean_values"] = demean_values
    results["mse_values"] = mse_values
    results["edge_values"] = edge_values
    results["node_values"] = node_values
    results["network_values"] = network_values
    with open(os.path.join(output_path, "network_data.pkl"), "wb") as file:
        pickle.dump(results, file)

                    
def collect_system_data(output_path, system_label_f, n_test=1000):
    system_names = pd.read_csv(system_label_f, index_col=0).iloc[:,0].values

    for idx, system in enumerate(system_names):
        ablations = ["none", "age_sex", "lifestyle", "body", "cognitive", "all"]
        f_prefix = ["corr", "demean_corr", "mse", "system"]
        iden_metrics = ["hungarian_acc", "top1_acc", "diff_iden", "rank_prc"]
        n_ablations = len(ablations)
        n_iden = len(iden_metrics)
        edge_metrics = None
        node_metrics = None
        edge_values = None
        node_values = None
        corr_values = np.zeros((n_iden, n_test, n_ablations))
        demean_values = np.zeros((n_iden, n_test, n_ablations))
        mse_values = np.zeros((n_iden, n_test, n_ablations))

        for i, ablation in tqdm(enumerate(ablations)):
            folder_path = os.path.join(output_path, ablation)
            for j, prefix in enumerate(f_prefix):
                for k in range(n_test):
                    f = os.path.join(folder_path, f"{prefix}_eval{k}.csv")
                    df = pd.read_csv(f, index_col=0)
                    match prefix:
                        case "system":
                            edge_rows = (df["metrics_level"] == "edge_level") & (df["metrics"].str.contains(system))
                            node_rows = (df["metrics_level"] == "node_level") & (df["metrics"].str.contains(system))
                            network_rows = (df["metrics_level"] == "network_level") & (df["metrics"].str.contains(system))
                            if (edge_metrics is None) and (edge_values is None):
                                t = df.loc[edge_rows,"metrics"].str.split("_", expand=True)
                                edge_metrics = t.iloc[:,1].values
                                edge_values = np.zeros((len(edge_metrics), n_test, n_ablations))
                            if (node_metrics is None) and (node_values is None):
                                t = df.loc[node_rows,"metrics"].str.split("_", expand=True)
                                node_metrics = t.iloc[:,1].values
                                node_values = np.zeros((len(node_metrics), n_test, n_ablations))
                            edge_values[:,k,i] = df.loc[edge_rows,"gen_test_r"].values
                            node_values[:,k,i] = df.loc[node_rows,"gen_test_r"].values
                        case "corr":
                            corr_values[:,k,i] = df.loc[idx+1,iden_metrics].values
                        case "demean_corr":
                            demean_values[:,k,i] = df.loc[idx+1,iden_metrics].values
                        case "mse":
                            mse_values[:,k,i] = df.loc[idx+1,iden_metrics].values
        results = {}
        results["ablations"] = ablations
        results["iden_metrics"] = iden_metrics
        results["edge_metrics"] = edge_metrics
        results["node_metrics"] = node_metrics
        results["corr_values"] = corr_values
        results["demean_values"] = demean_values
        results["mse_values"] = mse_values
        results["edge_values"] = edge_values
        results["node_values"] = node_values
        with open(os.path.join(output_path, f"{system}_data.pkl"), "wb") as file:
            pickle.dump(results, file)


def sample_and_eval_acc(n_sbjs, test_path, gen_path):
    connectivity_preprocess = load_connectivity_preprocessor()
    idx = np.random.randint(1, 21, size=(n_sbjs,))
    test_data = None
    gen_data = None
    for i, repi in tqdm(enumerate(idx)):
        test_f = os.path.join(test_path, f"sbj_{i}.pkl")
        gen_f = os.path.join(gen_path, f"rep{repi}", f"sbj_{i}.pkl")
        with open(test_f, "rb") as file:
            testi = pickle.load(file)
            test_data = merge_metrics_dict(testi) if test_data is None else merge_metrics_dict(test_data, testi)
        with open(gen_f, "rb") as file:
            geni = pickle.load(file)["none"]
            gen_data = merge_metrics_dict(geni) if gen_data is None else merge_metrics_dict(gen_data, geni)
    r = np.zeros((n_sbjs,))
    triu_idx = np.triu_indices(200,k=1)
    for i in range(n_sbjs):
        r[i] = stats.pearsonr(test_data['edge_level']['connectivity'][i][triu_idx], gen_data['edge_level']['connectivity'][i][triu_idx])[0]
    print(f'mean: {r.mean()}, std: {r.std()}')

def fisherz_avg_r(r_values):
    valid_r_values = r_values[~np.isnan(r_values)] 
    if len(valid_r_values) == 0:
        return np.nan
    z_values = np.arctanh(valid_r_values)  # Fisher z-transform
    z_mean = np.mean(z_values[~np.isinf(z_values)])
    r_avg = np.tanh(z_mean) 
    return r_avg

if __name__ == "__main__":
    # 1617
    # # 6469
    
    ## tmp1. first sample and evaluate model performance (network w/ or w/o system depending on parc), run in parallel for bootstrapping
    # pyargs = parse_tmp_arguments()
    # # parc = 'aparc'
    # parc = 'Schaefer7n200p'
    # data_path = f"/data/gpfs/projects/punim1278/projects/vaegm/data/SC_{parc}"
    # metrics_path = f"/data/gpfs/projects/punim1278/projects/vaegm/data/SC_{parc}/metrics"
    # system_label_f = f"/data/gpfs/projects/punim1278/projects/vaegm/data/SC_{parc}/seven_net.csv" if 'Schaefer' in parc else None
    # out_path = "/data/gpfs/projects/punim1278/projects/vaegm/output" if 'Schaefer' in parc else f"/data/gpfs/projects/punim1278/projects/vaegm/output/{parc}"
    # os.makedirs(out_path, exist_ok=True)
    # test_path = os.path.join(metrics_path, "test")
    # gen_path = os.path.join(metrics_path, "gen")
    # ablations = ["none", "age_sex", "lifestyle", "body", "cognitive", "all"]
    # # ablation = "none"
    # n_sbjs = 1617
    # for i, ablation in enumerate(ablations):
    #     sample_and_eval(n_sbjs, test_path, gen_path, system_label_f, out_path, ablation, pyargs.eval, data_path)


    # tmp2. used to collect network (optionally also system depending on parc) data
    # parc = 'aparc'
    parc = 'Schaefer7n200p'
    if 'Schaefer' in parc:
        system_label_f = "/data/gpfs/projects/punim1278/projects/vaegm/data/SC_Schaefer7n200p/seven_net.csv"
        out_path = "/data/gpfs/projects/punim1278/projects/vaegm/output"
        collect_network_data(out_path)
        collect_system_data(out_path, system_label_f)
    else:
        out_path = f"/data/gpfs/projects/punim1278/projects/vaegm/output/{parc}"
        collect_network_data(out_path)
