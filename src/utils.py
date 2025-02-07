import numpy as np
import pandas as pd
import pickle
import argparse
import re
import os
import yaml
import matplotlib.pyplot as plt
import torch

def load_trait_preprocessor(settings):
    f = f"{settings.data_path}/trait_preprocess.pkl"
    with open(f, "rb") as file:
        trait_preprocess = pickle.load(file)
    return trait_preprocess

def load_connectivity_preprocessor(settings):
    f = f"{settings.data_path}/connectivity_preprocess.pkl"
    with open(f, "rb") as file:
        connectivity_preprocess = pickle.load(file)
    return connectivity_preprocess

def preprocess_traits(traits, trait_preprocess):
    preprocessed = trait_preprocess["trait_imputer"].transform(traits) # mean imputation for all
    preprocessed[:,trait_preprocess["trait_continuous"]] = trait_preprocess["trait_scaler"].transform(preprocessed[:,trait_preprocess["trait_continuous"]]) # standardization for continuous variables
    return preprocessed

def ablate_and_preprocess_traits(traits, trait_preprocess, ablation_indx):
    preprocessed = traits.copy()
    preprocessed[:, ablation_indx] = np.nan
    preprocessed = trait_preprocess["trait_imputer"].transform(preprocessed) # mean imputation for all
    preprocessed[:, trait_preprocess["trait_continuous"]] = trait_preprocess["trait_scaler"].transform(preprocessed[:,trait_preprocess["trait_continuous"]]) # standardization for continuous variables
    return preprocessed

def preprocess_connectivity(c, connectivity_preprocess, use_raw=False, apply_pca=False, n_components=1024):
    demean = c - connectivity_preprocess["connectivity_scaler"].mean_[None,:]
    if use_raw:
        return c, demean
    else:
        preprocessed = connectivity_preprocess["connectivity_scaler"].transform(c)
        if apply_pca:
            assert n_components <= 1024
            preprocessed = connectivity_preprocess["connectivity_pca"].transform(preprocessed)[:,:n_components]
        return preprocessed, demean

def inverse_connectivity_preprocess(scores, connectivity_preprocess, use_raw=False, apply_pca=False, n_components=1024):
    inverted = scores.copy()
    if not use_raw:
        if apply_pca:
            assert n_components <= 1024
            inverted = np.dot(scores, connectivity_preprocess["connectivity_pca"].components_[:n_components, :]) + connectivity_preprocess["connectivity_pca"].mean_
        inverted = connectivity_preprocess["connectivity_scaler"].inverse_transform(inverted)
    inverted = np.clip(inverted, 0, None) # ensure non-negative edges
    demean = inverted - connectivity_preprocess["connectivity_scaler"].mean_[None,:]
    return inverted, demean

def load_yaml(yaml_f):
    '''
    load the model setting from yaml file yaml_f, as a dictionary
    '''
    with open(yaml_f,'r') as f:
        return yaml.safe_load(f)

class Args():
    '''
    initialize study setting from yaml file
    '''
    def __init__(self, yaml_f):
        setting_dict = load_yaml(yaml_f)
        for k, v in setting_dict.items():
            setattr(self, k, v)

def parse_arguments():
    parser = argparse.ArgumentParser(prog="dgm", description="yaml and checkpoint parser")
    parser.add_argument("yaml_f", type=os.path.abspath, help="yaml file specifying the model and data settings")
    parser.add_argument("-c","--checkpoint",dest="checkpoint_f",required=False,type=os.path.abspath,default=None)
    parser.add_argument("--dataspec", dest="data_spec", required=False, choices=["train", "test", "gen"])
    parser.add_argument("--sbjidx", dest="sbj_idx", required=False, type=int)
    parser.add_argument("--rep", dest="rep", required=False, type=int, default=None)
    return parser.parse_args()

def parse_args_setting():
    pyargs = parse_arguments()
    yaml_f = pyargs.yaml_f
    setting = Args(yaml_f)
    print(f"model setting loaded from yaml file: {yaml_f}")
    return pyargs, setting

def load_checkpoint_file(cp_path, obj, k, is_state_dict):
    '''
    load checkpoint from file path specified by cp_path
        cp_path: path to checkpoint file
        obj: list of values/objects to be loaded
        k: keys for each obj to be loaded
        is_state_idct: for each obj, if load their state_dict()
    '''
    checkpoint = torch.load(cp_path) if torch.cuda.is_available() else torch.load(cp_path, map_location=torch.device('cpu'))
    loaded_obj = []
    for i, obji in enumerate(obj):
        if is_state_dict[i]:
            obji.load_state_dict(checkpoint[k[i]])
        else:
            obji = checkpoint[k[i]]
        loaded_obj.append(obji)
    return loaded_obj

def find_latest_checkpoint(cp_folder, re_pattern=re.compile(r"checkpoint_\d+\.pt")):
    '''
    find the latest checkpoint in cp_folder
    '''
    file_list = os.listdir(cp_folder)
    cp_list = [cp for cp in file_list if re_pattern.match(cp)]
    if not cp_list:
        return None
    else:
        cp_list = sorted(cp_list, key=lambda x: int(re.findall(r"\d+", x)[0]))
        return os.path.join(cp_folder, cp_list[-1])

def load_checkpoint(pyargs, settings, obj, k, is_state_dict):
    '''
    if pyargs.checkpoint_f exists and is a file, load checkpoints from the file
    elif pyargs.checkpoint_f is not a file, load the latest checkpoints (search in settings.checkpoint_path)
    else: do not load check points
    '''
    if pyargs.checkpoint_f is not None:
        if os.path.isfile(pyargs.checkpoint_f):
            loaded_obj = load_checkpoint_file(pyargs.checkpoint_f, obj, k, is_state_dict)
            print(f"checkpoint loaded from specified file {pyargs.checkpoint_f}")
        elif find_latest_checkpoint(settings.checkpoint_path, re_pattern=re.compile(r"checkpoint_\d+\.pt")):
            latest_checkpoint = find_latest_checkpoint(settings.checkpoint_path, re_pattern=re.compile(r"checkpoint_\d+\.pt"))
            loaded_obj = load_checkpoint_file(latest_checkpoint, obj, k, is_state_dict)
            print(f"checkpoint loaded from latest checkpoint file {latest_checkpoint}")
        else:
            loaded_obj = obj
            print(f"no checkpoints found, train from scratch")
    else:
        loaded_obj = obj
        print("required to train from scratch")
    return loaded_obj

def save_checkpoint(cp_path, obj, k, is_state_dict):
    '''
    save checkpoint to file path specified by cp_path
        cp_path: path to checkpoint file
        obj: list of values/objects to be stored
        k: keys for each obj to be stored
        is_state_idct: for each obj, if store their state_dict()
    '''
    assert len(obj) == len(k)
    assert len(obj) == len(is_state_dict)
    torch.save({k[i]: obj[i].state_dict() if is_state_dict[i] else obj[i] for i in range(len(obj))}, cp_path)

def save_adj_fig_triples(adj1, adj2, adj3, path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    cax1 = ax1.imshow(adj1)
    fig.colorbar(cax1, ax=ax1)
    cax2 = ax2.imshow(adj2)
    fig.colorbar(cax2, ax=ax2)  
    cax3 = ax3.imshow(adj3)
    fig.colorbar(cax3, ax=ax3) 
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)

def save_example_images(settings, examples, path):
    n_region = settings.n_region
    triu_indices = np.triu_indices(n_region, k=1)
    adjs = []
    for _, example in enumerate(examples):
        adj = np.zeros((settings.n_region, settings.n_region))
        adj[triu_indices] = example
        adj = adj + np.transpose(adj)
        adjs.append(adj)
    save_adj_fig_triples(*adjs, path=path)
        

def get_system_variations(ablation_indices,parc="Schaefer7n200p"):
    if parc=="Schaefer7n200p":
        f_path = "/data/gpfs/projects/punim1278/projects/vaegm/output/network_data.pkl"
    else:
        f_path = f"/data/gpfs/projects/punim1278/projects/vaegm/output/{parc}/network_data.pkl"
    metric_types = ["corr", "demean", "mse", "edge", "node", "network"]
    values = {}
    metrics = {}
    for metric_type in metric_types:
        values[metric_type] = None
        metrics[metric_type] = None
    metric_saved = False
    with open(f_path, "rb") as file:
        data = pickle.load(file)
    if not metric_saved:
        metrics["corr"] = data["iden_metrics"]
        metrics["demean"] = data["iden_metrics"]
        metrics["mse"] = data["iden_metrics"]
        metrics["edge"] = data["edge_metrics"]
        node_metrics = data["node_metrics"]
        for i, node_metric in enumerate(node_metrics):
            if node_metric == "local":
                node_metrics[i] = "local assortativity"
        metrics["node"] = node_metrics
        metrics["network"] = data["network_metrics"]
        metric_saved = True
    for metric_type in metric_types:
        values[metric_type] = data[f"{metric_type}_values"][:,:,ablation_indices]
    return values, metrics #shape (metrics, 1000, ablations) for each metric type (level); netrics: key - metric type(level), value - list of metric names correspond to axis 0 in variable values

def get_subsystem_variations(ablation_indices):
    system_label_f = "/data/gpfs/projects/punim1278/projects/vaegm/data/SC_Schaefer7n200p/seven_net.csv"
    pkl_path = "/data/gpfs/projects/punim1278/projects/vaegm/output"
    system_names = pd.read_csv(system_label_f, index_col=0).iloc[:,0].values
    metric_types = ["corr", "demean", "mse", "edge", "node"]
    values = {}
    metrics = {}
    metric_saved = False
    for metric_type in metric_types:
        values[metric_type] = []
        metrics[metric_type] = None
    
    for i, name in enumerate(system_names):
        with open(os.path.join(pkl_path, f"{name}_data.pkl"), "rb") as file:
            data = pickle.load(file)
        if not metric_saved:
            metrics["corr"] = data["iden_metrics"]
            metrics["demean"] = data["iden_metrics"]
            metrics["mse"] = data["iden_metrics"]
            metrics["edge"] = data["edge_metrics"]
            node_metrics = data["node_metrics"]
            for i, node_metric in enumerate(node_metrics):
                if node_metric == "local":
                    node_metrics[i] = "local assortativity"
            metrics["node"] = node_metrics
            metric_saved = True
        for metric_type in metric_types:
            values[metric_type].append(data[f"{metric_type}_values"][:,:,ablation_indices])
    
    for metric_type in metric_types:
        values[metric_type] = np.asarray(values[metric_type])
    return values, metrics, system_names # shape (7nets, metrics, 1000, ablations) for each metric type (level); metrics: key - metric type(level), value - list of metric names correspond to axis 1 in variable values; system_names


if __name__ == "__main__":
    pass
    # trait_scaler, connectivity_scaler, connectivity_pca = load_scaler_pca()
    # print(trait_scaler.mean_, trait_scaler.mean_.shape)
    # print(connectivity_scaler.mean_, connectivity_scaler.mean_.shape)
    # print(connectivity_pca.mean_, connectivity_pca.mean_.shape)


