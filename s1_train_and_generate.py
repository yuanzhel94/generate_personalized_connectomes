import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot

import pandas as pd

from model import *
from loader import *
from eval import *
from utils import *
from train import *
from tqdm import tqdm
import time
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def train_model():
    print('running')
    pyargs, settings = parse_args_setting() # get parser and yaml setting details
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # specify device for training/inference
    trait_preprocess = load_trait_preprocessor(settings) # get sklearn trait preprocessor models pretrained on training data
    connectivity_preprocess = load_connectivity_preprocessor(settings) # get sklearn connectivity preprocessor models pretrained on training data

    train_dataset, test_dataset = get_datasets(settings, trait_preprocess, "train_traits.csv", "test_traits.csv")
    train_loader, test_loader = get_loaders(settings, True, train_dataset, test_dataset)

    epoch = 0
    model = construct_model(settings).to(device)
    optimizer = construct_optimizer(settings, model)
    epoch, model, optimizer = load_checkpoint(pyargs, settings, [epoch, model, optimizer], ["epoch", "model", "optimizer"], [False, True, True])
    writer = SummaryWriter(log_dir=settings.tensorboard_path)

    for epoch_idx in tqdm(range(epoch, epoch + settings.epochs)):
        train_loss = train_epoch(settings, model, optimizer, train_loader, connectivity_preprocess, writer, epoch_idx, device)
        test_loss, examples = test_model(settings, model, optimizer, test_loader, connectivity_preprocess, writer, epoch_idx, device)
        print(f"train loss: {train_loss}; test loss: {test_loss}")
        save_example_images(settings, examples, f"{settings.generation_path}/epoch{epoch_idx}.png")

        if epoch_idx % 10 == 0:
            save_checkpoint(os.path.join(settings.checkpoint_path, f"checkpoint_{epoch_idx}.pt"), [epoch_idx, model, optimizer], ["epoch", "model", "optimizer"], [False, True, True])

def group_features():
    # 1617 (0-1616) test subjects and 6469 (0-6468) train subjects
    pyargs, settings = parse_args_setting() # get parser and yaml setting details
    trait_preprocess = load_trait_preprocessor(settings) # get sklearn trait preprocessor models pretrained on training data
    connectivity_preprocess = load_connectivity_preprocessor(settings) # get sklearn connectivity preprocessor models pretrained on training data
    match pyargs.data_spec:
        case "train":
            dataset = get_datasets(settings, trait_preprocess, "train_traits.csv")
        case _:
            dataset = get_datasets(settings, trait_preprocess, "test_traits.csv")
    
    sbj_idx = pyargs.sbj_idx
    sbj_data = dataset[sbj_idx]

    # get group average metrics, i.e., group consensus community, and group average graph metrics, if the data storage path does not exist compute from data
    connectivity_mu = connectivity_preprocess["connectivity_scaler"].mean_
    triu_indices = np.triu_indices(settings.n_region, k=1)
    group_metrics_f = os.path.join(settings.data_path, "group_metrics.pkl")
    group_metrics = get_group_average_metrics(group_metrics_f, settings, connectivity_mu, triu_indices)
    group_community = group_metrics["group_community"]
    average_metrics = group_metrics["average_metrics"]
    system_info = system_label = system_name = None
    has_subsystem = 'Schaefer' in settings.parc
    if has_subsystem:
        system_info = pd.read_csv(os.path.join(settings.data_path, "seven_net.csv"), index_col=0)
        system_label = system_info.values[:,1:].astype(bool)
        system_name = system_info.values[:,0]
    

    if pyargs.data_spec == "gen":
        traits_raw = dataset.traits_raw.iloc[sbj_idx,:].values
        # imputed_values = trait_preprocess["trait_imputer"].statistics_
        trait_masks = pd.read_csv(os.path.join(settings.data_path, "trait_mask.csv"), index_col=0)
        t1 = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # specify device for training/inference
        traits = np.tile(traits_raw, (6,1))
        for i in range(trait_masks.shape[1]):
            maski = trait_masks.iloc[:,i].values
            traits[i+1,maski] = np.nan
        traits[-1,:] = np.nan
        ablations = ["none", "age_sex", "lifestyle", "body", "cognitive", "all"]
        traits = preprocess_traits(traits, trait_preprocess)
        traits = torch.from_numpy(traits).float().to(device)
        epoch = 0
        model = construct_model(settings).to(device)
        optimizer = construct_optimizer(settings, model)
        epoch, model, optimizer = load_checkpoint(pyargs, settings, [epoch, model, optimizer], ["epoch", "model", "optimizer"], [False, True, True])

        with torch.no_grad():
            model.eval()
            gen = model.sample(traits)
        gen_raw, gen_demean = inverse_connectivity_preprocess(gen.detach().cpu().numpy(), connectivity_preprocess, settings.apply_pca, settings.n_components)
        t2 = time.time()
        print(f"generation costs {t2 - t1} sec")
        t3 = time.time()
        results = group_graph_metrics(gen_raw, average_metrics, group_community, settings, system_label, triu_indices, is_parallel=False)
        t4 = time.time()
        print(f"evaluation costs {t4 - t3} sec")
        result = {}
        for i, ablation in enumerate(ablations):
            result[ablation] = results[i]
    else:
        connectivity = sbj_data["connectome"][None,:]
        result = group_graph_metrics(connectivity, average_metrics, group_community, settings, system_label, triu_indices, is_parallel=False)[0]

    if (pyargs.data_spec == "gen") and (pyargs.rep is not None):
        outpath = os.path.join(settings.data_path, "metrics", pyargs.data_spec, f"rep{pyargs.rep}")
    else:
        outpath = os.path.join(settings.data_path, "metrics", pyargs.data_spec)
    os.makedirs(outpath, exist_ok=True)
    
    with open(os.path.join(outpath, f"sbj_{sbj_idx}.pkl"), "wb") as file:
        pickle.dump(result, file)

def visualize_net():
    pyargs, settings = parse_args_setting() # get parser and yaml setting details
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # specify device for training/inference
    trait_preprocess = load_trait_preprocessor() # get sklearn trait preprocessor models pretrained on training data
    connectivity_preprocess = load_connectivity_preprocessor() # get sklearn connectivity preprocessor models pretrained on training data

    train_dataset, test_dataset = get_datasets(settings, trait_preprocess, "train_traits.csv", "test_traits.csv")
    train_loader, test_loader = get_loaders(settings, True, train_dataset, test_dataset)

    epoch = 0
    model = construct_model(settings).to(device)
    optimizer = construct_optimizer(settings, model)
    epoch, model, optimizer = load_checkpoint(pyargs, settings, [epoch, model, optimizer], ["epoch", "model", "optimizer"], [False, True, True])
    writer = SummaryWriter(log_dir=settings.tensorboard_path)
    print(model)

    for i, data in enumerate(train_loader):
        x_raw = data["connectome"].numpy() # have been log-scaled but not been preprocessed (standardized, and pca if settings.apply_pca is true) yet
        x, x_demean = preprocess_connectivity(x_raw, connectivity_preprocess, settings.use_raw, settings.apply_pca, settings.n_components)
        x = torch.from_numpy(x) # standardize and pca if settings.apply_pca is true and convert to tensor
        traits = data["traits"] # have been preprocessed (mean imputed and standardized)
        x = x.float().to(device)
        traits = traits.float().to(device)
        y, mu, logvar = model(x, traits)
        make_dot(y, params=dict(list(model.named_parameters()))).render("train_torchviz", format="png")
        break

    model.zero_grad()
    for i, data in enumerate(train_loader):
        x_raw = data["connectome"].numpy() # have been log-scaled but not been preprocessed (standardized, and pca if settings.apply_pca is true) yet
        x, x_demean = preprocess_connectivity(x_raw, connectivity_preprocess, settings.use_raw, settings.apply_pca, settings.n_components)
        x = torch.from_numpy(x) # standardize and pca if settings.apply_pca is true and convert to tensor
        traits = data["traits"] # have been preprocessed (mean imputed and standardized)
        x = x.float().to(device)
        traits = traits.float().to(device)
        gen = model.sample(traits)
        make_dot(gen, params=dict(list(model.named_parameters()))).render("gen_torchviz", format="png")
        break

def visualize_example_connectomes():
    # generate example connectome figures for model illustration
    pyargs, settings = parse_args_setting() # get parser and yaml setting details
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # specify device for training/inference
    trait_preprocess = load_trait_preprocessor() # get sklearn trait preprocessor models pretrained on training data
    connectivity_preprocess = load_connectivity_preprocessor() # get sklearn connectivity preprocessor models pretrained on training data

    train_dataset, test_dataset = get_datasets(settings, trait_preprocess, "train_traits.csv", "test_traits.csv")
    train_loader, test_loader = get_loaders(settings, True, train_dataset, test_dataset)

    epoch = 0
    model = construct_model(settings).to(device)
    optimizer = construct_optimizer(settings, model)
    epoch, model, optimizer = load_checkpoint(pyargs, settings, [epoch, model, optimizer], ["epoch", "model", "optimizer"], [False, True, True])
    writer = SummaryWriter(log_dir=settings.tensorboard_path)

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            x_raw = data["connectome"].numpy() # have been log-scaled but not been preprocessed (standardized, and pca if settings.apply_pca is true) yet
            x, x_demean = preprocess_connectivity(x_raw, connectivity_preprocess, settings.use_raw, settings.apply_pca, settings.n_components)
            x = torch.from_numpy(x) # standardize and pca if settings.apply_pca is true and convert to tensor
            traits = data["traits"] # have been preprocessed (mean imputed and standardized)
            x = x.float().to(device)
            traits = traits.float().to(device)
            y, mu, logvar = model(x, traits)
            y = y.detach().cpu().numpy()
            y_raw, y_demean = inverse_connectivity_preprocess(y, connectivity_preprocess, settings.apply_pca, settings.n_components)
            gen = model.sample(traits)
            gen = gen.detach().cpu().numpy()
            gen_raw, gen_demean = inverse_connectivity_preprocess(gen, connectivity_preprocess, settings.apply_pca, settings.n_components)
            break
    
    savepath = "/data/gpfs/projects/punim1278/projects/vaegm/figures/connectomes"
    os.makedirs(savepath, exist_ok=True)
    n = settings.n_region
    cmap1 = "viridis"
    # cmap2 = "bwr"
    save_connectome(x_raw[0,:], n, os.path.join(savepath, "x_raw_triu.png"), cmap=cmap1, triu_boarder=True)
    save_connectome(y_raw[0,:], n, os.path.join(savepath, "y_raw_triu.png"), cmap=cmap1, triu_boarder=True)
    save_connectome(gen_raw[0,:], n, os.path.join(savepath, "gen_raw_triu.png"), cmap=cmap1, triu_boarder=True)

    save_connectome(x_raw[0,:], n, os.path.join(savepath, "x_raw.png"), cmap=cmap1)
    save_connectome(y_raw[0,:], n, os.path.join(savepath, "y_raw.png"), cmap=cmap1)
    save_connectome(gen_raw[0,:], n, os.path.join(savepath, "gen_raw.png"), cmap=cmap1)

    # save_connectome(x[0,:], n, os.path.join(savepath, "x.png"), cmap=cmap2, triu_boarder=True)
    # save_connectome(y[0,:], n, os.path.join(savepath, "y.png"), cmap=cmap2, triu_boarder=True)
    # save_connectome(gen[0,:], n, os.path.join(savepath, "gen.png"), cmap=cmap2, triu_boarder=True)
    
def visualize_graph_metrics_identifiability():
    n_samp = 1
    with open(os.path.join("/data/gpfs/projects/punim1278/projects/vaegm/data/SC_Schaefer7n200p/metrics/gen/rep0", "collected.pkl"), "rb") as file:
        data = pickle.load(file)
        data = data["collected"]
    with open(os.path.join("/data/gpfs/projects/punim1278/projects/vaegm/data/SC_Schaefer7n200p/metrics/test", "collected.pkl"), "rb") as file:
        test_data = pickle.load(file)
        test_data = test_data["collected"]
    ebc = data["edge_level"]["edge betweenness centrality"]
    n = ebc.shape[-1]
    triu_indices = np.triu_indices(n, k=1)
    savepath = "/data/gpfs/projects/punim1278/projects/vaegm/figures/connectomes"
    for i in range(n_samp):
        vi = ebc[i,:,:][triu_indices]
        vi = np.log10(vi)
        save_connectome(vi, n, os.path.join(savepath, f"ebc{i}.png"), cmap='viridis', triu_boarder=False)
    strengths = data["node_level"]["nodal strengths"][:300,:]
    plt.imshow(strengths, cmap='viridis', interpolation='none')
    plt.axis("off")
    plt.savefig(os.path.join(savepath,f"strength.png"), dpi=300, bbox_inches='tight')
    plt.close()
    charpath = data['network_level']['charpath'][:20]
    idx = np.random.choice(charpath.shape[0], size=charpath.shape, replace=False)
    charpath1 = charpath[idx]
    plt.figure(figsize=(50,30))
    plt.imshow(np.concatenate((charpath[:,None],charpath1[:,None]),axis=1), cmap='viridis', interpolation='none')
    plt.axis("off")
    plt.savefig(os.path.join(savepath,f"charpath.png"), dpi=300, bbox_inches='tight')
    plt.close()
    # identifiability
    c = data["edge_level"]["connectivity"]
    test_c = test_data["edge_level"]["connectivity"]
    idx = np.random.choice(c.shape[0], size=(50,), replace=False)
    c = c[idx,:,:]
    test_c = test_c[idx,:,:]
    row, col = triu_indices
    connectivity_preprocess = load_connectivity_preprocessor()
    c_mu = connectivity_preprocess["connectivity_scaler"].mean_
    c = c[:, row, col] - c_mu[None,:]
    test_c = test_c[:, row, col] - c_mu[None,:]
    demean_corr = crosscorr_np(c, test_c, eps=1e-8)
    print(demean_corr.shape)
    plt.imshow(demean_corr, cmap='viridis', interpolation='none')
    plt.axis("off")
    plt.savefig(os.path.join(savepath,f"similarity_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    


def save_connectome(v, n, fig_path, cmap='viridis', triu_boarder=False): # convert upper triangular vector into matrix and save
    mat = np.zeros((n, n))
    triu_indices = np.triu_indices(n, k=1)
    mat[triu_indices] = v
    mat = mat + mat.T

    plt.imshow(mat, cmap=cmap, interpolation='none')
    if triu_boarder:
    # Draw the triangular border
        plt.plot([0.5, n-0.5], [-0.5, -0.5], color='yellow', linewidth=10)   # Bottom horizontal line
        plt.plot([n-0.5, n-0.5], [-0.5, n-0.5], color='yellow', linewidth=10)  # Right vertical line
        plt.plot([0.5, n-0.5], [-0.5, n-0.5], color='yellow', linewidth=5)    # Diagonal line
    plt.axis("off")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # train_model()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    group_features()       
    # visualize_net()
    # visualize_example_connectomes()
    # visualize_graph_metrics_identifiability()
