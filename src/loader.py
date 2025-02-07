import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from utils import *


class TraitConnectomeDataset(Dataset):
    def __init__(self, settings, trait_csv, trait_preprocess):
        trait_df = pd.read_csv(os.path.join(settings.data_path, trait_csv), index_col=0)
        self.subject_ids = trait_df["Subject_ID"].values
        traits = trait_df.iloc[:,1:]
        self.traits_raw = traits
        self.traits = preprocess_traits(traits, trait_preprocess)
        self.settings = settings
        self.triu_indices = np.triu_indices(settings.n_region, k=1)

    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]
        subject_traits = self.traits[idx,:]
        subject_connectome = np.loadtxt(os.path.join(self.settings.connectome_path, subject_id, f"{self.settings.modality}_{self.settings.parc}+{self.settings.subcortex}.csv.gz"), delimiter=",")
        subject_connectome = np.log10(subject_connectome[:-self.settings.n_subcortex,:-self.settings.n_subcortex] + 1) if self.settings.log_scale else subject_connectome[:-self.settings.n_subcortex,:-self.settings.n_subcortex]
        subject_connectome = subject_connectome[self.triu_indices]
        return {"ID": subject_id, "traits": subject_traits, "connectome": subject_connectome}

    def __len__(self):
        return len(self.subject_ids)

def get_datasets(settings, trait_preprocess, *trait_csvs):
    datasets = []
    if not trait_csvs:
        raise Exception("trait_csv file is not given")
    for trait_csv in trait_csvs:
        datasets.append(TraitConnectomeDataset(settings, trait_csv, trait_preprocess))
    return tuple(datasets) if len(datasets) > 1 else datasets[0]

def get_loaders(settings, shuffle, *datasets):
    loaders = []
    if not datasets:
        raise Exception("dataset is not given")
    for dataset in datasets:
        loaders.append(DataLoader(dataset, batch_size=settings.batchsize, shuffle=shuffle))
    return tuple(loaders) if len(loaders) > 1 else loaders[0]


