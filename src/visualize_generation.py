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

import scipy.stats as stats

def visualize_population():
    traits = pd.read_csv('/data/gpfs/projects/punim1278/projects/vaegm/data/SC_Schaefer7n200p/test_traits.csv',index_col=0)
    age_idx = np.argsort(traits['age'].values)
    n = 100
    young = age_idx[:n]
    old = age_idx[-n:]

    with open(os.path.join("/data/gpfs/projects/punim1278/projects/vaegm/data/SC_Schaefer7n200p/metrics/gen/rep0", "collected.pkl"), "rb") as file:
        data = pickle.load(file)
        data = data["collected"]

    with open(os.path.join("/data/gpfs/projects/punim1278/projects/vaegm/data/SC_Schaefer7n200p/metrics/test", "collected.pkl"), "rb") as file:
        test_data = pickle.load(file)
        test_data = test_data["collected"]

    c_emp = test_data['edge_level']['connectivity']
    c_gen = data['edge_level']['connectivity']

    #compare young old
    young_emp = c_emp[young,:,:].mean(axis=0)
    old_emp = c_emp[old,:,:].mean(axis=0)
    young_gen = c_gen[young,:,:].mean(axis=0)
    old_gen = c_gen[old,:,:].mean(axis=0)
    matices = [young_emp, old_emp, young_emp - old_emp, young_gen, old_gen, young_gen - old_gen]
    matrices = np.asarray(matices)
    contrast_idx = np.asarray([2,5])
    connectivity_idx = np.asarray([0,1,3,4])
    vmin = np.nanmin(matrices[connectivity_idx,:,:])
    vmax = np.nanmax(matrices[connectivity_idx,:,:])
    contrast_max = np.nanmax(np.abs(matrices[contrast_idx,:,:]))
    triu_indices = np.triu_indices(n=matrices.shape[-1],k=1)

    fig_path = '/data/gpfs/projects/punim1278/projects/vaegm/figures/vis_gen/young_old'
    os.makedirs(fig_path,exist_ok=True)
    fs = 5
    fontsize=20
    fig, axes = plt.subplots(3, 3, figsize=(fs*3, fs*3))
    for i, ax in enumerate(axes.flatten()):
        if i<6 and (i+1)%3!=0:
            im = ax.imshow(matrices[i], cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        elif i<6 and (i+1)%3==0:
            im = ax.imshow(matrices[i], cmap='bwr', vmin=-contrast_max, vmax=contrast_max)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            x = matices[i%3][triu_indices]
            y = matices[i%3 + 3][triu_indices]
            im = ax.scatter(x, y)
            ax.tick_params(axis='x', labelsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize)
            coeffs = np.polyfit(x, y, 1)  # Fit a line (degree 1)
            slope, intercept = coeffs
            fit_line = slope * x + intercept  # Compute the line
            ax.plot(x, fit_line, color='black')
            r, p_value = stats.pearsonr(x, y)
            ax.text(0.05, 0.95, f'r = {r:.2f}', transform=ax.transAxes,
                fontsize=fontsize, verticalalignment='top', color='black')
            print(r)
            # ax.set_aspect('equal', adjustable='datalim')
            ax.set_box_aspect(1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            

    plt.tight_layout()
    plt.savefig(os.path.join(fig_path,f'young_old.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # generate each subplot individually (with and without text for last correlations)
    fig_names = ['emp_young','emp_old','emp_young-old','gen_young','gen_old','gen_young-old','corr_young','corr_old','corr_young-old']
    for i, fig_name in enumerate(fig_names):
        fig, ax = plt.subplots(figsize=(fs,fs))
        if i<6 and (i+1)%3!=0:
            im = ax.imshow(matrices[i], cmap='viridis', vmin=vmin, vmax=vmax)
            ax.axis('off')
        elif i<6 and (i+1)%3==0:
            im = ax.imshow(matrices[i], cmap='bwr', vmin=-contrast_max, vmax=contrast_max)
            ax.axis('off')
        else:
            x = matices[i%3][triu_indices]
            y = matices[i%3 + 3][triu_indices]
            im = ax.scatter(x, y)
            ax.tick_params(axis='x', labelsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize)
            coeffs = np.polyfit(x, y, 1)  # Fit a line (degree 1)
            slope, intercept = coeffs
            fit_line = slope * x + intercept  # Compute the line
            ax.plot(x, fit_line, color='black')
            r, p_value = stats.pearsonr(x, y)
            ax.text(0.05, 0.95, f'r = {r:.2f}', transform=ax.transAxes,
                fontsize=fontsize, verticalalignment='top', color='black')
            print(r)
            # ax.set_aspect('equal', adjustable='datalim')
            ax.set_box_aspect(1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(fig_path,f'{fig_name}.png'), dpi=300, bbox_inches='tight')
        
        if i>= 6:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            for text in ax.texts:
                text.set_alpha(0)
            plt.savefig(os.path.join(fig_path,f'{fig_name}_notxt.png'), dpi=300, bbox_inches='tight')
        else:
            cbar=fig.colorbar(im, ax=ax, orientation='vertical')
            cbar.ax.tick_params(labelsize=25)
            plt.savefig(os.path.join(fig_path,f'{fig_name}_colorbar.png'), dpi=300, bbox_inches='tight')
        plt.close()


    # compare male female
    sex_data = traits['sex'].values
    male = sex_data == 1
    female = sex_data == 0

    male_emp = c_emp[male,:,:].mean(axis=0)
    female_emp = c_emp[female,:,:].mean(axis=0)
    male_gen = c_gen[male,:,:].mean(axis=0)
    female_gen = c_gen[female,:,:].mean(axis=0)
    matices = [male_emp, female_emp, male_emp - female_emp, male_gen, female_gen, male_gen - female_gen]
    matrices = np.asarray(matices)
    contrast_idx = np.asarray([2,5])
    connectivity_idx = np.asarray([0,1,3,4])
    vmin = np.nanmin(matrices[connectivity_idx,:,:])
    vmax = np.nanmax(matrices[connectivity_idx,:,:])
    contrast_max = np.nanmax(np.abs(matrices[contrast_idx,:,:]))
    triu_indices = np.triu_indices(n=matrices.shape[-1],k=1)

    fig_path = '/data/gpfs/projects/punim1278/projects/vaegm/figures/vis_gen/male_female'
    os.makedirs(fig_path,exist_ok=True)
    fs = 5
    fig, axes = plt.subplots(3, 3, figsize=(fs*3, fs*3))
    for i, ax in enumerate(axes.flatten()):
        if i<6 and (i+1)%3!=0:
            im = ax.imshow(matrices[i], cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        elif i<6 and (i+1)%3==0:
            im = ax.imshow(matrices[i], cmap='bwr', vmin=-contrast_max, vmax=contrast_max)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            x = matices[i%3][triu_indices]
            y = matices[i%3 + 3][triu_indices]
            im = ax.scatter(x, y)
            ax.tick_params(axis='x', labelsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize)
            coeffs = np.polyfit(x, y, 1)  # Fit a line (degree 1)
            slope, intercept = coeffs
            fit_line = slope * x + intercept  # Compute the line
            ax.plot(x, fit_line, color='black')

            r, p_value = stats.pearsonr(x, y)
            ax.text(0.05, 0.95, f'r = {r:.2f}', transform=ax.transAxes,
                fontsize=fontsize, verticalalignment='top', color='black')
            print(r)

            # ax.set_aspect('equal', adjustable='datalim')
            ax.set_box_aspect(1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            

    plt.tight_layout()
    plt.savefig(os.path.join(fig_path,f'male_female.png'), dpi=300, bbox_inches='tight')
    plt.close()


    # generate each subplot individually (with and without text for last correlations)
    fig_names = ['emp_male','emp_female','emp_male-female','gen_male','gen_female','gen_male-female','corr_male','corr_female','corr_male-female']
    for i, fig_name in enumerate(fig_names):
        fig, ax = plt.subplots(figsize=(fs,fs))
        if i<6 and (i+1)%3!=0:
            im = ax.imshow(matrices[i], cmap='viridis', vmin=vmin, vmax=vmax)
            ax.axis('off')
        elif i<6 and (i+1)%3==0:
            im = ax.imshow(matrices[i], cmap='bwr', vmin=-contrast_max, vmax=contrast_max)
            ax.axis('off')
        else:
            x = matices[i%3][triu_indices]
            y = matices[i%3 + 3][triu_indices]
            im = ax.scatter(x, y)
            ax.tick_params(axis='x', labelsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize)
            coeffs = np.polyfit(x, y, 1)  # Fit a line (degree 1)
            slope, intercept = coeffs
            fit_line = slope * x + intercept  # Compute the line
            ax.plot(x, fit_line, color='black')
            r, p_value = stats.pearsonr(x, y)
            ax.text(0.05, 0.95, f'r = {r:.2f}', transform=ax.transAxes,
                fontsize=fontsize, verticalalignment='top', color='black')
            print(r)
            # ax.set_aspect('equal', adjustable='datalim')
            ax.set_box_aspect(1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(fig_path,f'{fig_name}.png'), dpi=300, bbox_inches='tight')
        
        if i>= 6:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            for text in ax.texts:
                text.set_alpha(0)
            plt.savefig(os.path.join(fig_path,f'{fig_name}_notxt.png'), dpi=300, bbox_inches='tight')
        else:
            cbar=fig.colorbar(im, ax=ax, orientation='vertical')
            cbar.ax.tick_params(labelsize=25)
            plt.savefig(os.path.join(fig_path,f'{fig_name}_colorbar.png'), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    visualize_population()
