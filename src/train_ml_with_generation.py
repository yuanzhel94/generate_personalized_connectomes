
import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import argparse

def train_model(test_idx):
    sample_sizes = [8, 16, 32, 64, 128, 256, 512] # old starts from 4
    n_rep = 10
    gen_ratio = 0.5
    sbj_trait = pd.read_csv("/data/gpfs/projects/punim1278/projects/vaegm/data/SC_Schaefer7n200p/test_traits.csv",index_col=0)
    c_path = "/data/gpfs/projects/punim1278/projects/dgm/data/connectomes"
    c_file = "SC_Schaefer7n200p+Tian_Subcortex_S1_3T.csv.gz"
    n_subcortex = 16
    n_nodes = 200
    n_edges = int(n_nodes * (n_nodes - 1) /2)
    triu_indices = np.triu_indices(n_nodes, k=1)

    age_mae = np.zeros((4,len(sample_sizes),n_rep))
    sex_acc = np.zeros((4,len(sample_sizes),n_rep))
    for i, n_select in enumerate(sample_sizes):
        for rep in range(n_rep):
            train_emp, train_mix, train_syn, train_avg, test, train_age_sex, train_age_sex_avg, test_age_sex = select_data(n_select, gen_ratio, sbj_trait, c_path, c_file, n_subcortex, triu_indices, n_edges)
            # unique_sex_counts = len(np.unique(train_age_sex[:,-1]))
            # while unique_sex_counts != 2:
            #     train_emp, train_mix, train_syn, train_avg, test, train_age_sex, train_age_sex_avg, test_age_sex = select_data(n_select, gen_ratio, sbj_trait, c_path, c_file, n_subcortex, triu_indices, n_edges)
            #     unique_sex_counts = len(np.unique(train_age_sex[:,-1]))
            L_age_emp = Lasso()
            L_sex_emp = LogisticRegression(penalty='l1', solver='liblinear')
            L_age_mix = Lasso()
            L_sex_mix = LogisticRegression(penalty='l1', solver='liblinear')
            L_age_syn = Lasso()
            L_sex_syn = LogisticRegression(penalty='l1', solver='liblinear')
            L_age_avg = Lasso()
            L_sex_avg = LogisticRegression(penalty='l1', solver='liblinear')

            L_age_emp.fit(train_emp, train_age_sex[:,0])
            L_age_mix.fit(train_mix, train_age_sex[:,0])
            L_age_syn.fit(train_syn, train_age_sex[:,0])
            L_age_avg.fit(train_avg, train_age_sex_avg[:,0])
            age_emp_predict = L_age_emp.predict(test)
            age_mix_predict = L_age_mix.predict(test)
            age_syn_predict = L_age_syn.predict(test)
            age_avg_predict = L_age_avg.predict(test)
            age_emp_mae = np.abs(age_emp_predict - test_age_sex[:,0]).mean()
            age_mix_mae = np.abs(age_mix_predict - test_age_sex[:,0]).mean()
            age_syn_mae = np.abs(age_syn_predict - test_age_sex[:,0]).mean()
            age_avg_mae = np.abs(age_avg_predict - test_age_sex[:,0]).mean()

            L_sex_emp.fit(train_emp, train_age_sex[:,1])
            L_sex_mix.fit(train_mix, train_age_sex[:,1])
            L_sex_syn.fit(train_syn, train_age_sex[:,1])
            L_sex_avg.fit(train_avg, train_age_sex_avg[:,1])
            sex_emp_predict = L_sex_emp.predict(test)
            sex_mix_predict = L_sex_mix.predict(test)
            sex_syn_predict = L_sex_syn.predict(test)
            sex_avg_predict = L_sex_avg.predict(test)
            sex_emp_acc = np.equal(sex_emp_predict, test_age_sex[:,1]).mean()
            sex_mix_acc = np.equal(sex_mix_predict, test_age_sex[:,1]).mean()
            sex_syn_acc = np.equal(sex_syn_predict, test_age_sex[:,1]).mean()
            sex_avg_acc = np.equal(sex_avg_predict, test_age_sex[:,1]).mean()

            age_mae[:,i,rep] = np.asarray([age_emp_mae, age_mix_mae, age_syn_mae, age_avg_mae])
            sex_acc[:,i,rep] = np.asarray([sex_emp_acc, sex_mix_acc, sex_syn_acc, sex_avg_acc])

            print(f"size {n_select}, rep {rep}, diff_age {age_emp_mae - age_mix_mae} and {age_emp_mae - age_syn_mae}, diff_sex {sex_emp_acc - sex_mix_acc} and {sex_emp_acc - sex_syn_acc}")
    savepath = "/data/gpfs/projects/punim1278/projects/vaegm/output/ml_with_gen"
    os.makedirs(savepath, exist_ok=True)
    np.savez(os.path.join(savepath, f'train_ml_with_generation_{test_idx}.npz'), age_mae=age_mae, sex_acc=sex_acc)
    print(f"mean_age_diff {(age_mae[0,:,:] - age_mae[1,:,:]).mean(axis=1)} and {(age_mae[0,:,:] - age_mae[2,:,:]).mean(axis=1)}")
    print(f"mean_sex_diff {(sex_acc[0,:,:] - sex_acc[1,:,:]).mean(axis=1)} and {(sex_acc[0,:,:] - sex_acc[2,:,:]).mean(axis=1)}")  

def select_data(n_select, gen_ratio, emp_trait, c_path, c_file, n_subcortex, triu_indices, n_edges):
    n_sbj = emp_trait.shape[0]
    rep_avail = np.arange(1,21)
    sbj_rep_pairs = np.random.choice(rep_avail, size=(n_sbj,))

    valid=False
    while not valid:
        select_idx = np.sort(np.random.choice(n_sbj, size=(n_select,), replace=False)) # 0:n_sbj
        n_gen = np.rint(n_select * gen_ratio).astype(int)
        gen_idx_in_select = np.random.choice(n_select,size=(n_gen,), replace=False) # 0:n_select
        avg_src_idx = np.setdiff1d(np.arange(n_select), gen_idx_in_select)

        age_sex = emp_trait[["age","sex"]].values
        train_age_sex = age_sex[select_idx,:]

        n_M = train_age_sex[avg_src_idx,-1].sum()
        n_F = n_select - n_gen - n_M

        if n_M >= 2 and n_F >= 2:
            valid=True

    M_src_idx = np.where(train_age_sex[avg_src_idx,-1] == 1)[0]
    F_src_idx = np.where(train_age_sex[avg_src_idx,-1] == 0)[0]

    test_mask = np.ones((n_sbj,), dtype=bool)
    test_mask[select_idx] = False
    test_age_sex = age_sex[test_mask,:]

    train_emp = np.zeros((n_select, n_edges))
    test = np.zeros((n_sbj - n_select,n_edges))

    train_count = 0
    test_count = 0
    for i in range(n_sbj):
        subject_id = emp_trait["Subject_ID"][i]
        ci = np.loadtxt(os.path.join(c_path, subject_id, c_file), delimiter=",")
        ci = np.log10(ci[:-n_subcortex,:-n_subcortex] + 1)
        ci = ci[triu_indices]

        if i in select_idx:
            train_emp[train_count,:] = ci
            train_count += 1
        else:
            test[test_count,:] = ci
            test_count += 1

    
    train_mix = train_emp.copy()
    for row_idx in gen_idx_in_select:
        sbj_idx = select_idx[row_idx]
        with open(f"/data/gpfs/projects/punim1278/projects/vaegm/data/SC_Schaefer7n200p/metrics/gen/rep{sbj_rep_pairs[sbj_idx]}/sbj_{sbj_idx}.pkl", "rb") as file:
            gen_data = pickle.load(file)
            gen_c = gen_data["none"]["edge_level"]["connectivity"]
            train_mix[row_idx,:] = gen_c[triu_indices]

    train_syn = train_emp.copy()
    syn_count = 0
    for i in range(n_sbj):
        if i in select_idx:
            with open(f"/data/gpfs/projects/punim1278/projects/vaegm/data/SC_Schaefer7n200p/metrics/gen/rep{sbj_rep_pairs[i]}/sbj_{i}.pkl", "rb") as file:
                gen_data = pickle.load(file)
                gen_c = gen_data["none"]["edge_level"]["connectivity"]
                train_syn[syn_count,:] = gen_c[triu_indices]
                syn_count += 1

    train_avg = train_emp.copy()
    train_age_sex_avg = train_age_sex.copy()
    for row_idx in gen_idx_in_select:
        if train_age_sex[row_idx,-1] == 1:
            idx = M_src_idx
        else:
            idx = F_src_idx
        select_src = np.random.choice(avg_src_idx[idx], size=(2,),replace=False)
        train_avg[row_idx,:] = train_emp[select_src,:].mean(axis=0)
        train_age_sex_avg[row_idx,0] = train_age_sex[select_src,0].mean()

    return train_emp, train_mix, train_syn, train_avg, test, train_age_sex, train_age_sex_avg, test_age_sex

def collect_results():
    folder_path = "/data/gpfs/projects/punim1278/projects/vaegm/output/ml_with_gen"
    age_mae = []
    sex_acc = []
    n_tests = 100
    for i in range(n_tests):
        f_path = os.path.join(folder_path,f'train_ml_with_generation_{i}.npz')
        if os.path.exists(f_path):
            results_i = np.load(f_path)
            age_mae.append(results_i["age_mae"])
            sex_acc.append(results_i["sex_acc"])
    age_mae = np.asarray(age_mae).transpose(1,2,0,3)
    sex_acc = np.asarray(sex_acc).transpose(1,2,0,3)
    shape=age_mae.shape
    age_mae = age_mae.reshape(shape[0],shape[1],-1)
    sex_acc = sex_acc.reshape(shape[0],shape[1],-1)
    np.savez(os.path.join(folder_path, 'train_ml_with_generation_collected.npz'), age_mae=age_mae, sex_acc=sex_acc)
    return os.path.join(folder_path, 'train_ml_with_generation_collected.npz')

def bootstrap_results(results,n_bootstrap): #results: np.ndarray of shape (3,4,n)
    out = []
    for i in range(n_bootstrap):
        selected_idx = np.random.choice(results.shape[-1],size=(results.shape[-1],))
        bootstrap_results = results[:,:,selected_idx].mean(axis=-1)
        out.append(bootstrap_results)
    out = np.asarray(out).transpose(1,2,0)
    prc5 = np.percentile(out,5,axis=-1)
    prc95 = np.percentile(out,95,axis=-1)
    return prc5, prc95

def remove_all_text(fig):
    for ax in fig.get_axes():
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        for text in ax.texts:
            text.set_text('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

def make_plots(f):
    figsize = (8,6)
    results = np.load(f)
    age_mae = results["age_mae"] # (train_type, sample_size, rep) - (3,4,n): train_type: [emp, mix, gen]; sample_size: [50,100,500,1000]; rep: n, in this case, 1000
    sex_acc = results["sex_acc"]

    train_type = ["empirical", "mix-trait", "generated", "mix-avg"]
    sample_sizes = [8, 16, 32, 64, 128, 256, 512]# old starts from 4
    x = np.arange(len(sample_sizes))

    outpath = "/data/gpfs/projects/punim1278/projects/vaegm/figures/ml_comp"
    os.makedirs(outpath, exist_ok=True)
    
    n_bootstrap = 100
    age_mean = np.mean(age_mae, axis=-1)
    age_prc5, age_prc95 = bootstrap_results(age_mae,n_bootstrap)
    sex_mean = np.mean(sex_acc, axis=-1)
    sex_prc5, sex_prc95 = bootstrap_results(sex_acc,n_bootstrap)

    cmap = plt.colormaps["tab10"]
    colors = [cmap(i/len(train_type)) for i in range(len(train_type))]
    # plot for age
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(age_mae.shape[0]):
        ax.plot(x, age_mean[i,:], label=train_type[i], color=colors[i],linewidth=2,marker='o',markersize=12)
        ax.fill_between(x, age_prc5[i,:], age_prc95[i,:], alpha=0.3,color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(sample_sizes)
    ax.legend()
    fig.savefig(os.path.join(outpath, "age.png"), bbox_inches="tight")
    remove_all_text(fig)
    fig.savefig(os.path.join(outpath, "age_notxt.png"), bbox_inches="tight")
    plt.close(fig)

    # plot for sex
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(sex_acc.shape[0]):
        ax.plot(x, sex_mean[i,:], label=train_type[i], color=colors[i],linewidth=2,marker='o',markersize=12)
        ax.fill_between(x, sex_prc5[i,:], sex_prc95[i,:], alpha=0.3,color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(sample_sizes)
    ax.legend()
    fig.savefig(os.path.join(outpath, "sex.png"), bbox_inches="tight")
    remove_all_text(fig)
    fig.savefig(os.path.join(outpath, "sex_notxt.png"), bbox_inches="tight")
    plt.close(fig)


    # do not plot linear augmentation for ohbm abstract
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(sex_acc.shape[0]-1):
        ax.plot(x, sex_mean[i,:], label=train_type[i], color=colors[i],linewidth=2,marker='o',markersize=12)
        ax.fill_between(x, sex_prc5[i,:], sex_prc95[i,:], alpha=0.3,color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(sample_sizes)
    ax.legend()
    fig.savefig(os.path.join(outpath, "sex_ohbbm.png"), bbox_inches="tight")
    remove_all_text(fig)
    fig.savefig(os.path.join(outpath, "sex_ohbm_notxt.png"), bbox_inches="tight")
    plt.close(fig)

    # plot age improvement with matched number of generated data - mae
    # cmap = plt.colormaps["Pastel1"]
    # colors = [cmap(i/2) for i in range(2)]
    age_improvements = age_mean[0,:-1] - age_mean[1,1:]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x[:-1], age_improvements, color='#81D8D0')
    ax.set_xticks(x[:-1])
    ax.set_xticklabels(sample_sizes[:-1])
    fig.savefig(os.path.join(outpath, "age_improvement.png"), bbox_inches="tight")
    remove_all_text(fig)
    fig.savefig(os.path.join(outpath, "age_improvement_notxt.png"), bbox_inches="tight")
    plt.close(fig)

    # plot sex improvement with matched number of generated data - acc
    sex_improvements = sex_mean[1,1:] - sex_mean[0,:-1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x[:-1], sex_improvements, color='#81D8D0')
    ax.set_xticks(x[:-1])
    ax.set_xticklabels(sample_sizes[:-1])
    fig.savefig(os.path.join(outpath, "sex_improvement.png"), bbox_inches="tight")
    remove_all_text(fig)
    fig.savefig(os.path.join(outpath, "sex_improvement_notxt.png"), bbox_inches="tight")
    plt.close(fig)

    # plot both mix improvement
    age_improvements = age_mean[0,:-1] - age_mean[[1,-1],1:]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x[:-1] - 0.2, age_improvements[0,:], width=0.4, color=colors[1],label=train_type[1])
    ax.bar(x[:-1] + 0.2, age_improvements[-1,:], width=0.4, color=colors[-1],label=train_type[-1])
    ax.set_xticks(x[:-1])
    ax.set_xticklabels(sample_sizes[:-1])
    ax.legend()
    fig.savefig(os.path.join(outpath, "age_improvement_both.png"), bbox_inches="tight")
    remove_all_text(fig)
    fig.savefig(os.path.join(outpath, "age_improvement_both_notxt.png"), bbox_inches="tight")
    plt.close(fig)

    # plot sex improvement with matched number of generated data - acc
    sex_improvements = sex_mean[[1,-1],1:] - sex_mean[0,:-1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x[:-1] - 0.2, sex_improvements[0,:], width=0.4, color=colors[1],label=train_type[1])
    ax.bar(x[:-1] + 0.2, sex_improvements[-1,:], width=0.4, color=colors[-1], label=train_type[-1])
    ax.set_xticks(x[:-1])
    ax.set_xticklabels(sample_sizes[:-1])
    ax.legend()
    fig.savefig(os.path.join(outpath, "sex_improvement_both.png"), bbox_inches="tight")
    remove_all_text(fig)
    fig.savefig(os.path.join(outpath, "sex_improvement_both_notxt.png"), bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="parser receives test index")
    # parser.add_argument('-i', type=int,required=True,dest='test_idx')
    # args = parser.parse_args()
    # train_model(args.test_idx)
    
    # collected_f = collect_results()

    folder_path = "/data/gpfs/projects/punim1278/projects/vaegm/output/ml_with_gen"
    collected_f = os.path.join(folder_path, 'train_ml_with_generation_collected.npz')
    make_plots(collected_f)
