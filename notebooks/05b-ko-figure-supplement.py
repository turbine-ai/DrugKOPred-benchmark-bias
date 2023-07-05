
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import os
import numpy as np

EX_SPLITS = ["RND", "CEX", "GEX", "AEX"] 

EVALS_DIR = "/dbfs/mnt/sandbox/benchmark-paper/final/evals/ko"
BD_DIR = "/dbfs/mnt/sandbox/benchmark-paper/final/bds/ko"

def load_data():
    global_data  = pd.DataFrame()
    node_data = pd.DataFrame()
    cell_data = pd.DataFrame()
    for filename in os.listdir(EVALS_DIR):
        # drop .csv, split along _
        fn_tokens = filename[:-4].split("_")
        # split exclusivity
        ex_split = fn_tokens[1]
        # random resampling index
        random_split = int(fn_tokens[2])
        # linear, random forest or MLP model
        model = fn_tokens[3]
        features = fn_tokens[4] + '_' + fn_tokens[5]
        if ex_split not in EX_SPLITS:
            continue
        print(filename)
        df = pd.read_csv(os.path.join(EVALS_DIR, filename))
        df['ex_split'] = ex_split
        df['random_split'] = random_split
        df['model'] = model
        df['features'] = features
        node_data = node_data.append(df[df.type == 'perturbation'])
        cell_data = cell_data.append(df[df.type == 'cell'])

        # we append node and cell data before to avoid adding the bdt col
        # read bias detector results
        bdt_file = os.path.join(BD_DIR, f"geneeffect_{ex_split}_{random_split}_{model}_{dataset}.csv")
        bdt_df = pd.read_csv(bdt_file)

        bdt_pert = bdt_df[bdt_df.type == 'pert']
        n_sig = len(np.where((bdt_pert.p < 0.05) & (bdt_pert.r > 0))[0])
        df['bdt_n_sig_gene'] = n_sig
        df['bdt_ratio_sig_gene'] = n_sig/len(bdt_pert)

        bdt_cell = bdt_df[bdt_df.type == 'cell']
        n_sig = len(np.where((bdt_cell.p < 0.05) & (bdt_cell.r > 0))[0])
        df['bdt_n_sig_cell'] = n_sig
        df['bdt_ratio_sig_cell'] = n_sig/len(bdt_cell)
        global_data = global_data.append(df[df.type == 'global'])

    return global_data, node_data, cell_data

def rename_dataset(orig_name):
    name = orig_name.replace("onehot", 'OHE')
    name = name.replace("allfeatures", 'ALL')
    #order is important, mut is in mutation
    name = name.replace("mutation", 'MUT')
    name = name.replace("mut", 'MUT')
    name = name.replace("expression", 'TPM')
    name = name.replace("tpm", 'TPM')
    name = name.replace("go", 'GO')
    name = name.replace("n2v", 'N2V')
    name = name.replace("p2v", 'P2V')
    return name
    

global_data, node_data, cell_data = load_data()

# shorter model names
global_data.model.replace("randomforest", "RF", inplace=True)
global_data.model.replace("linear", "LR", inplace=True)
node_data.model.replace("randomforest", "RF", inplace=True)
node_data.model.replace("linear", "LR", inplace=True)
cell_data.model.replace("randomforest", "RF", inplace=True)
cell_data.model.replace("linear", "LR", inplace=True)

# needed for non-alphabetical sorting
global_data.model = pd.Categorical(global_data.model, categories=["LR", "RF", "MLP"], ordered=True)
node_data.model = pd.Categorical(node_data.model, categories=["LR", "RF", "MLP"], ordered=True)
cell_data.model = pd.Categorical(cell_data.model, categories=["LR", "RF", "MLP"], ordered=True)

# short featureset names
global_data['features'] = global_data.features.apply(rename_dataset)
node_data['features'] = node_data.features.apply(rename_dataset)
cell_data['features'] = cell_data.features.apply(rename_dataset)

# fill NaN pearson coming from 0 variance predictions
node_data.pearson.fillna(0, inplace=True)
cell_data.pearson.fillna(0, inplace=True)

# sorting to get the same order of models and featuresets in all plots
global_data = global_data.sort_values(['model', 'dataset'])
node_data = node_data.sort_values(['model', 'dataset'])
cell_data = cell_data.sort_values(['model', 'dataset'])

fig, ax = plt.subplots(4,5, figsize=(20,20))
barplot_kwargs = {
    "linewidth": 1,
    "edgecolor": "black"
}
sns.set_palette(sns.color_palette('tab20'))
for i, ex in enumerate(EX_SPLITS):
    g = sns.barplot(x='model', y='pearson', hue='features', data=global_data[global_data.ex_split == ex], ax=ax[i, 0], **barplot_kwargs)
    g.get_legend().remove()
    ax[i, 0].set_xlabel('')
    ax[i, 0].set_ylabel('Global Pearson')
    ax[i, 0].set_ylim(0,1)

    g = sns.boxplot(x='model', y='pearson', hue='features', data=cell_data[cell_data.ex_split == ex], ax=ax[i, 1])
    g.get_legend().remove()
    ax[i, 1].set_xlabel('')
    ax[i, 1].set_ylabel('Per cell line pearson')
    ax[i, 1].set_title(ex)
    ax[i, 1].set_ylim(-1,1)

    g = sns.barplot(x='model', y='bdt_ratio_sig_cell', hue='features', data=global_data[global_data.ex_split == ex], ax=ax[i, 2], **barplot_kwargs)
    g.get_legend().remove()
    ax[i, 2].set_ylabel('Ratio of significant cells')
    ax[i, 2].set_ylim(0,1)
    if i != 2:
        ax[i, 2].set_xlabel('')

    g = sns.boxplot(x='model', y='pearson', hue='features', data=node_data[node_data.ex_split == ex], ax=ax[i, 3])
    g.get_legend().remove()
    ax[i, 3].set_xlabel('')
    ax[i, 3].set_ylabel('Per gene pearson')
    ax[i, 3].set_title(ex)
    ax[i, 3].set_ylim(-1,1)

    g = sns.barplot(x='model', y='bdt_ratio_sig_gene', hue='features', data=global_data[global_data.ex_split == ex], ax=ax[i, 4], **barplot_kwargs)
    g.get_legend().remove()
    ax[i, 4].set_xlabel('')
    ax[i, 4].set_ylabel('Ratio of significant genes')
    ax[i, 4].set_ylim(0,1)
ax[1, 4].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
fig.savefig("ko_perf_figure_supp.png")

