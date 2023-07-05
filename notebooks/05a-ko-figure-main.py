import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import os
import numpy as np

EX_SPLITS = ["RND", "CEX", "GEX"] 
DATASET_FILTER = ['onehot_onehot', 'allfeatures_onehot', 'onehot_allfeatures', 'allfeatures_allfeatures']

EVALS_DIR = "/dbfs/mnt/sandbox/benchmark-paper/final/evals/ko"
BD_DIR = "/dbfs/mnt/sandbox/benchmark-paper/final/bds/ko"


def load_data():
    global_data  = pd.DataFrame()
    node_data = pd.DataFrame()
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
        if dataset not in DATASET_FILTER:
            continue
        print(filename)
        df = pd.read_csv(os.path.join(EVALS_DIR, filename))
        df['ex_split'] = ex_split
        df['random_split'] = random_split
        df['model'] = model
        df['features'] = features
        node_data = node_data.append(df[df.type == 'perturbation'])

        #we append node data before to avoid adding the bdt col
        #read in bias detector results
        bdt_file = os.path.join(BD_DIR, f"geneeffect_{ex_split}_{random_split}_{model}_{dataset}.csv")
        bdt_df = pd.read_csv(bdt_file)
        bdt_df = bdt_df[bdt_df.type == 'pert']
        n_sig = len(np.where((bdt_df.p < 0.05) & (bdt_df.r > 0))[0])
        df['bdt_n_sig'] = n_sig
        df['bdt_ratio_sig'] = n_sig/len(bdt_df)
        global_data = global_data.append(df[df.type == 'global'])

    return global_data, node_data


global_data, node_data = load_data()

# shorter model names
global_data.model.replace("randomforest", "RF", inplace=True)
global_data.model.replace("linear", "LR", inplace=True)
node_data.model.replace("randomforest", "RF", inplace=True)
node_data.model.replace("linear", "LR", inplace=True)

# needed for not alphabetical sorting
global_data.model = pd.Categorical(global_data.model, categories=["LR", "RF", "MLP"], ordered=True)
node_data.model = pd.Categorical(node_data.model, categories=["LR", "RF", "MLP"], ordered=True)

# short featureset names
global_data.features.replace("allfeatures_allfeatures", "ALL-ALL", inplace=True)
global_data.features.replace("allfeatures_onehot", "ALL-OHE", inplace=True)
global_data.features.replace("onehot_allfeatures", "OHE-ALL", inplace=True)
global_data.features.replace("onehot_onehot", "OHE-OHE", inplace=True)
node_data.features.replace("allfeatures_allfeatures", "ALL-ALL", inplace=True)
node_data.features.replace("allfeatures_onehot", "ALL-OHE", inplace=True)
node_data.features.replace("onehot_allfeatures", "OHE-ALL", inplace=True)
node_data.features.replace("onehot_onehot", "OHE-OHE", inplace=True)

# fill NaN pearson coming from 0 variance predictions
node_data.pearson.fillna(0, inplace=True)

# sorting to get the same order of models and featuresets in all plots
global_data = global_data.sort_values(['model', 'features'])
node_data = node_data.sort_values(['model', 'features'])

fig, ax = plt.subplots(3,3, figsize=(9,10))
colors = ['#78577A', '#33C7CC', '#FF6666', '#FCFAFF']
sns.set_palette(sns.color_palette(colors))
sns.palplot(sns.color_palette(colors))
barplot_kwargs = {
    "linewidth": 1,
    "edgecolor": "black"
}
for i, ex in enumerate(EX_SPLITS):
    g = sns.barplot(x='model', y='pearson', hue='features', data=global_data[global_data.ex_split == ex], ax=ax[i, 0], **barplot_kwargs)
    g.get_legend().remove()
    ax[i, 0].set_xlabel('')
    ax[i, 0].set_ylabel('Global Pearson')
    ax[i, 0].set_ylim(0,1)
    g = sns.boxplot(x='model', y='pearson', hue='features', data=node_data[node_data.ex_split == ex], ax=ax[i, 1])
    g.get_legend().remove()
    ax[i, 1].set_ylabel('Per gene pearson')
    ax[i, 1].set_title(ex)
    ax[i, 1].set_ylim(-1,1)
    if i != 2:
        ax[i, 1].set_xlabel('')
    g = sns.barplot(x='model', y='bdt_ratio_sig', hue='features', data=global_data[global_data.ex_split == ex], ax=ax[i, 2], **barplot_kwargs)
    g.get_legend().remove()
    ax[i, 2].set_xlabel('')
    ax[i, 2].set_ylabel('Ratio of significant genes')
    ax[i, 2].set_ylim(0,1)
ax[1, 2].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
fig.savefig("ko_figure_main.png")

