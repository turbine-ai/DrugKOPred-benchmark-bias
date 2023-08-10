
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import os
import numpy as np

from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols

EX_SPLITS = ["RND", "CEX", "GEX", "AEX"] 
THRSH = 0.05

EVALS_DIR = "/home/centi/benchmark-paper-2023/data/final/evals/ko"
BD_DIR = "/home/centi/benchmark-paper-2023/data/final/bds/ko"

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
        #import pdb; pdb.set_trace()

        bdt_file = os.path.join(BD_DIR, f"geneeffect_{ex_split}_{random_split}_{model}_{features}.csv")
        bdt_df = pd.read_csv(bdt_file)
        df = pd.merge(
                df,
                bdt_df[["name", "r", "p"]],
                how='left',
                on="name"
        )

        node_data = node_data.append(df[df.type == 'perturbation'])
        cell_data = cell_data.append(df[df.type == 'cell'])

        # we append node and cell data before to avoid adding the bdt col
        # read bias detector results
        bdt_file = os.path.join(BD_DIR, f"geneeffect_{ex_split}_{random_split}_{model}_{features}.csv")
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
        
        cell_data.to_csv("cell_data.csv")

    return global_data, node_data, cell_data

def get_significance_labels(df, col_id, split, _thrsh=THRSH):
  filt = df['ex_split'] == split
  models = set(df['model'])
  feature_combinations = set(df['features'])
  model = ols(f"""{col_id} ~ C(model) + C(features) +
                C(model):C(features)""", data=df[filt]).fit()

  _anova = sm.stats.anova_lm(model, typ=2)
  p_values=[]
  filt2 = np.logical_and(df['model'] == 'LR', df['features'] == 'ALL_ALL')
  reference = df.loc[np.logical_and(filt, filt2), col_id]
  for model in models:
    for fc in feature_combinations:
      filt2 = np.logical_and(df['model'] == model, df['features'] == fc)
      test = df.loc[np.logical_and(filt, filt2),col_id]
      res = ttest_rel(reference, test)
      p_values.append({'vs':'LR + ALL + ALL',
                      'model':model,
                      'features':fc,
                      'p-value':res.pvalue})
 
  p_values_df = pd.DataFrame(p_values)
  p_values_df = p_values_df.fillna(1)
  _,p_values_df['adjusted_p-value'],_,_ = multipletests(p_values_df['p-value'], alpha = _thrsh, method = 'fdr_bh')
  p_values_df['label'] = ['+' if x < _thrsh and _anova.loc['C(model)','PR(>F)'] <= _thrsh else ' ' for x in p_values_df['adjusted_p-value']]

  p_values=[]

  for model in models:
    filt2 = np.logical_and(df['model'] == model, df['features'] == 'OHE_OHE')
    reference = df.loc[np.logical_and(filt, filt2),col_id]
    for fc in feature_combinations:
      filt2 = np.logical_and(df['model'] == model, df['features'] == fc)
      test = df.loc[np.logical_and(filt, filt2),col_id]
      res = ttest_rel(reference, test)
      p_values.append({'vs':f'{model} + OHE + OHE',
                      'model':model,
                      'features':fc,
                      'p-value':res.pvalue})
  p_values = pd.DataFrame(p_values)
  p_values = p_values.fillna(1)
  _,p_values['adjusted_p-value'],_,_ = multipletests(p_values['p-value'], alpha = _thrsh, method = 'fdr_bh')
  p_values['label'] = ['*' if x < _thrsh and _anova.loc['C(model):C(features)','PR(>F)']< _thrsh else ' ' for x in p_values['adjusted_p-value']]

  labels = p_values_df[['model','features','label', 'p-value', 'adjusted_p-value']]\
           .merge(
                p_values[['model','features','label', 'p-value', 'adjusted_p-value']],
                how='left',
                on=['model','features'],
                suffixes = ['_model','_fc']
            )
  labels['label'] = labels['label_model'] + '\n' + labels['label_fc']
  
  return labels

def add_sign_labels(ax, sign_labels, rep, max_y):
    start = -0.3
    end = 0.5
    model_order = ["LR", "RF", "MLP"]
    feature_order = list(sorted(sign_labels.features.unique()))
    step = (end-start)/len(feature_order)
    #rep = len(set(corr_df['model']))
    pos = np.arange(start,end,step)
    scal = 1.0
    for r in range(1,rep):
        pos = np.concatenate((pos, np.arange(r+start,r+end,step)))
    pos = pos * scal
    tick = 0
    for _o in model_order:
        for _fo in feature_order:
          _label = sign_labels.loc[np.logical_and(sign_labels['model'] == _o, sign_labels['features'] == _fo),'label'].tolist()[0]
        
          ax.text(pos[tick], max_y*0.9, _label, ha = 'center', weight='bold', color = 'red')
          tick +=1

def merge_p_values(global_data, sign_labels, ex_split, score_name):
    for ix, row in global_data.iterrows():
        if row.ex_split == ex_split:
            filt = (sign_labels.model == row.model) & (sign_labels.features == row.features)
            sign_data = sign_labels[filt].iloc[0] 
            global_data.loc[ix, score_name + "_p_value_model"] = sign_data["p-value_model"]
            global_data.loc[ix, score_name + "_p_value_fc"] = sign_data["p-value_fc"]
            global_data.loc[ix, score_name + "_adj_p_value_model"] = sign_data["adjusted_p-value_model"]
            global_data.loc[ix, score_name + "_adj_p_value_fc"] = sign_data["adjusted_p-value_fc"]

def rename_features(orig_name):
    name = name.replace("mut", 'mutation')
    name = name.replace("_", ' + ')
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
global_data['features'] = global_data.features.apply(rename_features)
node_data['features'] = node_data.features.apply(rename_features)
cell_data['features'] = cell_data.features.apply(rename_features)

# fill NaN pearson coming from 0 variance predictions
node_data.pearson.fillna(0, inplace=True)
cell_data.pearson.fillna(0, inplace=True)

# sorting to get the same order of models and featuresets in all plots
global_data = global_data.sort_values(['model', 'features']).reset_index()
node_data = node_data.sort_values(['model', 'features']).reset_index()
cell_data = cell_data.sort_values(['model', 'features']).reset_index()

fig, ax = plt.subplots(7,4, figsize=(20,28))
barplot_kwargs = {
    "linewidth": 1,
    "edgecolor": "black"
}
sns.set_palette(sns.color_palette('tab20'))

for i, ex in enumerate(EX_SPLITS):
    g = sns.barplot(x='model', y='pearson', hue='features', data=global_data[global_data.ex_split == ex], ax=ax[0, i], **barplot_kwargs)
    g.get_legend().remove()
    ax[0, i].set_xlabel('')
    ax[0, i].set_ylabel('Global Pearson')
    ax[0, i].set_ylim(0,1)
    ax[0, i].set_title(ex)
    sign_labels = get_significance_labels(global_data, 'pearson', ex)
    print(ex, "global pearson")
    print(sign_labels)
    #add_sign_labels(ax[0,i], sign_labels, len(global_data.model.unique()), 1)
    merge_p_values(global_data, sign_labels, ex, 'global_pearson')

    g = sns.boxplot(x='model', y='pearson', hue='features', data=cell_data[cell_data.ex_split == ex], ax=ax[1, i])
    g.get_legend().remove()
    ax[1, i].set_xlabel('')
    ax[1, i].set_ylabel('Per cell line pearson')
    ax[1, i].set_title(ex)
    ax[1, i].set_ylim(-1,1)
    ax[1, i].set_title(ex)
    sign_labels = get_significance_labels(cell_data, 'pearson', ex)
    print(ex, "cell pearson")
    print(sign_labels)
    if i < 2:
        max_y = 0
    else:
        max_y = 0.9
    #add_sign_labels(ax[1,i], sign_labels, len(cell_data.model.unique()), max_y)
    merge_p_values(global_data, sign_labels, ex, 'cell_pearson')

    g = sns.boxplot(x='model', y='r', hue='features', data=cell_data[cell_data.ex_split == ex], ax=ax[2, i])
    g.get_legend().remove()
    ax[2, i].set_ylabel('Per cell partial correlation')
    ax[2, i].set_title(ex)
    ax[2, i].set_ylim(-1,1)
    ax[2, i].set_xlabel('method')
    sign_labels = get_significance_labels(node_data, 'r', ex)
    print(ex, "pcorr")
    print(sign_labels)
    #add_sign_labels(ax[2,i], sign_labels, len(cell_data.model.unique()), 0.9)
    merge_p_values(global_data, sign_labels, ex, 'cell_partial_corr')

    g = sns.barplot(x='model', y='bdt_ratio_sig_cell', hue='features', data=global_data[global_data.ex_split == ex], ax=ax[3, i], **barplot_kwargs)
    g.get_legend().remove()
    ax[3, i].set_ylabel('Ratio of significant cells')
    ax[3, i].set_ylim(0,1)
    if i != 3:
        ax[3, i].set_xlabel('')
    ax[3, i].set_title(ex)
    sign_labels = get_significance_labels(global_data, 'bdt_ratio_sig_cell', ex)
    print(ex, "cell bdt")
    print(sign_labels)
    #add_sign_labels(ax[3,i], sign_labels, len(cell_data.model.unique()), 1)
    merge_p_values(global_data, sign_labels, ex, 'cell_bd_ratio')

    g = sns.boxplot(x='model', y='pearson', hue='features', data=node_data[node_data.ex_split == ex], ax=ax[4, i])
    g.get_legend().remove()
    ax[4, i].set_xlabel('')
    ax[4, i].set_ylabel('Per gene pearson')
    ax[4, i].set_title(ex)
    ax[4, i].set_ylim(-1,1)
    ax[4, i].set_title(ex)
    sign_labels = get_significance_labels(node_data, 'pearson', ex)
    print(ex, "node pearson")
    print(sign_labels)
    #add_sign_labels(ax[4,i], sign_labels, len(node_data.model.unique()), 0.9)
    merge_p_values(global_data, sign_labels, ex, 'node_pearson')

    g = sns.boxplot(x='model', y='r', hue='features', data=node_data[node_data.ex_split == ex], ax=ax[5, i])
    g.get_legend().remove()
    ax[5, i].set_ylabel('Per gene partial correlation')
    ax[5, i].set_title(ex)
    ax[5, i].set_ylim(-1,1)
    ax[5, i].set_xlabel('method')
    sign_labels = get_significance_labels(node_data, 'r', ex)
    print(ex, "pcorr")
    print(sign_labels)
    #add_sign_labels(ax[5,i], sign_labels, len(node_data.model.unique()), 0.9)
    merge_p_values(global_data, sign_labels, ex, 'node_partial_corr')

    g = sns.barplot(x='model', y='bdt_ratio_sig_gene', hue='features', data=global_data[global_data.ex_split == ex], ax=ax[6, i], **barplot_kwargs)
    g.get_legend().remove()
    ax[6, i].set_xlabel('')
    ax[6, i].set_ylabel('Ratio of significant genes')
    ax[6, i].set_ylim(0,1)
    ax[6, i].set_title(ex)
    sign_labels = get_significance_labels(global_data, 'bdt_ratio_sig_gene', ex)
    print(ex, "node bdt")
    print(sign_labels)
    #add_sign_labels(ax[6,i], sign_labels, len(node_data.model.unique()), 1)
    merge_p_values(global_data, sign_labels, ex, 'node_bd_ratio')

ax[6, 2].legend(loc='center left', bbox_to_anchor=(0.5, -0.7), ncol=2)
fig.tight_layout()
fig.savefig("ko_perf_figure_supp.png")
fig.savefig("ko_perf_figure_supp.pdf")

#supplementary table stuff
cell_groups = cell_data.groupby(["ex_split", "random_split", "model", "features"])
node_groups = node_data.groupby(["ex_split", "random_split", "model", "features"])
cell_means = cell_groups.mean()["pearson"]
cell_sd = cell_groups.std()["pearson"]
node_means = node_groups.mean()["pearson"]
node_sd = node_groups.std()["pearson"]
pearson_dist_data = pd.DataFrame({
    "per_cell_pearson_mean": cell_means,
    "per_cell_pearson_sd": cell_sd,
    "per_node_pearson_mean": node_means,
    "per_node_pearson_sd": node_sd
    }).reset_index()
supp_table = pd.merge(global_data, pearson_dist_data, on=["ex_split", "random_split", "model", "features"])
supp_table = supp_table[['model', 'features', 'ex_split', 'random_split', 'n',
       'spearman', 'rmse',    
       'pearson', 'global_pearson_p_value_model',
       'global_pearson_p_value_fc', 'global_pearson_adj_p_value_model',
       'global_pearson_adj_p_value_fc',

       'per_node_pearson_mean', 'per_node_pearson_sd',
       'node_pearson_p_value_model', 'node_pearson_p_value_fc',
       'node_pearson_adj_p_value_model', 'node_pearson_adj_p_value_fc',

       'bdt_n_sig_gene', 'bdt_ratio_sig_gene', 'node_bd_ratio_p_value_model',
       'node_bd_ratio_p_value_fc', 'node_bd_ratio_adj_p_value_model',
       'node_bd_ratio_adj_p_value_fc',

       'per_cell_pearson_mean', 'per_cell_pearson_sd',
       'cell_pearson_p_value_model', 'cell_pearson_p_value_fc',
       'cell_pearson_adj_p_value_model', 'cell_pearson_adj_p_value_fc',

       'bdt_n_sig_cell', 'bdt_ratio_sig_cell', 'cell_bd_ratio_p_value_model',
       'cell_bd_ratio_p_value_fc', 'cell_bd_ratio_adj_p_value_model',
       'cell_bd_ratio_adj_p_value_fc']]
supp_table.to_csv("ko_supplementary_table.csv")
