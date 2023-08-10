import random
import os
import string

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols

EX_SPLITS = ["RND", "CEX", "GEX"] 
FEATURESET_FILTER = ['onehot_onehot', 'allfeatures_onehot', 'onehot_allfeatures', 'allfeatures_allfeatures']
THRSH = 0.05

EVALS_DIR = "/home/centi/benchmark-paper-2023/data/final/evals/ko"
BD_DIR = "/home/centi/benchmark-paper-2023/data/final/bds/ko"


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
        if features not in FEATURESET_FILTER:
            continue
        print(filename)
        df = pd.read_csv(os.path.join(EVALS_DIR, filename))
        df['ex_split'] = ex_split
        df['random_split'] = random_split
        df['model'] = model
        df['features'] = features

        bdt_file = os.path.join(BD_DIR, f"geneeffect_{ex_split}_{random_split}_{model}_{features}.csv")
        bdt_df = pd.read_csv(bdt_file)
        bdt_df = bdt_df[bdt_df.type == 'pert']
        df = pd.merge(
                df,
                bdt_df[["name", "r", "p"]],
                how='left',
                on="name"
        )

        node_data = node_data.append(df[df.type == 'perturbation'])

        #we append node data before to avoid adding the bdt_n_sig col
        #read in bias detector results
        n_sig = len(np.where((bdt_df.p < 0.05) & (bdt_df.r > 0))[0])
        df['bdt_n_sig'] = n_sig
        df['bdt_ratio_sig'] = n_sig/len(bdt_df)
        global_data = global_data.append(df[df.type == 'global'])

    return global_data, node_data

def get_significance_labels(df, col_id, split, _thrsh=THRSH):
  filt = df['ex_split'] == split
  models = set(df['model'])
  feature_combinations = set(df['features'])
  model = ols(f"""{col_id} ~ C(model) + C(features) +
                C(model):C(features)""", data=df[filt]).fit()

  _anova = sm.stats.anova_lm(model, typ=2)
  p_values=[]
  filt2 = np.logical_and(df['model'] == 'LR', df['features'] == 'allfeatures + allfeatures')
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
    filt2 = np.logical_and(df['model'] == model, df['features'] == 'onehot + onehot')
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

  labels = p_values_df[['model','features','label']].merge(p_values[['model','features','label']], how = 'left', on=['model','features'], suffixes = ['_model','_fc'])
  labels['label'] = labels['label_model'] + '\n' + labels['label_fc']
  
  return labels[['model','features','label']]

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


global_data, node_data = load_data()

# shorter model names
global_data.model.replace("randomforest", "RF", inplace=True)
global_data.model.replace("linear", "LR", inplace=True)
node_data.model.replace("randomforest", "RF", inplace=True)
node_data.model.replace("linear", "LR", inplace=True)

# needed for non-alphabetical sorting
global_data.model = pd.Categorical(global_data.model, categories=["LR", "RF", "MLP"], ordered=True)
node_data.model = pd.Categorical(node_data.model, categories=["LR", "RF", "MLP"], ordered=True)

# short featureset names
global_data.features.replace("allfeatures_allfeatures", "allfeatures + allfeatures", inplace=True)
global_data.features.replace("allfeatures_onehot", "allfeatures + onehot", inplace=True)
global_data.features.replace("onehot_allfeatures", "onehot + allfeatures", inplace=True)
global_data.features.replace("onehot_onehot", "onehot + onehot", inplace=True)
node_data.features.replace("allfeatures_allfeatures", "allfeatures + allfeatures", inplace=True)
node_data.features.replace("allfeatures_onehot", "allfeatures + onehot", inplace=True)
node_data.features.replace("onehot_allfeatures", "onehot + allfeatures", inplace=True)
node_data.features.replace("onehot_onehot", "onehot + onehot", inplace=True)

# fill NaN pearson coming from 0 variance predictions
node_data.pearson.fillna(0, inplace=True)

# sorting to get the same order of models and featuresets in all plots
global_data = global_data.sort_values(['model', 'features'])
node_data = node_data.sort_values(['model', 'features'])

fig, ax = plt.subplots(4,3, figsize=(9,13))
colors = ['#78577A', '#33C7CC', '#FF6666', '#FCFAFF']
sns.set_palette(sns.color_palette(colors))
barplot_kwargs = {
    "linewidth": 1,
    "edgecolor": "black"
}
for i, ex in enumerate(EX_SPLITS):
    g = sns.barplot(x='model', y='pearson', hue='features', data=global_data[global_data.ex_split == ex], ax=ax[0, i], **barplot_kwargs)
    g.get_legend().remove()
    ax[0, i].set_xlabel('')
    ax[0, i].set_ylabel('Global Pearson')
    ax[0, i].set_title(ex)
    ax[0, i].set_ylim(0,1)
    sign_labels = get_significance_labels(global_data, 'pearson', ex)
    print(ex, "global pearson")
    print(sign_labels)
    add_sign_labels(ax[0,i], sign_labels, len(global_data.model.unique()), 1)

    g = sns.boxplot(x='model', y='pearson', hue='features', data=node_data[node_data.ex_split == ex], ax=ax[1, i])
    g.get_legend().remove()
    ax[1, i].set_ylabel('Per gene pearson')
    ax[1, i].set_title(ex)
    ax[1, i].set_ylim(-1,1)
    ax[1, i].set_xlabel('method')
    sign_labels = get_significance_labels(node_data, 'pearson', ex)
    print(ex, "node pearson")
    print(sign_labels)
    add_sign_labels(ax[1,i], sign_labels, len(node_data.model.unique()), 0.9)

    g = sns.boxplot(x='model', y='r', hue='features', data=node_data[node_data.ex_split == ex], ax=ax[2, i])
    g.get_legend().remove()
    ax[2, i].set_ylabel('Per gene partial correlation')
    ax[2, i].set_title(ex)
    ax[2, i].set_ylim(-1,1)
    ax[2, i].set_xlabel('method')
    sign_labels = get_significance_labels(node_data, 'r', ex)
    print(ex, "pcorr")
    print(sign_labels)
    add_sign_labels(ax[2,i], sign_labels, len(node_data.model.unique()), 0.9)

    g = sns.barplot(x='model', y='bdt_ratio_sig', hue='features', data=global_data[global_data.ex_split == ex], ax=ax[3, i], **barplot_kwargs)
    g.get_legend().remove()
    ax[3, i].set_xlabel('')
    ax[3, i].set_ylabel('Ratio of significant genes')
    ax[3, i].set_title(ex)
    ax[3, i].set_ylim(0,1)
    sign_labels = get_significance_labels(global_data, 'bdt_ratio_sig', ex)
    print(ex, "bdt")
    print(sign_labels)
    add_sign_labels(ax[3,i], sign_labels, len(global_data.model.unique()), 1)


for i in range(4):
    for j in range(3):
        ax[i,j].text(-0.1, 1.05, string.ascii_uppercase[i*3+j],
                transform=ax[i,j].transAxes, size=20, weight='bold')

ax[3, 1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.7))
fig.tight_layout()
fig.savefig("ko_figure_main.png")
fig.savefig("ko_figure_main.pdf")

