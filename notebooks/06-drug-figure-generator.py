# Databricks notebook source
import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

import json
import os
import string


from scipy.stats import pearsonr, spearmanr, ttest_rel, kruskal, mannwhitneyu
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_recall_curve, f1_score, auc
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols


# COMMAND ----------

# MAGIC %md
# MAGIC ## Functions

# COMMAND ----------

colors = ['#78577A', '#FF6666', '#33c7cc', '#FCFAFF']
sns.set_palette(sns.color_palette(colors))
THRSH = 0.05

def plotBias(tmp_df, _title = '', col_id = 'f', _order = None, labels = None):
  rows = len(interesting_types)
  cols = len(interesting_splits)
  
  f, axs = plt.subplots(rows, cols, figsize = (cols * 5.0, rows * 5.0), dpi=100)
  f.patch.set_facecolor('white')
  f.suptitle(_title)
  for i in range(rows):
    max_y = tmp_df.loc[tmp_df['type']== interesting_types[i],col_id].max()*1.05
    min_y = tmp_df.loc[tmp_df['type']== interesting_types[i],col_id].min()-max_y*0.05
    for j in range(cols):
      fil = np.logical_and(tmp_df['split'] == interesting_splits[j], tmp_df['type']== interesting_types[i])
      #print(tmp_df[fil].shape[0])
      g = sns.barplot(x='method', y=col_id, hue='feature_combination', data=tmp_df[fil], linewidth=1, edgecolor = '#000000', order=_order, ax=axs[i,j])
      if labels is not None: g.set(xticklabels=labels)
      if ((i==0) & (j==2)):
        axs[i,j].legend(bbox_to_anchor=(1, 1))
      else:
        axs[i,j].legend([])
      if i==0:
        axs[i,j].set_title(interesting_splits[j])
      if j == 0:
        axs[i,j].set(ylabel=f'Fraction of {interesting_types[i]}s over bias')
      else:
        axs[i,j].set(ylabel ='')
      
      axs[i,j].set(ylim=(min_y,max_y))
  plt.tight_layout()
  return plt

def plotCorr(tmp_df, _title = '', col_id ='pearson', _order = None, labels = None):
  rows = len(interesting_types)
  cols = len(interesting_splits)
  
  f, axs = plt.subplots(rows, cols, figsize = (cols * 5.0, rows * 5.0), dpi=100)
  f.patch.set_facecolor('white')
  f.suptitle(_title)
  
  for i in range(rows):
    max_y = tmp_df.loc[tmp_df['type']== interesting_types[i],col_id].max()*1.05
    min_y = tmp_df.loc[tmp_df['type']== interesting_types[i],col_id].min()-max_y*0.05
    for j in range(cols):
      fil = (tmp_df['type'] == interesting_types[i]) & (tmp_df['split'] == interesting_splits[j])
      g = sns.boxplot(x='method', y=col_id, hue='feature_combination', data=tmp_df[fil], order = _order, ax=axs[i, j])
      if labels is not None: g.set(xticklabels=labels)
      if ((i==0) & (j==2)):
        axs[i, j].legend(bbox_to_anchor=(1, 1))
      else:
        axs[i,j].legend([])
      if i==0:
        axs[i, j].set_title(interesting_splits[j])
      if j == 0:
        axs[i,j].set(ylabel=f'Pearson correlation per {interesting_types[i]}')
      else:
        axs[i,j].set(ylabel ='')
      
      axs[i,j].set(ylim=(min_y, max_y))
  plt.tight_layout()
  return plt

def plotComposite(corr_df, bd_df, _title, corr_col_id = 'pearson', bd_col_id = 'f', _order = None, labels = None, draw_significance = True, stat_function = 'non-paramteric'):
  stat_functions = {'parametric':get_significance_labels,
                    'non-paramteric':get_significance_labels_np}
  def add_sign_labels(ax, sign_labels, rep):
    start = -0.3
    end = 0.5
    feature_order = interesting_combinations
    step = (end-start)/len(feature_order)
    rep = len(set(corr_df['method']))
    pos = np.arange(start,end,step)
    scal =1.0
    for r in range(1,rep):
      pos = np.concatenate((pos, np.arange(r+start,r+end,step)))
    pos = pos * scal
    tick = 0
    for _o in _order:
      for _fo in feature_order:
        _label = sign_labels.loc[np.logical_and(sign_labels['method'] == _o, sign_labels['feature_combination'] == _fo),'label'].tolist()[0]
        
        ax.text(pos[tick], max_y*0.9, _label, ha = 'center', weight='bold', color = 'red')
        tick +=1

  bd_param = 'Fraction'
  if bd_col_id != 'f': bd_param = 'Number'
  rows = len(interesting_types)
  cols = len(interesting_splits)
  
  if 'global' in interesting_types:
    f, axs = plt.subplots(2 * rows -1, cols, figsize = (cols * 5.0, 6.0 + (rows - 1) * 12.0+1.0), dpi=300)
  else:
    f, axs = plt.subplots(2 * rows, cols, figsize = (cols * 5.0, rows * 12.0+1.0), dpi=300)
  row_cnt = 0

  if len(set(corr_df['feature_combination'])) <=4: 
    sns.set_palette(sns.color_palette(colors))
  else:
    sns.set_palette('tab20')

  for i in range(rows):
    # plot correlations
    max_y = corr_df.loc[corr_df['type']== interesting_types[i],corr_col_id].max()*1.1
    min_y = corr_df.loc[corr_df['type']== interesting_types[i],corr_col_id].min()-max_y*0.05
    for j in range(cols):
      fil = (corr_df['type'] == interesting_types[i]) & (corr_df['split'] == interesting_splits[j])
      g = sns.boxplot(x='method', y=corr_col_id, hue='feature_combination', data=corr_df[fil], order = _order, ax=axs[row_cnt, j])
      
      if draw_significance and stat_function in stat_functions:
        sign_labels = stat_functions[stat_function](corr_df, corr_col_id, tpe = interesting_types[i], split = interesting_splits[j])
        add_sign_labels(axs[row_cnt, j], sign_labels, len(set(corr_df['method'])))
     
          
      
      if labels is not None: g.set(xticklabels=labels)
      if ((i==0) & (j==cols-1)):
        axs[row_cnt, j].legend(bbox_to_anchor=(1, 1))
      else:
        axs[row_cnt,j].legend([])
      if i==0:
        axs[row_cnt, j].set_title(interesting_splits[j])
      if j == 0:
        if 'global' in interesting_types[i]:
          axs[row_cnt,j].set(ylabel=f'Global {corr_col_id.title()} correlation')
        else:
          axs[row_cnt,j].set(ylabel=f'{corr_col_id.title()} correlation per {interesting_types[i]}')
      else:
        axs[row_cnt,j].set(ylabel ='')
      axs[row_cnt,j].set(ylim=(min_y, max_y))
      axs[row_cnt,j].text(-0.1, 1.05, string.ascii_uppercase[row_cnt*cols+j], transform=axs[row_cnt,j].transAxes, 
            size=20, weight='bold')
    row_cnt += 1
    # plot corresponding BD results
    if 'global' not in interesting_types[i]:
      max_y = bd_df.loc[bd_df['type']== interesting_types[i],bd_col_id].max()*1.05
      min_y = bd_df.loc[bd_df['type']== interesting_types[i],bd_col_id].min()-max_y*0.05
      for j in range(cols):
        fil = (bd_df['type'] == interesting_types[i]) & (bd_df['split'] == interesting_splits[j])
        g = sns.barplot(x='method', y=bd_col_id, hue='feature_combination', data=bd_df[fil], linewidth=1, edgecolor = '#000000', order=_order, ax=axs[row_cnt,j])
        if draw_significance and stat_function in stat_functions:
          sign_labels = stat_functions[stat_function](bd_df, bd_col_id, tpe = interesting_types[i], split = interesting_splits[j])
          add_sign_labels(axs[row_cnt, j], sign_labels, len(set(corr_df['method'])))

        if labels is not None: g.set(xticklabels=labels)
        axs[row_cnt,j].legend([])
        if j == 0:
          axs[row_cnt,j].set(ylabel=f'{bd_param} of {interesting_types[i]}s over bias')
        else:
          axs[row_cnt,j].set(ylabel ='')
        axs[row_cnt,j].set(ylim=(min_y, max_y))
        axs[row_cnt,j].text(-0.1, 1.05, string.ascii_uppercase[row_cnt*cols+j], transform=axs[row_cnt,j].transAxes, 
            size=20, weight='bold')
      row_cnt += 1

  f.patch.set_facecolor('white')
  f.suptitle(_title)
  plt.tight_layout()
  return plt

def get_significance_labels(df, col_id, tpe = 'perturbation', split = 'RND', _thrsh = THRSH):
  filt = np.logical_and(df['type'] == tpe, df['split'] == split)
  methods = set(df['method'])
  feature_combinations = set(df['feature_combination'])
  model = ols(f"""{col_id} ~ C(method) + C(feature_combination) +
                C(method):C(feature_combination)""", data=df[filt]).fit()

  _anova = sm.stats.anova_lm(model, typ=2)
  p_values=[]
  filt2 = np.logical_and(df['method'] == 'linear', df['feature_combination'] == 'allfeatures + allfeatures')
  reference = df.loc[np.logical_and(filt, filt2),col_id]
  for method in methods:
    for fc in feature_combinations:
      filt2 = np.logical_and(df['method'] == method, df['feature_combination'] == fc)
      test = df.loc[np.logical_and(filt, filt2),col_id]
      res = ttest_rel (reference, test)
      p_values.append({'vs':'LR + ALL + ALL',
                      'method':method,
                      'feature_combination':fc,
                      'p-value':res.pvalue})
 
  p_values_df = pd.DataFrame(p_values)
  p_values_df = p_values_df.fillna(1)
  _,p_values_df['adjusted_p-value'],_,_ = multipletests(p_values_df['p-value'], alpha = _thrsh, method = 'fdr_bh')
  p_values_df['label'] = ['+' if x < _thrsh and _anova.loc['C(method)','PR(>F)'] <= _thrsh else '' for x in p_values_df['adjusted_p-value']]

  p_values=[]

  for method in methods:
    filt2 = np.logical_and(df['method'] == method, df['feature_combination'] == 'onehot + onehot')
    reference = df.loc[np.logical_and(filt, filt2),col_id]
    for fc in feature_combinations:
      filt2 = np.logical_and(df['method'] == method, df['feature_combination'] == fc)
      test = df.loc[np.logical_and(filt, filt2),col_id]
      res = ttest_rel (reference, test)
      p_values.append({'vs':f'{method}+ OHE + OHE',
                      'method':method,
                      'feature_combination':fc,
                      'p-value':res.pvalue})
  #print(p_values)
  p_values = pd.DataFrame(p_values)
  p_values = p_values.fillna(1)
  _,p_values['adjusted_p-value'],_,_ = multipletests(p_values['p-value'], alpha = _thrsh, method = 'fdr_bh')
  p_values['label'] = ['*' if x < _thrsh and _anova.loc['C(method):C(feature_combination)','PR(>F)']< _thrsh else '' for x in p_values['adjusted_p-value']]

  labels = p_values_df[['method','feature_combination','label']].merge(p_values[['method','feature_combination','label']], how = 'left', on=['method','feature_combination'], suffixes = ['_method','_fc'])
  labels['label'] = labels['label_method'] + '\n' + labels['label_fc']
  
  return labels[['method','feature_combination','label']]

def get_significance_labels_np(df, col_id, tpe = 'perturbation', split = 'RND', _thrsh = THRSH):
  filt = np.logical_and(df['type'] == tpe, df['split'] == split)
  methods = set(df['method'])
  feature_combinations = set(df['feature_combination'])

  p_values=[]
  observations=[]
  filt2 = np.logical_and(df['method'] == 'linear', df['feature_combination'] == 'allfeatures + allfeatures')
  reference = df.loc[np.logical_and(filt, filt2),col_id]

  for method in methods:
    for fc in feature_combinations:
      filt2 = np.logical_and(df['method'] == method, df['feature_combination'] == fc)
      test = df.loc[np.logical_and(filt, filt2),col_id]
      observations.append(test.tolist())

      res = mannwhitneyu(reference, test)
      p_values.append({'vs':'LR + ALL + ALL',
                      'method':method,
                      'feature_combination':fc,
                      'p-value':res.pvalue})
  #print(observations)    
  kruskal_test_full = kruskal(*observations)

  p_values_df = pd.DataFrame(p_values)
  p_values_df = p_values_df.fillna(1)
  _,p_values_df['adjusted_p-value'],_,_ = multipletests(p_values_df['p-value'], alpha = _thrsh, method = 'fdr_bh')
  p_values_df['label'] = ['+' if x < _thrsh and kruskal_test_full.pvalue <= _thrsh else '' for x in p_values_df['adjusted_p-value']]

  p_values=[]
  kw_p_values=[]
  for method in methods:
    filt2 = np.logical_and(df['method'] == method, df['feature_combination'] == 'onehot + onehot')
    reference = df.loc[np.logical_and(filt, filt2),col_id]
    observations=[]
    for fc in feature_combinations:
      filt2 = np.logical_and(df['method'] == method, df['feature_combination'] == fc)
      test = df.loc[np.logical_and(filt, filt2),col_id]
      observations.append(test.tolist())
      res = mannwhitneyu(reference, test)
      p_values.append({'vs':f'{method}+ OHE + OHE',
                      'method':method,
                      'feature_combination':fc,
                      'p-value':res.pvalue})
    kw_p_values.append({'method':method,
                        'kw_p_value':kruskal(*observations).pvalue
                        })
  p_values = pd.DataFrame(p_values)
  p_values = p_values.fillna(1)
  _,p_values['adjusted_p-value'],_,_ = multipletests(p_values['p-value'], alpha = _thrsh, method = 'fdr_bh')
  p_values = p_values.merge(pd.DataFrame(kw_p_values), how = 'left', on = 'method')
  p_values['label'] = ['*' if x < _thrsh and y < _thrsh else '' for x, y in zip(p_values['adjusted_p-value'],p_values['kw_p_value'])]

  labels = p_values_df[['method','feature_combination','label']].merge(p_values[['method','feature_combination','label']], how = 'left', on=['method','feature_combination'], suffixes = ['_method','_fc'])
  labels['label'] = labels['label_method'] + '\n' + labels['label_fc']
  
  return labels[['method','feature_combination','label']]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load evaluation and BD results

# COMMAND ----------

# MAGIC %md
# MAGIC ### eval data

# COMMAND ----------

source_dirs = ['/dbfs/mnt/sandbox/benchmark-paper/final/evals/drug/']
experiment_dict = {}

for source_dir in source_dirs:
  fnames = os.listdir(source_dir)
  fnames = [x for x in fnames if 'geneeffect' not in x]
  for fn in fnames:
    if '.csv' in fn:
      experiment_dict[fn] = fn.split('_') + [source_dir]
      if (len(fn.split('_'))!=6) : print(fn)

experiments_df = pd.DataFrame(experiment_dict).T
experiments_df.columns = ['drug_sensitivity_metric','split','split_no','method','cell_features','drug_features','source_dir']
experiments_df['drug_features'] = experiments_df['drug_features'].str.replace('\\.csv','')
experiments_df

# COMMAND ----------

print(set(experiments_df['cell_features']))
experiments_df.loc[experiments_df['cell_features'] == 'mut', 'cell_features'] = 'mutation'
experiments_df.loc[experiments_df['cell_features'] == 'tpm', 'cell_features'] = 'expression'
print(set(experiments_df['cell_features']))


# COMMAND ----------

collector_df = pd.DataFrame()
for i,row in experiments_df.iterrows():
  tmp_csv = pd.read_csv(row['source_dir'] + i, index_col = 0)
  tmp_csv = tmp_csv.assign(**pd.DataFrame(row).T.iloc[0]) 
  collector_df = pd.concat([collector_df, tmp_csv])
collector_df

# COMMAND ----------

collector_df['feature_combination'] = collector_df['cell_features']+' + '+collector_df['drug_features']
set(collector_df['feature_combination'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### BD data

# COMMAND ----------

bd_dir = '/dbfs/mnt/sandbox/benchmark-paper/final/bds/drug/'
bd_files = os.listdir(bd_dir)
bd_collector_df = pd.DataFrame()
#col_names = ['BD_version','parameter','split','no','method','cell_features','drug_features']
col_names = ['parameter','split','no','method','cell_features','drug_features']
for bd_file in bd_files:
  tmp_desc = pd.DataFrame(bd_file[:-4].split('_')).T
  tmp_desc.columns = col_names
  tmp_csv = pd.read_csv(bd_dir + bd_file, index_col = 0)
  tmp_csv = tmp_csv.assign(**tmp_desc.iloc[0]) 
  bd_collector_df = pd.concat([bd_collector_df, tmp_csv])
bd_collector_df

# COMMAND ----------

print(set(bd_collector_df['cell_features']))
bd_collector_df.loc[bd_collector_df['cell_features'] == 'mut', 'cell_features'] = 'mutation'
bd_collector_df.loc[bd_collector_df['cell_features'] == 'tpm', 'cell_features'] = 'expression'
print(set(bd_collector_df['cell_features']))
bd_collector_df['feature_combination'] = bd_collector_df['cell_features']+' + '+bd_collector_df['drug_features']
bd_collector_df.loc[bd_collector_df['type']=='pert','type'] = 'perturbation'

# COMMAND ----------

THRSH = 0.05 #adjusted p-value threshold to call significant performers

bd_collector_2_sum = bd_collector_df[np.logical_and(bd_collector_df['r']>0, bd_collector_df['p']<=THRSH)].groupby(['type','parameter','split','no','method','feature_combination'])['name'].count().reset_index().rename(columns={'name':'no_BD'}).merge(bd_collector_df.groupby(['type','parameter','split','no','method','feature_combination'])['name'].count().reset_index().rename(columns={'name':'total_no'}), how= 'right').fillna(0)
bd_collector_2_sum['f'] = bd_collector_2_sum['no_BD']/bd_collector_2_sum['total_no']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Plot figures

# COMMAND ----------

# MAGIC %md
# MAGIC ### Figure 4

# COMMAND ----------

prm = 'ic50' # which metric to plot, ['zscore','ic50']

#facets and groups to plot
interesting_combinations = ['allfeatures + allfeatures', 'allfeatures + onehot', 'onehot + allfeatures', 'onehot + onehot']
#interesting_combinations = list(set(collector_df['feature_combination']))
interesting_splits = ['RND','CEX','DEX']
interesting_types = ['perturbation']

corr_df = collector_df[collector_df['drug_sensitivity_metric'] == prm]

filt = np.logical_and(np.logical_and(np.in1d(corr_df['feature_combination'],interesting_combinations), np.in1d(corr_df['type'],interesting_types)), np.in1d(corr_df['split'],interesting_splits))
                    
corr_df = corr_df[filt].copy()
corr_df = corr_df.fillna(0)

filt = np.logical_and(bd_collector_2_sum['parameter']== prm, np.logical_and(np.in1d(bd_collector_2_sum['feature_combination'],interesting_combinations), np.in1d(bd_collector_2_sum['split'],interesting_splits)))
                      
bd_df = bd_collector_2_sum[filt].copy()

plotComposite(corr_df, bd_df, "" , corr_col_id = 'pearson', bd_col_id = 'f', _order=['linear','randomforest','MLP'], labels = ['LR','RF','MLP'], stat_function='nonparametric')
plt.savefig('/dbfs/mnt/sandbox/benchmark-paper/results/figures/fig4_ic50.pdf', format='pdf')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Figure SX, z-scores, identical to Figure 4

# COMMAND ----------

prm = 'zscore' # which metric to plot, ['zscore','ic50']

#facets and groups to plot
interesting_combinations = ['allfeatures + allfeatures', 'allfeatures + onehot', 'onehot + allfeatures', 'onehot + onehot']
interesting_splits = ['RND','CEX','DEX']
interesting_types = ['perturbation']

corr_df = collector_df[collector_df['drug_sensitivity_metric'] == prm]

filt = np.logical_and(np.logical_and(np.in1d(corr_df['feature_combination'],interesting_combinations), np.in1d(corr_df['type'],interesting_types)), np.in1d(corr_df['split'],interesting_splits))
                      
corr_df = corr_df[filt].copy()
corr_df = corr_df.fillna(0)

filt = np.logical_and(bd_collector_2_sum['parameter']== prm, np.logical_and(np.in1d(bd_collector_2_sum['feature_combination'],interesting_combinations), np.in1d(bd_collector_2_sum['split'],interesting_splits)))
                      
bd_df = bd_collector_2_sum[filt].copy()

plotComposite(corr_df, bd_df, '', corr_col_id = 'pearson', bd_col_id = 'f', _order=['linear','randomforest','MLP'], labels = ['LR','RF','MLP'], stat_function='non-parametric')
plt.savefig('/dbfs/mnt/sandbox/benchmark-paper/results/figures/figSX_zscore.pdf', format='pdf')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Figure SY, Full evaluation figure

# COMMAND ----------

prm = 'ic50' # which metric to plot, ['zscore','ic50']

#facets and groups to plot
interesting_combinations = list(set(collector_df['feature_combination']))
interesting_splits = ['RND','CEX','DEX']
interesting_types = ['global','perturbation','cell']

corr_df = collector_df[collector_df['drug_sensitivity_metric'] == prm]

filt = np.logical_and(np.logical_and(np.in1d(corr_df['feature_combination'],interesting_combinations), np.in1d(corr_df['type'],interesting_types)), np.in1d(corr_df['split'],interesting_splits))
                      
corr_df = corr_df[filt].copy()
corr_df = corr_df.fillna(0)

filt = np.logical_and(bd_collector_2_sum['parameter']== prm, np.logical_and(np.in1d(bd_collector_2_sum['feature_combination'],interesting_combinations), np.in1d(bd_collector_2_sum['split'],interesting_splits)))
                      
bd_df = bd_collector_2_sum[filt].copy()

plotComposite(corr_df, bd_df, '' , corr_col_id = 'pearson', bd_col_id = 'f', _order=['linear','randomforest','MLP'], labels = ['LR','RF','MLP'], draw_significance=False)
plt.savefig('/dbfs/mnt/sandbox/benchmark-paper/results/figures/figSY_ic50_full.pdf', format='pdf')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Figure SZ, full evaluation figure z-scores

# COMMAND ----------

prm = 'zscore' # which metric to plot, ['zscore','ic50']

#facets and groups to plot
interesting_combinations = list(set(collector_df['feature_combination']))
interesting_splits = ['RND','CEX','DEX']
interesting_types = ['global','perturbation','cell']

corr_df = collector_df[collector_df['drug_sensitivity_metric'] == prm]

filt = np.logical_and(np.logical_and(np.in1d(corr_df['feature_combination'],interesting_combinations), np.in1d(corr_df['type'],interesting_types)), np.in1d(corr_df['split'],interesting_splits))
                      
corr_df = corr_df[filt].copy()
corr_df = corr_df.fillna(0)

filt = np.logical_and(bd_collector_2_sum['parameter']== prm, np.logical_and(np.in1d(bd_collector_2_sum['feature_combination'],interesting_combinations), np.in1d(bd_collector_2_sum['split'],interesting_splits)))
                      
bd_df = bd_collector_2_sum[filt].copy()

plotComposite(corr_df, bd_df, '' , corr_col_id = 'pearson', bd_col_id = 'f', _order=['linear','randomforest','MLP'], labels = ['LR','RF','MLP'], draw_significance=False)
plt.savefig('/dbfs/mnt/sandbox/benchmark-paper/results/figures/figSZ_zscore_full.pdf', format='pdf')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Supplementary tables

# COMMAND ----------

bd_collector_sum_aggregated = bd_collector_2_sum.groupby(['type','parameter','split','method','feature_combination'])['total_no'].agg('sum').reset_index().merge(bd_collector_2_sum.groupby(['type','parameter','split','method','feature_combination'])['f'].agg(['mean','std','count']).reset_index().rename(columns = {'count':'no_of_splits'}))
bd_collector_sum_aggregated.to_csv('/dbfs/mnt/sandbox/benchmark-paper/results/tables/Table_SY_BD_metrics_aggregated.csv', index = None)

_collector_df_aggregated = collector_df.groupby(['type','drug_sensitivity_metric','split','method','feature_combination'])[['pearson','spearman','rmse']].agg(['mean','std']).reset_index()
_collector_df_aggregated.columns = ['_'.join(col) for col in _collector_df_aggregated.columns]
_collector_df_aggregated.columns = _collector_df_aggregated.columns.str.replace('_$','', regex = True)
_collector_df_aggregated= _collector_df_aggregated.merge(bd_collector_sum_aggregated.rename(columns = {'parameter':'drug_sensitivity_metric','mean':'bdt_fraction_mean','std':'bdt_fraction_std'}))
_collector_df_aggregated.to_csv('/dbfs/mnt/sandbox/benchmark-paper/results/tables/Table_SX_drug_correlation+BD_metrics_aggregated.csv', index = None)
                                                         

# COMMAND ----------

# MAGIC %md
# MAGIC ## Auxillary

# COMMAND ----------



interesting_combinations = ['allfeatures + allfeatures', 'allfeatures + onehot', 'onehot + allfeatures', 'onehot + onehot','expression + target']
interesting_splits = ['RND','CEX','DEX']
interesting_types = ['perturbation','cell']


prm = 'zscore'

tmp_df = collector_df[collector_df['drug_sensitivity_metric'] == prm]
tmp_df['feature_combination'] = tmp_df['cell_features']+' + '+tmp_df['drug_features']

filt = np.logical_and(np.logical_and(np.in1d(tmp_df['feature_combination'],interesting_combinations), np.in1d(tmp_df['type'],interesting_types)), np.in1d(tmp_df['split'],interesting_splits))
                      
tmp_df = tmp_df[filt].copy()
tmp_df = tmp_df.fillna(0)

plotCorr(tmp_df, prm, _order=['linear','randomforest','MLP'], labels = ['LR','RF','MLP'])

# COMMAND ----------

THRSH = 0.05

interesting_combinations = ['allfeatures + allfeatures', 'allfeatures + onehot', 'onehot + allfeatures', 'onehot + onehot','expression + target']
interesting_splits = ['RND','CEX','DEX']
interesting_types = ['perturbation','cell']

bd_collector_2_sum = bd_collector_df[np.logical_and(bd_collector_df['r']>0, bd_collector_df['p']<=THRSH)].groupby(['type','parameter','split','no','method','feature_combination'])['name'].count().reset_index().rename(columns={'name':'no_BD'}).merge(bd_collector_df.groupby(['type','parameter','split','no','method','feature_combination'])['name'].count().reset_index().rename(columns={'name':'total_no'}), how= 'right').fillna(0)
bd_collector_2_sum['f'] = bd_collector_2_sum['no_BD']/bd_collector_2_sum['total_no']

prm = 'zscore'
filt = np.logical_and(bd_collector_2_sum['parameter']== prm, np.logical_and(np.in1d(bd_collector_2_sum['feature_combination'],interesting_combinations), np.in1d(bd_collector_2_sum['split'],interesting_splits)))
                      
tmp_df = bd_collector_2_sum[filt].copy()

plotBias(tmp_df, prm + ' global BD', col_id = 'f', _order=['linear','randomforest','MLP'], labels = ['LR','RF','MLP'])

# COMMAND ----------


