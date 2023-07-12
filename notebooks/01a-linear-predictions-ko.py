# Databricks notebook source
import os

import datetime

import json

import pandas as pd

import numpy as np

from sklearn.linear_model import ElasticNetCV, LassoCV, LogisticRegressionCV, Ridge, RidgeCV

# COMMAND ----------

import mlflow

mlflow.autolog(disable=True)

# COMMAND ----------

TARGETS_PATH = '/dbfs/mnt/sandbox/benchmark-paper/data/final/targets/ko/'
KO_FEATURES_PATH = '/dbfs/mnt/sandbox/benchmark-paper/data/final/features/ko/'
CELL_FEATURES_PATH = '/dbfs/mnt/sandbox/benchmark-paper/data/final/features/cell/'

RESULTS_PATH = '/dbfs/mnt/sandbox/benchmark-paper/data/final/results/'

# COMMAND ----------

model_type = 'linear'
target_types = ['geneeffect']
splits = ['AEX', 'CEX', 'EXT_AEX', 'GEX', 'RND']
nr_of_splits = [0, 1, 2]
cell_line_feature_types = ['expression', 'mutation', 'onehot', 'allfeatures']
ko_feature_types = ['go', 'n2v', 'onehot', 'p2v', 'allfeatures']

# COMMAND ----------

def create_dataset(split_file_name, ko_file_name, cell_line_file_name):
  with open(os.path.join(TARGETS_PATH, split_file_name), mode='r') as input_json:
    split_json = json.load(input_json)
  split_frame = pd.DataFrame(split_json)
  
  y = split_frame['gene_effect'].to_numpy()
  
  ko_features = pd.read_csv(os.path.join(KO_FEATURES_PATH, ko_file_name))
  ko_features.columns.values[0] = 'perturbation'
  ko_features = split_frame[['perturbation']].merge(
    ko_features,
    on='perturbation',
    how='left'
  ).iloc[:, 1:].to_numpy()

  cell_line_features = pd.read_csv(os.path.join(CELL_FEATURES_PATH, cell_line_file_name))
  cell_line_features.columns.values[0] = 'cell_line'
  cell_line_features = split_frame[['cell_line']].merge(
    cell_line_features,
    on='cell_line',
    how='left'
  ).iloc[:, 1:].to_numpy()

  X = np.concatenate([ko_features, cell_line_features], axis=1)

  del ko_features
  del cell_line_features

  return X, y

# COMMAND ----------

result_files_generated = os.listdir(RESULTS_PATH)
result_files_generated

# COMMAND ----------

results_to_be_generated_dict = {}
for target_type in target_types:

  cell_line_feature_types_dict = {}
  for cell_line_feature_type in cell_line_feature_types:

    ko_feature_types_dict = {}
    for ko_feature_type in ko_feature_types:

      nr_of_splits_dict = {}
      for nr_of_split in nr_of_splits:

        splits_dict = {}
        for split in splits:
          results_file_name = f'{target_type}_{split}_{nr_of_split}_{model_type}_{cell_line_feature_type}_{ko_feature_type}.npy'
          
          if results_file_name not in result_files_generated:
            splits_dict[split] = results_file_name
        
        if len(splits_dict) > 0:
          nr_of_splits_dict[nr_of_split] = splits_dict

      if len(nr_of_splits_dict) > 0:
        ko_feature_types_dict[ko_feature_type] = nr_of_splits_dict
  
    if len(ko_feature_types_dict) > 0:
      cell_line_feature_types_dict[cell_line_feature_type] = ko_feature_types_dict

  if len(cell_line_feature_types_dict) > 0:
    results_to_be_generated_dict[target_type] = cell_line_feature_types_dict

results_to_be_generated_dict

# COMMAND ----------

for target_type in results_to_be_generated_dict:
  for cell_line_feature_type in results_to_be_generated_dict[target_type]:
    for ko_feature_type in results_to_be_generated_dict[target_type][cell_line_feature_type]:

      selected_alpha = None
      
      for nr_of_split in results_to_be_generated_dict[target_type][cell_line_feature_type][ko_feature_type]:
    
        start_time = datetime.datetime.now()

        print(f'Fitting for model: {target_type}_{model_type}_{cell_line_feature_type}_{ko_feature_type} {start_time}')

        X_train, y_train = create_dataset(
          split_file_name=f'TRAIN_{nr_of_split}.json',
          ko_file_name=f'{ko_feature_type}.csv',
          cell_line_file_name=f'{cell_line_feature_type}.csv'
        )
        
        if selected_alpha is None:
          model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1.0], cv=5)
          model.fit(X_train, y_train)
          selected_alpha = model.alpha_
        else:
          model = Ridge(alpha=selected_alpha)
          model.fit(X_train, y_train)

        del X_train
        del y_train

        current_time = datetime.datetime.now()
        print(f'Fitted for model: {target_type}_{model_type}_{cell_line_feature_type}_{ko_feature_type} {current_time - start_time}')

        for split in results_to_be_generated_dict[target_type][cell_line_feature_type][ko_feature_type][nr_of_split]:
          
          results_file_name = f'{target_type}_{split}_{nr_of_split}_{model_type}_{cell_line_feature_type}_{ko_feature_type}.npy'

          X_test, y_test = create_dataset(
            split_file_name=f'{split}_{nr_of_split}.json',
            ko_file_name=f'{ko_feature_type}.csv',
            cell_line_file_name=f'{cell_line_feature_type}.csv'
          )

          predictions = model.predict(X_test).reshape(-1)
          np.save(os.path.join(RESULTS_PATH, results_file_name), predictions)

          del X_test
          del y_test

          current_time = datetime.datetime.now()
          print(f'Predicted and saved result file: {results_file_name} {current_time - start_time}')



# COMMAND ----------


