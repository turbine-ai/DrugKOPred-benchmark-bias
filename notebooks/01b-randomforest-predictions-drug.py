# Databricks notebook source
import os

import datetime

import json

import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor

# COMMAND ----------

import mlflow

mlflow.autolog(disable=True)

# COMMAND ----------

TARGETS_PATH = '/dbfs/mnt/sandbox/benchmark-paper/data/final/targets/drug/'
DRUG_FEATURES_PATH = '/dbfs/mnt/sandbox/benchmark-paper/data/final/features/drug/'
CELL_FEATURES_PATH = '/dbfs/mnt/sandbox/benchmark-paper/data/final/features/cell/'

RESULTS_PATH = '/dbfs/mnt/sandbox/benchmark-paper/data/final/results/'

# COMMAND ----------

model_type = 'randomforest'
target_types = ['ic50', 'zscore']
splits = ['AEX', 'CEX', 'DEX', 'RND']
nr_of_splits = [0, 1, 2, 3, 4]
cell_line_feature_types = ['expression', 'mutation', 'onehot', 'allfeatures']
drug_feature_types = ['target', 'affinity', 'fingerprint', 'onehot', 'allfeatures']

# COMMAND ----------

def create_dataset(split_file_name, target_column, drug_file_name, cell_line_file_name):
  with open(os.path.join(TARGETS_PATH, split_file_name), mode='r') as input_json:
    split_json = json.load(input_json)
  split_frame = pd.DataFrame(split_json)
  
  if target_column == 'ic50':
    y = split_frame['LN_IC50'].to_numpy()
  elif target_column == 'zscore':
    y = split_frame['z_score'].to_numpy()
  else:
    raise Exception(f'Illegal target column provided! {target_column}')

  split_frame['perturbation'] = split_frame['perturbation'].astype('int64')
  
  drug_features = pd.read_csv(os.path.join(DRUG_FEATURES_PATH, drug_file_name))
  drug_features.columns.values[0] = 'perturbation'
  drug_features = split_frame[['perturbation']].merge(
    drug_features,
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

  X = np.concatenate([drug_features, cell_line_features], axis=1)

  del drug_features
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

    drug_feature_types_dict = {}
    for drug_feature_type in drug_feature_types:

      nr_of_splits_dict = {}
      for nr_of_split in nr_of_splits:

        splits_dict = {}
        for split in splits:
          results_file_name = f'{target_type}_{split}_{nr_of_split}_{model_type}_{cell_line_feature_type}_{drug_feature_type}.npy'
          
          if results_file_name not in result_files_generated:
            splits_dict[split] = results_file_name
        
        if len(splits_dict) > 0:
          nr_of_splits_dict[nr_of_split] = splits_dict

      if len(nr_of_splits_dict) > 0:
        drug_feature_types_dict[drug_feature_type] = nr_of_splits_dict
  
    if len(drug_feature_types_dict) > 0:
      cell_line_feature_types_dict[cell_line_feature_type] = drug_feature_types_dict

  if len(cell_line_feature_types_dict) > 0:
    results_to_be_generated_dict[target_type] = cell_line_feature_types_dict

results_to_be_generated_dict

# COMMAND ----------

for target_type in results_to_be_generated_dict:
  for cell_line_feature_type in results_to_be_generated_dict[target_type]:
    for drug_feature_type in results_to_be_generated_dict[target_type][cell_line_feature_type]:
      
      for nr_of_split in results_to_be_generated_dict[target_type][cell_line_feature_type][drug_feature_type]:
    
        start_time = datetime.datetime.now()

        print(f'Fitting for model: {target_type}_{model_type}_{cell_line_feature_type}_{drug_feature_type} {start_time}')

        X_train, y_train = create_dataset(
          split_file_name=f'TRAIN_{nr_of_split}.json',
          target_column=target_type,
          drug_file_name=f'{drug_feature_type}.csv',
          cell_line_file_name=f'{cell_line_feature_type}.csv'
        )
        
        model = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
        model.fit(X_train, y_train)

        del X_train
        del y_train

        current_time = datetime.datetime.now()
        print(f'Fitted for model: {target_type}_{model_type}_{cell_line_feature_type}_{drug_feature_type} {current_time - start_time}')

        for split in results_to_be_generated_dict[target_type][cell_line_feature_type][drug_feature_type][nr_of_split]:
          
          results_file_name = f'{target_type}_{split}_{nr_of_split}_{model_type}_{cell_line_feature_type}_{drug_feature_type}.npy'

          X_test, y_test = create_dataset(
            split_file_name=f'{split}_{nr_of_split}.json',
            target_column=target_type,
            drug_file_name=f'{drug_feature_type}.csv',
            cell_line_file_name=f'{cell_line_feature_type}.csv'
          )

          predictions = model.predict(X_test).reshape(-1)
          np.save(os.path.join(RESULTS_PATH, results_file_name), predictions)

          del X_test
          del y_test

          current_time = datetime.datetime.now()
          print(f'Predicted and saved result file: {results_file_name} {current_time - start_time}')



# COMMAND ----------


