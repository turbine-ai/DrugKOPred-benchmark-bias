# Databricks notebook source
# MAGIC %md
# MAGIC ### Imports and paths

# COMMAND ----------

import pandas as pd
import numpy as np

import seaborn as sns

import json
import os

from tqdm import tqdm

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

# COMMAND ----------

def read_json(fname):
    with open(fname) as fin:
        data = json.load(fin)
    return data


def calc_regression_metrics(y_true, y_pred):
    """calculates pearsonr, spearmanr and rmse for input"""
    n = len(y_true)
    if (n > 1) & (y_true.std() > 1e-10) & (y_pred.std() > 1e-10):
        r1 = pearsonr(y_true, y_pred)[0]
        r2 = spearmanr(y_true, y_pred)[0]
    else:
        r1 = 0.0
        r2 = 0.0
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return [r1, r2, rmse, n]


def make_all_eval(data):
    """calculates global, per cell and per pert metrics"""
    truth = pd.pivot_table(
        data, index="cell_line", columns="perturbation", values="target"
    )
    pred = pd.pivot_table(
        data, index="cell_line", columns="perturbation", values="pred"
    )
    truth = truth.loc[pred.index, pred.columns]
    results = pd.DataFrame(columns=["type", "name", "pearson", "spearman", "rmse", "n"])

    results.loc[len(results)] = ["global", "global"] + calc_regression_metrics(
        data["target"], data["pred"]
    )
    for cell in truth.index:
        y_true = truth.loc[cell]
        y_pred = pred.loc[cell]
        fil = y_true.isna() | y_pred.isna()
        results.loc[len(results)] = ["cell", cell] + calc_regression_metrics(
            y_true[~fil], y_pred[~fil]
        )

    for pert in truth.columns:
        y_true = truth[pert]
        y_pred = pred[pert]
        fil = y_true.isna() | y_pred.isna()
        results.loc[len(results)] = ["perturbation", pert] + calc_regression_metrics(
            y_true[~fil], y_pred[~fil]
        )
    return results


def split_fname(fname):
    metrics, split_type, split, model_type, cell_feature, pert_feature = fname[
        :-4
    ].split("_")
    return metrics, split_type, split, model_type, cell_feature, pert_feature

# COMMAND ----------

### replace them according to your setup
PRED_DIR = '/dbfs/mnt/sandbox/benchmark-paper/final/predictions/'
TARGET_DIR= '/dbfs/mnt/sandbox/benchmark-paper/final/targets/'
EVALS_DIR = '/dbfs/mnt/sandbox/benchmark-paper/final/evals//'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation

# COMMAND ----------

### for drugs
fnames = os.listdir(PRED_DIR + "drug/")
for fname in tqdm(fnames):
    pred = np.load(PRED_DIR + "drug/" + fname)
    metrics, split_type, split, model_type, cell_feature, pert_feature = split_fname(
        fname
    )
    ### read corresponding target file
    data = pd.DataFrame(read_json(TARGET_DIR + "drug/%s_%s.json" % (split_type, split)))
    if metrics == "ic50":
        data = data[["cell_line", "perturbation", "LN_IC50"]]
    elif metrics == "zscore":
        data = data[["cell_line", "perturbation", "z_score"]]
    else:
        raise Exception("Sorry, not good metrics!")
    data.columns = ["cell_line", "perturbation", "target"]
    ### perturbation is pubchem id
    data["perturbation"] = data["perturbation"].astype(str)
    data["pred"] = pred
    results = make_all_eval(data)
    results.to_csv(EVALS_DIR + "drug/" + fname[:-4] + ".csv")

# COMMAND ----------

### for cell lines
fnames = os.listdir(PRED_DIR + "ko/")
for fname in tqdm(fnames):
    pred = np.load(PRED_DIR + "ko/" + fname)
    metrics, split_type, split, model_type, cell_feature, pert_feature = split_fname(
        fname
    )
    ### read corresponding target file
    data = pd.DataFrame(read_json(TARGET_DIR + "ko/%s_%s.json" % (split_type, split)))
    data.columns = ["cell_line", "perturbation", "target"]
    data["pred"] = pred
    results = make_all_eval(data)
    results.to_csv(EVALS_DIR + "ko/" + fname[:-4] + ".csv")

# COMMAND ----------


