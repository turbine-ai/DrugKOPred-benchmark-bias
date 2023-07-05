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

from scipy.stats import pearsonr
from statsmodels.formula.api import ols

import mlflow

mlflow.autolog(disable=True)

# COMMAND ----------

def read_json(fname):
    with open(fname) as fin:
        data = json.load(fin)
    return data


def split_fname(fname):
    """get model information from file name"""
    metrics, split_type, split, model_type, cell_feature, pert_feature = fname[
        :-4
    ].split("_")
    return metrics, split_type, split, model_type, cell_feature, pert_feature


def bias_detector(data):
    """calculates partial correlation between y_true and y_pred)"""
    n = len(data)
    if (n > 1) & (data["target"].std() > 1e-10) & (data["pred"].std() > 1e-10):
        model1 = ols("target ~ cell_bias + pert_bias", data).fit()
        model2 = ols("pred~ cell_bias + pert_bias", data).fit()
        r, p = pearsonr(model1.resid, model2.resid)
        return [r, p]
    else:
        return [0.0, 1.0]


def bias_detector_wrapper(data):
    """calculates global, per perturbation and per cell line bias_detector scores"""
    results = pd.DataFrame(columns=["type", "name", "r", "p"])
    results.loc[len(results)] = ["global", "global"] + bias_detector(data)
    for cell in data["cell_line"].unique():
        fil = data["cell_line"] == cell
        results.loc[len(results)] = ["cell", cell] + bias_detector(data[fil])
    for pert in data["perturbation"].unique():
        fil = data["perturbation"] == pert
        results.loc[len(results)] = ["pert", pert] + bias_detector(data[fil])
    return results

# COMMAND ----------

### replace them according to your setup
PRED_DIR = "/dbfs/mnt/sandbox/benchmark-paper/final/predictions/"
TARGET_DIR = "/dbfs/mnt/sandbox/benchmark-paper/final/targets/"
BIAS_DIR = "/dbfs/mnt/sandbox/benchmark-paper/final/biases/"
BD_DIR = "/dbfs/mnt/sandbox/benchmark-paper/final/bds/"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bias detector

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
        cell_bias = pd.read_csv(
            BIAS_DIR + "cell_bias_ic50.csv", sep=",", header=0, index_col=0
        )
        pert_bias = pd.read_csv(
            BIAS_DIR + "pert_bias_ic50.csv", sep=",", header=0, index_col=0
        )
        pert_bias.index = pert_bias.index.astype(str)
    elif metrics == "zscore":
        data = data[["cell_line", "perturbation", "z_score"]]
        cell_bias = pd.read_csv(
            BIAS_DIR + "cell_bias_zscore.csv", sep=",", header=0, index_col=0
        )
        pert_bias = pd.read_csv(
            BIAS_DIR + "pert_bias_zscore.csv", sep=",", header=0, index_col=0
        )
        pert_bias.index = pert_bias.index.astype(str)
    else:
        raise Exception("Sorry, not good metrics!")
    data.columns = ["cell_line", "perturbation", "target"]
    ### perturbation is pubchem id
    data["perturbation"] = data["perturbation"].astype(str)
    data["pred"] = pred
    data["cell_bias"] = cell_bias.loc[data["cell_line"].values, "cell_bias"].values
    data["pert_bias"] = pert_bias.loc[data["perturbation"].values, "drug_bias"].values
    results = bias_detector_wrapper(data)
    results.to_csv(BD_DIR + "drug/" + fname[:-4] + ".csv")

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
    data = data[["cell_line", "perturbation", "gene_effect"]]
    cell_bias = pd.read_csv(
        BIAS_DIR + "cell_bias_geneeffect.csv", sep=",", header=0, index_col=0
    )
    pert_bias = pd.read_csv(
        BIAS_DIR + "pert_bias_geneeffect.csv", sep=",", header=0, index_col=0
    )
    data.columns = ["cell_line", "perturbation", "target"]
    ### perturbation is pubchem id
    data["perturbation"] = data["perturbation"].astype(str)
    data["pred"] = pred
    data["cell_bias"] = cell_bias.loc[data["cell_line"].values, "cell_bias"].values
    data["pert_bias"] = pert_bias.loc[data["perturbation"].values, "pert_bias"].values
    results = bias_detector_wrapper(data)
    results.to_csv(BD_DIR + "ko/" + fname[:-4] + ".csv")

# COMMAND ----------


