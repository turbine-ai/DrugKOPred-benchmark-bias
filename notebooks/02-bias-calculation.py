# Databricks notebook source
# MAGIC %md
# MAGIC ### Imports and paths

# COMMAND ----------

import pandas as pd
import numpy as np

import json
import os

from statsmodels.formula.api import ols
import mlflow

mlflow.autolog(disable=True)

# COMMAND ----------

def read_json(fname):
    with open(fname) as fin:
        data = json.load(fin)
    return data

# COMMAND ----------

### dirs of drug and ko target data
dir_drug = "/dbfs/mnt/sandbox/benchmark-paper/final/targets/drug/"
dir_ko = "/dbfs/mnt/sandbox/benchmark-paper/final/targets/ko/"
### save bias data to this dir
dir_bias = "/dbfs/mnt/sandbox/benchmark-paper/final/biases/"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate bias for drug data

# COMMAND ----------

### read ful dataset
fnames = os.listdir(dir_drug)
fnames = [x for x in fnames if "_0" in x]
data = []
for fname in fnames:
    data = data + read_json(dir_drug + fname)
data = pd.DataFrame(data)
### pertturbation is durg pubchem id
data["perturbation"] = data["perturbation"].astype(str)

# COMMAND ----------

### bias is fitted as a linear model
model1 = ols("z_score ~ cell_line + perturbation", data=data).fit()
model2 = ols("LN_IC50 ~ cell_line + perturbation", data=data).fit()

# COMMAND ----------

### format ols output for cell line and perturbation
fil = pd.Series(model1.params.index).apply(lambda x: x[:4] == "cell")
model1_cell = model1.params[fil.values]
fil = pd.Series(model1.params.index).apply(lambda x: x[:4] == "pert")
model1_pert = model1.params[fil.values]
model1_cell.index = pd.Series(model1_cell.index).apply(lambda x: x.split(".")[1][:-1])
model1_pert.index = pd.Series(model1_pert.index).apply(lambda x: x.split(".")[1][:-1])

fil = pd.Series(model2.params.index).apply(lambda x: x[:4] == "cell")
model2_cell = model2.params[fil.values]
fil = pd.Series(model2.params.index).apply(lambda x: x[:4] == "pert")
model2_pert = model2.params[fil.values]
model2_cell.index = pd.Series(model2_cell.index).apply(lambda x: x.split(".")[1][:-1])
model2_pert.index = pd.Series(model2_pert.index).apply(lambda x: x.split(".")[1][:-1])

# COMMAND ----------

### define bias matrix
cell_bias_zscore = pd.DataFrame(
    0.0, index=data["cell_line"].unique(), columns=["cell_bias"]
)
cell_bias_ic50 = pd.DataFrame(
    0.0, index=data["cell_line"].unique(), columns=["cell_bias"]
)
drug_bias_zscore = pd.DataFrame(
    0.0, index=data["perturbation"].unique(), columns=["drug_bias"]
)
drug_bias_ic50 = pd.DataFrame(
    0.0, index=data["perturbation"].unique(), columns=["drug_bias"]
)

# COMMAND ----------

cell_bias_zscore.loc[
    model1_cell.index,
    "cell_bias",
] = model1_cell.values

cell_bias_ic50.loc[
    model2_cell.index,
    "cell_bias",
] = model2_cell.values

drug_bias_zscore.loc[
    model1_pert.index,
    "drug_bias",
] = model1_pert.values

drug_bias_ic50.loc[
    model2_pert.index,
    "drug_bias",
] = model2_pert.values

# COMMAND ----------

cell_bias_zscore.to_csv(dir_bias + "cell_bias_zscore.csv")
cell_bias_ic50.to_csv(dir_bias + "cell_bias_ic50.csv")
drug_bias_zscore.to_csv(dir_bias + "pert_bias_zscore.csv")
drug_bias_ic50.to_csv(dir_bias + "pert_bias_ic50.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate bias for gene effect

# COMMAND ----------

### read full dataset
fnames = os.listdir(dir_ko)
fnames = [x for x in fnames if "_0" in x]
### do not calcualte bias for extended
fnames = [x for x in fnames if "EXT" not in x]
data = []
for fname in fnames:
    data = data + read_json(dir_ko + fname)
data = pd.DataFrame(data)

# COMMAND ----------

### bias is fitted as a linear model
model = ols("gene_effect ~ cell_line + perturbation", data=data).fit()

# COMMAND ----------

### format ols output for cell line and perturbation
fil = pd.Series(model.params.index).apply(lambda x: x[:4] == "cell")
model_cell = model.params[fil.values]
fil = pd.Series(model.params.index).apply(lambda x: x[:4] == "pert")
model_pert = model.params[fil.values]
model_cell.index = pd.Series(model_cell.index).apply(lambda x: x.split(".")[1][:-1])
model_pert.index = pd.Series(model_pert.index).apply(lambda x: x.split(".")[1][:-1])

# COMMAND ----------

cell_bias = pd.DataFrame(0.0, index=data["cell_line"].unique(), columns=["cell_bias"])
pert_bias = pd.DataFrame(
    0.0, index=data["perturbation"].unique(), columns=["pert_bias"]
)

# COMMAND ----------

cell_bias.loc[
    model_cell.index,
    "cell_bias",
] = model_cell.values

pert_bias.loc[
    model_pert.index,
    "pert_bias",
] = model_pert.values

# COMMAND ----------

cell_bias.to_csv(dir_bias + "cell_bias_geneeffect.csv")
pert_bias.to_csv(dir_bias + "pert_bias_geneffect.csv")
