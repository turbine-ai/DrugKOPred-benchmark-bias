import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset,DataLoader
import os
import json
import pandas as pd

class CombiDataset(Dataset):
    def __init__(self, pert_feature_lookup, cell_line_feature_lookup, exps):
        self.pertf_lookup = pert_feature_lookup
        self.clf_lookup = cell_line_feature_lookup
        self.exps = exps
    def __len__(self):
        return len(self.exps)
    def __getitem__(self, idx):
        sample = self.exps[idx]
        cell_line = sample[0]
        node = sample[1]
        y_true = torch.tensor([sample[2]])

        pert_feature = torch.from_numpy(self.pertf_lookup[node])
        cl_feature = torch.from_numpy(self.clf_lookup[cell_line])
        feature = torch.concat([pert_feature,cl_feature]).float()
        return feature, y_true

def get_exp_list(exp_json_path,cell_line_id_lookup,perturbation_id_lookup,max_sample_num=None, pert_type='ko',target='gene_effect'):
    print(f'loading experiments from file: {exp_json_path}')
    EXPS = pd.DataFrame.from_dict(json.load(open(exp_json_path,'r')))

    cl_dict = {v: k for k, v in cell_line_id_lookup.to_dict().items()}
    pert_dict = {v: k for k, v in perturbation_id_lookup.to_dict().items()}
    cl_ind = EXPS.cell_line.map(lambda x: cl_dict[x]).values

    if target=='auc':
        EXPS['target'] = EXPS.auc.astype('float32')
    elif target=='LN_IC50':
        EXPS['target'] = EXPS.LN_IC50.values.astype('float32')
    elif target=='z_score':
        EXPS['target'] = EXPS.z_score.values.astype('float32')
    elif target=='gene_effect':
        EXPS['target'] = EXPS.gene_effect.values.astype('float32')
    
    if pert_type=='ko':
        pert_ind = EXPS.perturbation.map(lambda x: pert_dict[x]).values
    if pert_type=='drug':
        pert_ind = EXPS.perturbation.map(lambda x: pert_dict[int(x)]).values
    targets = EXPS.target.values
    
    return list(zip(cl_ind,pert_ind,targets))[:max_sample_num]

def get_dataloader(datalist,pert_features,cell_line_features,batch_size,shuffle=False):
    dataset = CombiDataset(pert_features,cell_line_features,datalist)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
