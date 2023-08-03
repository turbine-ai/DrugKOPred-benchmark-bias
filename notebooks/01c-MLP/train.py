import torch
torch.set_float32_matmul_precision("medium") # to make lightning happy
import pytorch_lightning as pl
from model import MLP
from dataset import get_exp_list, get_dataloader
import config
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def single_gpu_pred(model,dataloader,pred_path):
    preds = []
    print(f"Dataloader len: {len(dataloader)}")
    for batch in tqdm(dataloader):
        x,y_true = batch
        y = model(x)
        preds.append(y.detach().cpu())
    pred = torch.concat(preds).numpy()
    np.save(pred_path,pred)


if __name__ == "__main__":
    for msplit in config.MAJOR_SPLITS: 
        config.PERT_TYPE, config.CL_FEATURES, config.PERT_FEATURES = msplit

        CELL_FEATURES_PATH = os.path.join(config.DATA_DIR,f'features/cell/{config.CL_FEATURES}.csv')
        PERT_FEATURES_PATH = os.path.join(config.DATA_DIR,f'features/{config.PERT_TYPE}/{config.PERT_FEATURES}.csv')
        features_dir = os.path.join(config.DATA_DIR,f'targets/{config.PERT_TYPE}')

        CELL_FEATURES = pd.read_csv(CELL_FEATURES_PATH)
        PERT_FEATURES = pd.read_csv(PERT_FEATURES_PATH)

        pert_features_npy = PERT_FEATURES.iloc[:,1:].to_numpy()
        cell_line_features_npy = CELL_FEATURES.iloc[:,1:].to_numpy()
        INPUT_FEATURE_SIZE = cell_line_features_npy.shape[1]+pert_features_npy.shape[1]

        pert_lookup = PERT_FEATURES.iloc[:,0]
        cell_line_lookup = CELL_FEATURES.iloc[:,0]

        if config.PERT_TYPE=='ko':
            PERTEX='GEX'
        elif config.PERT_TYPE=='drug':
            PERTEX='DEX'

        max_sample_num=None  # None for full dataset, else first X samples
        for SPLIT in range(config.NUM_SPLITS):
            train_exp_path = os.path.join(features_dir,f'TRAIN_{SPLIT}.json')
            train_exp_list = get_exp_list(train_exp_path,cell_line_lookup,pert_lookup,max_sample_num=max_sample_num, pert_type=config.PERT_TYPE, target=config.TARGET)
            train_loader = get_dataloader(train_exp_list,pert_features_npy,cell_line_features_npy,batch_size=config.BATCH_SIZE,shuffle=True)

            test_cex_exp_path = os.path.join(features_dir,f'CEX_{SPLIT}.json')
            test_cex_exp_list = get_exp_list(test_cex_exp_path,cell_line_lookup,pert_lookup,max_sample_num=max_sample_num, pert_type=config.PERT_TYPE,target=config.TARGET)
            test_cex_loader = get_dataloader(test_cex_exp_list,pert_features_npy,cell_line_features_npy,batch_size=config.PRED_BATCH_SIZE,shuffle=False)

            test_gex_exp_path = os.path.join(features_dir,f'{PERTEX}_{SPLIT}.json')
            test_gex_exp_list = get_exp_list(test_gex_exp_path,cell_line_lookup,pert_lookup,max_sample_num=max_sample_num, pert_type=config.PERT_TYPE,target=config.TARGET)
            pertex_loader = get_dataloader(test_gex_exp_list,pert_features_npy,cell_line_features_npy,batch_size=config.PRED_BATCH_SIZE,shuffle=False)

            test_aex_exp_path = os.path.join(features_dir,f'AEX_{SPLIT}.json')
            test_aex_exp_list = get_exp_list(test_aex_exp_path,cell_line_lookup,pert_lookup,max_sample_num=max_sample_num, pert_type=config.PERT_TYPE,target=config.TARGET)
            test_aex_loader = get_dataloader(test_aex_exp_list,pert_features_npy,cell_line_features_npy,batch_size=config.PRED_BATCH_SIZE,shuffle=False)

            test_rnd_exp_path = os.path.join(features_dir,f'RND_{SPLIT}.json')
            test_rnd_exp_list = get_exp_list(test_rnd_exp_path,cell_line_lookup,pert_lookup,max_sample_num=max_sample_num, pert_type=config.PERT_TYPE,target=config.TARGET)
            test_rnd_loader = get_dataloader(test_rnd_exp_list,pert_features_npy,cell_line_features_npy,batch_size=config.PRED_BATCH_SIZE,shuffle=False)

            INPUT_SIZE= next(iter(train_loader))[0].shape[1]

            model = MLP(
                input_feature_size=INPUT_SIZE,
                layer_num=config.LAYER_NUM,
                hidden_dim=config.HIDDEN_DIM,
                learning_rate=config.LEARNING_RATE,
                dropout_rate=config.DROPOUT_RATE
            )

            if config.TARGET=='z_score':
                prefix='zscore'
            elif config.TARGET=='LN_IC50':
                prefix='ic50'
            elif config.TARGET=='gene_effect':
                prefix='geneeffect'
            elif config.TARGET=='auc':
                prefix='auc'

            log_subdir = os.path.join(config.LOG_DIR,f"{prefix}_{config.CL_FEATURES}_{config.PERT_FEATURES}_{config.LAYER_NUM}_{config.HIDDEN_DIM}_{config.DROPOUT_RATE}_BS{config.BATCH_SIZE}")
            os.makedirs(log_subdir,exist_ok=True)

            trainer = pl.Trainer(
                strategy="ddp",
                logger=TensorBoardLogger(log_subdir,  name=f"split_{SPLIT}"),
                callbacks=[ModelCheckpoint(filename="model_{epoch:02d}",save_top_k=-1, monitor="val_RND_MSE_loss/dataloader_idx_3")], #this monitors RND metrics
                accelerator=config.ACCELERATOR,
                devices=config.DEVICES,
                min_epochs=1,
                max_epochs=config.NUM_EPOCHS,
                precision=config.PRECISION
            )

            trainer.fit(model=model,
                    train_dataloaders=train_loader,
                    val_dataloaders=[test_cex_loader,pertex_loader,test_aex_loader,test_rnd_loader])
            best_epoch_dir_path = trainer.checkpoint_callback.best_model_path.strip('.ckpt')
            os.makedirs(best_epoch_dir_path,exist_ok=True)

            if trainer.global_rank==0:
                model.eval()
                single_gpu_pred(model, test_cex_loader, os.path.join(best_epoch_dir_path,f'{prefix}_CEX_{SPLIT}_MLP_{config.CL_FEATURES}_{config.PERT_FEATURES}.npy'))
                single_gpu_pred(model, pertex_loader, os.path.join(best_epoch_dir_path,f'{prefix}_{PERTEX}_{SPLIT}_MLP_{config.CL_FEATURES}_{config.PERT_FEATURES}.npy'))
                single_gpu_pred(model, test_aex_loader, os.path.join(best_epoch_dir_path,f'{prefix}_AEX_{SPLIT}_MLP_{config.CL_FEATURES}_{config.PERT_FEATURES}.npy'))
                single_gpu_pred(model, test_rnd_loader, os.path.join(best_epoch_dir_path,f'{prefix}_RND_{SPLIT}_MLP_{config.CL_FEATURES}_{config.PERT_FEATURES}.npy'))
            else:
                pass
            trainer.strategy.barrier()
