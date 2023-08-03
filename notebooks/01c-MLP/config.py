# CONFIG

# Options for input features:
#   PERT_TYPE = ['ko','drug']
#   PERT_FEATURES = ['1hot','n2v','go','p2v','all'] # for KO
#   PERT_FEATURES = ['1hot','fingerprint','target','affinity','all'] # for drug
#   CL_FEATURES = ['1hot','tpm','mut']

# Options for TARGET variable:
#   KO perturbations : 'gene_effect'
#   drug pertrubations: 'auc', 'LN_IC50', 'z_score'
TARGET='gene_effect'

# MAJOR_SPLITS collects experimental setups as a list of lists
MAJOR_SPLITS = [
    ['ko','1hot','all'],
    ['ko','tpm','all'],
    ['ko','mut','all'],
    ['ko','all','all'],
    ['ko','all','1hot'],
    ['ko','all','n2v'],
    ['ko','all','go'],
    ['ko','all','p2v'],
    ['ko','1hot','1hot'],
    ['ko','1hot','n2v'],
    ['ko','1hot','go'],
    ['ko','1hot','p2v'],
    ['ko','tpm','1hot'],
    ['ko','mut','1hot'],
    # ['drug','1hot','all_new'],
    # ['drug','tpm','all_new'],
    # ['drug','mut','all_new'],
    # ['drug','all','all_new'],
    # ['drug','all','1hot'],
    # ['drug','all','fingerprint'],
    # ['drug','all','target'],
    # ['drug','all','affinity_new'],
    # ['drug','1hot','1hot'],
    # ['drug','1hot','fingerprint'],
    # ['drug','1hot','target'],
    # ['drug','1hot','affinity_new'],
    # ['drug','tpm','1hot'],
    # ['drug','mut','1hot']
]

# Training hyperparameters
LAYER_NUM = 3
HIDDEN_DIM = 128
LEARNING_RATE=1e-3
NUM_EPOCHS = 10
BATCH_SIZE = 256
PRED_BATCH_SIZE= 1024
DROPOUT_RATE=0.2

# Dataset
# DATA_DIR should point to the directory containing "features" and "targets"
DATA_DIR = "/dbfs/mnt/sandbox/benchmark-paper/data"

# LOG_DIR will contain the logs and predictions of each experimental run collected in MAJOR_SPLITS.
# The best epoch's predictions in each split are saved in the following directory tree structure:
# /LOG_DIR/MAJOR_SPLIT/CV_SPLIT/version_0/checkpoints/model_epoch=<best_epoch>/
LOG_DIR = "/dbfs/mnt/sandbox/benchmark-paper/logdir"
NUM_SPLITS = 3          # number of cross-validation splits

# Compute related
ACCELERATOR = "gpu"     # CPU or GPU
DEVICES = 8             # number of GPU accelerators
PRECISION = 16