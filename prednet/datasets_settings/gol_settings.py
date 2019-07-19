# Where data will be saved if you run process_kitti.py
DATA_DIR = 'D:/ShareData/gol_data/gol_hkl/random/'#'./kitti_data/'

# Where model weights and config will be saved if you run kitti_train.py
# If you directly download the trained weights, change to appropriate path.
WEIGHTS_DIR = './weights_data/single_layer_gol_1000/'
# WEIGHTS_DIR = './weights_data/gol_prednet_param/'

WEIGHTS_FILE = 'prednet_weights.hdf5'

# Where results (prediction plots and evaluation file) will be saved.
RESULTS_SAVE_DIR = './results/single_layer_gol_1000/'
# RESULTS_SAVE_DIR = './results/gol_prednet_param/'

MODEL_FILE = 'prednet_model.json'
