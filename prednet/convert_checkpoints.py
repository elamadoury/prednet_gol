'''
Extract single GPU model weights from a checkpoint of a multi-gpu model.
'''

import os

from keras import backend as K
from keras.models import model_from_json
from keras.utils import multi_gpu_model
from prednet import PredNet
from datasets_settings.kitti_settings import *

num_gpus = 4  # Number of GPUs the model was trained on. Not the number of available GPUs.

weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json') 

# Read the model file
f = open(json_file, 'r')
json_string = f.read()
f.close()

# Load the model
model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
distributed_model = multi_gpu_model(model, gpus=num_gpus) 

# Load the weights of the model
distributed_model.load_weights(weights_file)

# Extract the weights for a single gpu model
single_gpu_model = distributed_model.layers[-2]

# Save the new weights
single_gpu_model.save(WEIGHTS_DIR+'prednet_kitti_weights_single_gpu.hdf5')

