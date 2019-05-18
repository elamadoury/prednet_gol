'''
Train PredNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import numpy as np
np.random.seed(123)
from six.moves import cPickle

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
#K.set_image_data_format('channels_first')


from prednet import PredNet
from data_utils import SequenceGenerator
from Adam_lr_mult import Adam_lr_mult
#file with dataset settings
from datasets_settings.gol_settings import *


save_model = True  # if weights will be saved
weights_file = os.path.join(WEIGHTS_DIR, WEIGHTS_FILE)  # where weights will be saved
json_file =  os.path.join(WEIGHTS_DIR, MODEL_FILE)

# Data files
train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

# Training parameters
nb_epoch = 150
batch_size = 4
samples_per_epoch = 500
N_seq_val = 100  # number of sequences to use for validation

# Model parameters
n_channels, im_height, im_width = (3, 128, 160) #16, 20) #128, 160)
input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (8, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
nt = 10  # number of timesteps used for sequences in training
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
time_loss_weights[0] = 0


prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True)

inputs = Input(shape=(nt,) + input_shape)
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
model = Model(inputs=inputs, outputs=final_errors)

# different learning rates for each layer
# layers
# pred_net_1/layer_a_0/kernel , a_0 to a_2
# pred_net_1/layer_a_0/bias , a_0 to a_2
# pred_net_1/layer_ahat_0/kernel, ahat_0 to 3, bias and kernel
# pred_net_1/layer_c_0/kernel, 0 to 3 bias and kernel
# pred_net_1/layer_f_0/kernel, same as above
# pred_net_1/layer_i_0/kernel, same
# pred_net_1/layer_o_0/kernel, same
# basic learning rate is 0.00031623512

# learning_rate_multipliers = {}
# for i in range(0,n_channels+1):
# 	if i < 3:
# 		layer_name = 'pred_net_1/layer_a_' + str(i) + '/kernel'
# 		learning_rate_multipliers[layer_name] = 0
# 		layer_name = 'pred_net_1/layer_a_' + str(i) + '/bias'
# 		learning_rate_multipliers[layer_name] = 0
# 	layer_name = 'pred_net_1/layer_ahat_' + str(i) + '/kernel'
# 	learning_rate_multipliers[layer_name] = 0
# 	layer_name = 'pred_net_1/layer_ahat_' + str(i) + '/bias'
# 	learning_rate_multipliers[layer_name] = 0
# 	layer_name = 'pred_net_1/layer_c_' + str(i) + '/kernel'
# 	learning_rate_multipliers[layer_name] = 0
# 	layer_name = 'pred_net_1/layer_c_' + str(i) + '/bias'
# 	learning_rate_multipliers[layer_name] = 0
# 	layer_name = 'pred_net_1/layer_f_' + str(i) + '/kernel'
# 	learning_rate_multipliers[layer_name] = 0
# 	layer_name = 'pred_net_1/layer_f_' + str(i) + '/bias'
# 	learning_rate_multipliers[layer_name] = 0
# 	layer_name = 'pred_net_1/layer_i_' + str(i) + '/kernel'
# 	learning_rate_multipliers[layer_name] = 0
# 	layer_name = 'pred_net_1/layer_i_' + str(i) + '/bias'
# 	learning_rate_multipliers[layer_name] = 0
# 	layer_name = 'pred_net_1/layer_o_' + str(i) + '/kernel'
# 	learning_rate_multipliers[layer_name] = 0
# 	layer_name = 'pred_net_1/layer_o_' + str(i) + '/bias'
# 	learning_rate_multipliers[layer_name] = 0

# # keep original bias in 1 conv layer
# del learning_rate_multipliers['pred_net_1/layer_ahat_0/kernel']
# del learning_rate_multipliers['pred_net_1/layer_ahat_0/bias']

# print(learning_rate_multipliers)

# adam_with_lr_multipliers = Adam_lr_mult(multipliers=learning_rate_multipliers, debug_verbose=True)

# use this for original prednet with same lr
model.compile(loss='mean_absolute_error', optimizer='adam')
#model.compile(loss='mean_absolute_error', optimizer=adam_with_lr_multipliers)


train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)

lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks,
                validation_data=val_generator, validation_steps=N_seq_val / batch_size)

if save_model:
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)
