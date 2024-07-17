import os
import numpy as np
import sys
import tensorflow as tf
sys.path.append(r'/home/cs-solomon.asghar/AnDi_2024/software/')
from UNet3P_var import *
from UNet_Blocks import *
from DataGen_var_EMS import *
from Training_var_EMS import *
from utils import DiffTrajs

# Make the model
max_traj_len = 224
filters = [16, 32, 64, 64, 128, 128]

ConvBlockParams = {'num_filters': 128,
                   'kernel_size': 3,
                   'strides': 1,
                   'padding': 'same'}

SkipBlockParams = {'num_filters': 512,
                   'kernel_size': 3,
                   'strides': 1,
                   'padding': 'same'}

DecoderBlockParams = {'num_filters': 512,
                      'kernel_size' :3,
                      'strides': 1,
                      'padding': "same"}

model = UNet3P_var(filters, ConvBlockSimple, ConvBlockParams, SkipBlockParams, DecoderBlockParams, input_len=max_traj_len)

# Load the old net!
old_path = '/home/cs-solomon.asghar/AnDi_2024/ChallengeNets/Nets_6/ExpNet_0/model_checkpoint'
model.load_weights(old_path)

# Compile it 
losses = {"CPs": "binary_crossentropy",
          "K_and_alpha_and_class": "MSE"}
model.compile(optimizer="adam", loss=losses, metrics=["acc"])

# Train it
Ensemble_Mods = ['single_state', 'confinement', 'dimerization', 'immobile_traps']
Ensemble_Dist = [{'D_mean': 0.104229422, 'D_var': 0.001166479, 'alpha_mean': 0.970239179, 'alpha_var': 0.026103678}]
path = '/home/cs-solomon.asghar/AnDi_2024/ChallengeNets/Nets_6b/ExpNet_0/model_checkpoint'

# if new path exists, load it too!
if os.path.isfile(path + '.index'):
    model.load_weights(path)  

model = UnlimitedTrain_EMS(model, checkpoint_path=path, 
                           ensemble_models=Ensemble_Mods,ensemble_distributions=Ensemble_Dist,
                           data_post_process=DiffTrajs, data_gen_per_iteration=50000, verbose=True, max_traj_len=max_traj_len, patience=3)