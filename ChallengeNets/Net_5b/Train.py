import numpy as np
import sys
import tensorflow as tf
sys.path.append(r'/home/cs-solomon.asghar/AnDi_2024/software/')
from UNet3P_var_M2 import *
from UNet_Blocks import *
from DataGen_var_M2 import *
from Training_var_M2 import *
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

model = UNet3P_var_M(filters, ConvBlockSimple, ConvBlockParams, SkipBlockParams, DecoderBlockParams, input_len=max_traj_len)

# NORMAL TRAINING # 
losses = {"CPs": "binary_crossentropy",
          "K_and_alpha_and_class": "MSE",
          "Model": "categorical_crossentropy"}    # the axis crossentropy is calculated across is -1, this works fine
model.compile(optimizer="adam", loss=losses)

# Set file to save to
path = '/home/cs-solomon.asghar/AnDi_2024/ChallengeNets/Net_5b/model_checkpoint'
    
# Unlimited training
model = UnlimitedTrain_M(model, checkpoint_path=path, data_post_process=DiffTrajs, data_gen_per_iteration=50000, verbose=True, max_traj_len=max_traj_len)