import tensorflow as tf
import sys
sys.path.append(r'/home/cs-solomon.asghar/AnDi_2024/software/')
from UNet_Default_Block_Params import *
from UNet_Blocks import *

def UNet(filters, ComvBlock, BlockParams):
    input_len = 128
    
    inputs = tf.keras.layers.Input((input_len, 2))
    
    x = inputs
    skips = []
    
    ## Encoder
    for f in filters[:-1]:
        BlockParams['num_filters'] = f
        s, x = Encoder(x, ConvBlock, BlockParams) 
        skips += [s]
    
    ## Bottleneck
    BlockParams['num_filters'] = filters[-1]
    x = ConvBlock(x, BlockParams)
    
    ## Decoder
    for f, s in zip(filters[:-1][::-1], skips[::-1]):
        UpSamplingParams['num_filters'] = f
        BlockParams['num_filters'] = f
        s = ConvBlock(s, BlockParams)
        x = Decoder(x, s, ConvBlock, BlockParams, UpSamplingParams)
    
    ## FinalTransform
    x = tf.keras.layers.Conv1D(3, 1, activation='linear')(x)
    
    CPs = x[:,:,:1]
    CPs = tf.keras.layers.Activation('sigmoid', name='CPs')(CPs)
    K_and_alpha = x[:,:,1:]
    K_and_alpha = tf.keras.layers.Activation('linear', name='K_and_alpha')(K_and_alpha)
    
    model = tf.keras.models.Model(inputs, [CPs, K_and_alpha])
    return model