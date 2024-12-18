import tensorflow as tf
import sys
from UNet_Default_Block_Params import *

def ConvBlockSimple(x, Params):
    '''
    Simple conv block, consists of two 1D convolutions.
    '''
    num_filters = Params['num_filters']
    kernel_size = Params['kernel_size']
    strides = Params['strides']
    padding = Params['padding']
    
    x = tf.keras.layers.Conv1D(num_filters, kernel_size, strides=strides, padding=padding, activation="relu")(x)
    return x

def ConvBlock(x, Params):
    '''
    Simple conv block, consists of two 1D convolutions.
    '''
    num_filters = Params['num_filters']
    kernel_size = Params['kernel_size']
    strides = Params['strides']
    padding = Params['padding']
    
    x = tf.keras.layers.Conv1D(num_filters, kernel_size, strides=strides, padding=padding, activation="relu")(x)
    x = tf.keras.layers.Conv1D(num_filters, kernel_size, strides=strides, padding=padding, activation="relu")(x)
    return x


def WaveBlock(input_layer, Params):
    '''
    Based on WaveNet.
    Same padding instead of causal padding - i.e. sees forward and backwards in time
    
    num_filters (int): Desired number of filters for convolutions 
    kernel_size (int): Desired kernel size for convolutions
    num_dilated_conv_layers (int): Desired number of layers for dilated convolutions
    batch_norm (bool): Whether or not we want a batch norm to be applied
    '''
    num_filters = Params['num_filters']
    kernel_size = Params['kernel_size']
    num_dilated_conv_layers = Params['num_dilated_conv_layers']
    batch_norm = Params['batch_norm']
    
    def dilated_conv_layer(dilation_input, dilation):
        filter_conv = tf.keras.layers.Conv1D(num_filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation, kernel_initializer=tf.keras.initializers.GlorotNormal(), activation='tanh')(dilation_input)
        gate_conv = tf.keras.layers.Conv1D(num_filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation, kernel_initializer=tf.keras.initializers.GlorotNormal(), activation='sigmoid')(dilation_input)

        gated_activation = tf.keras.layers.multiply([filter_conv, gate_conv])

        skip_out = tf.keras.layers.Conv1D(num_filters, kernel_size=1, kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(gated_activation)
        layer_out = tf.keras.layers.Conv1D(num_filters, kernel_size=1, kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(gated_activation)
        if batch_norm is True:
            layer_out = tf.keras.layers.BatchNormalization()(layer_out)

        layer_out = tf.keras.layers.Add()([layer_out, dilation_input])      # residual connection  

        return layer_out, skip_out

    conv = tf.keras.layers.Conv1D(num_filters, kernel_size=kernel_size, padding='same',  kernel_initializer=tf.keras.initializers.HeNormal())(input_layer)
    if batch_norm is True:
            conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.ReLU()(conv)

    dilation_input = conv
    skip_connections = []
    for layer in range(num_dilated_conv_layers):
        dilation_input, skip_out = dilated_conv_layer(dilation_input, 2**(layer+1))
        skip_connections.append(skip_out)

    sum_skip = tf.keras.layers.Add()(skip_connections)
    relu_sum_skip = tf.keras.layers.ReLU()(sum_skip)
    conv_1x1 = tf.keras.layers.Conv1D(num_filters, kernel_size=1, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(relu_sum_skip)
    if batch_norm is True:
            conv_1x1 = tf.keras.layers.BatchNormalization()(conv_1x1)
    conv_1x1 = tf.keras.layers.ReLU()(conv_1x1)

    return conv_1x1


def Encoder(x, ConvBlock, BlockParams, DownSamplingParams=DownSamplingParams):
    s = ConvBlock(x, BlockParams)
    x = tf.keras.layers.MaxPool1D(DownSamplingParams['pool_size'], DownSamplingParams['strides'])(s)
    return s, x


def Decoder(x, skip, ConvBlock, BlockParams, UpSamplingParams):
    x = tf.keras.layers.Conv1DTranspose(UpSamplingParams['num_filters'], UpSamplingParams['kernel_size'], UpSamplingParams['strides'], UpSamplingParams['padding'])(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    
    x = ConvBlock(x, BlockParams)
    return x
