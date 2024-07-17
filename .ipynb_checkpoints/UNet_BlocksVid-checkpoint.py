import tensorflow as tf
import numpy as np

def VidConvBlockSimple(x, Params):
    '''
    ConvBlockSimple for videos! Uses 3D convolutions
    '''
    num_filters = Params['num_filters']
    kernel_size = Params['kernel_size']
    strides = Params['strides']
    padding = Params['padding']
    
    x = tf.keras.layers.Conv3D(num_filters, kernel_size, strides=strides, padding=padding, activation="relu")(x)
    return x

def FlattenPhotoChannels(x, Params):
    '''
    Converts a photo (a frame video) into desired number of channels
    '''
    num_filters = Params['num_filters']
    image_h_w = np.shape(x)[2]
    kernel_size = (1,image_h_w,image_h_w)
    
    x = tf.keras.layers.Conv3D(num_filters, kernel_size)(x)
    x = tf.squeeze(x, axis=(2,3))
    return x

def FlattenVideoChannels(x):
    '''
    Reduce the number of channels in some video data to 1.
    '''
    x = tf.keras.layers.Conv1D(1, 1)(x)
    return x

def VideoToTimeseries_A(x, Params):
    '''
    Go right from a video to a timeseries
    '''
    x = FlattenVideoChannels(x)
    x = FlattenPhotoChannels(x, Params)
    return x

def VideoToTimeseries_B(x, Params):
    '''
    An alternate way to go from video to timeseries
    '''
    num_filters = Params['num_filters']
    image_h_w = np.shape(x)[2]
    kernel_size = (1,image_h_w,image_h_w)
    
    x = tf.keras.layers.Conv3D(num_filters, kernel_size)(x)
    x = tf.squeeze(x, axis=(2,3))
    
    return x