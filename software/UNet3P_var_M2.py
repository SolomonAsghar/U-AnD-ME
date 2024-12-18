import tensorflow as tf
import sys
from UNet_Default_Block_Params import *
from UNet_Blocks import *

def UNet3P_var_M(filters, ConvBlock, BlockParams, SkipBlockParams, DecoderBlockParams, input_len=200): 
    '''
    Create a U-Net 3+ network.
    '''
    U_depth = len(filters)

    # Set up the encoder and populate it with the input
    input_layer = tf.keras.layers.Input((input_len, 2))
    Encoder = [input_layer]

    # Set up the decoder
    Decoder = [[] for level in range(U_depth)]

    # Create the encoder and pass on any required skip connections
    for a in range(U_depth):
        # Apply Convolutional block for this encoder layer
        BlockParams['num_filters']  = filters[a]
        Encoder[-1] = (ConvBlock(Encoder[-1], BlockParams))

        # For all but the bridge...
        if a+1 < U_depth:
            # Pass on skip connections
            for b in range(U_depth-1)[a:]:
                skip_connection = Encoder[-1]
                pool_size = 2**(b-a)
                if pool_size > 1:    # 1x1 maxpool is identity
                    skip_connection = tf.keras.layers.MaxPool1D(pool_size,pool_size)(skip_connection)    # downsample
                skip_connection = ConvBlock(skip_connection, SkipBlockParams)
                Decoder[b].append(skip_connection)

            # Downsample
            Encoder.append(tf.keras.layers.MaxPool1D(2,2)(Encoder[-1]))

    # Bridge the encoder and decoder        
    Decoder[-1] = Encoder[-1]

    # Create the decoder, starting from the deepest layer
    decoder_range = range(U_depth)[::-1]    # flip the iterator, build from bottom up
    for b in decoder_range:
        # For all the but the bridge...
        if b+1 < U_depth:
            Decoder[b] = tf.keras.layers.Concatenate()(Decoder[b])
            Decoder[b] = ConvBlock(Decoder[b], DecoderBlockParams)
            Decoder[b] = tf.keras.layers.BatchNormalization()(Decoder[b])
            Decoder[b] = tf.keras.layers.ReLU()(Decoder[b])

        # Upsample to higher decoder levels, if there are any
        if b > 0:
            for b_2 in decoder_range[-b:]:
                kernel_size = stride = 2**(b-b_2)    # This is the upscaling factor
                upsampled = tf.keras.layers.Conv1DTranspose(SkipBlockParams['num_filters'], 
                                                            kernel_size, 
                                                            stride, 
                                                            SkipBlockParams['padding'])(Decoder[b])
                upsampled = ConvBlock(upsampled, SkipBlockParams)
                Decoder[b_2].append(upsampled)

    ## FinalTransform
    x = tf.keras.layers.Conv1D(9, 1, activation='linear')(Decoder[b])

    CPs = x[:,:,:1]
    CPs = tf.keras.layers.Activation('sigmoid', name='CPs')(CPs)
    K_and_alpha_and_class = x[:,:,1:4]
    K_and_alpha_and_class = tf.keras.layers.Activation('linear', name='K_and_alpha_and_class')(K_and_alpha_and_class)
    
    # Predict the diffusion model!
    Model = x[:,:,4:]
    Model =  tf.keras.layers.Activation('softmax', name='Model')(Model)
    
    model = tf.keras.models.Model(input_layer, [CPs, K_and_alpha_and_class, Model])
    return model
