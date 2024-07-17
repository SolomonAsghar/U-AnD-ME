import tensorflow as tf
import sys
sys.path.append(r'/home/cs-solomon.asghar/AnDi_2024/software/')
from UNet_Default_Block_Params import *
from UNet_Blocks import *

def UNetPP(filters, ConvBlock, BlockParams):
    input_len = 200
    U_depth = len(filters)
    
    # Set up nodes and store 0,0 node as input
    Nodes = [[[] for b in range(U_depth)[:U_depth-a]] for a in range(U_depth)]
    input_layer = tf.keras.layers.Input((input_len, 2))
    Nodes[0][0] = [input_layer] 

    # Create the nested structure
    for a in range(U_depth):
        for b in range(U_depth-a):
            BlockParams['num_filters'] = filters[b]

            # Concatenate 
            Nodes[a][b] = tf.keras.layers.Concatenate(name=f'concat_{a}{b}')(Nodes[a][b])

            # Apply Convutation block
            Nodes[a][b] = ConvBlock(Nodes[a][b], BlockParams)

            # Pass on any skip connections
            if a+b < U_depth-1:
                for a_i in range(a+1,U_depth-b):
                        Nodes[a_i][b].append(Nodes[a][b])

            # Downsample if needed
            if a == 0 and b < U_depth-1:
                Nodes[a][b+1].append(tf.keras.layers.MaxPool1D(2,2)(Nodes[a][b]))

            # Upsample if needed
            if b > 0:
                Nodes[a+1][b-1].append(tf.keras.layers.Conv1DTranspose(filters[b-1], 2, 2, 'same')(Nodes[a][b]))

    # Do final transform to get desired output width
    x = tf.keras.layers.Conv1D(4, 1, activation='linear')(Nodes[a][b])

    CPs = x[:,:,:1]
    CPs = tf.keras.layers.Activation('sigmoid', name='CPs')(CPs)
    K_and_alpha_and_class = x[:,:,1:]
    K_and_alpha_and_class = tf.keras.layers.Activation('linear', name='K_and_alpha_and_class')(K_and_alpha_and_class)

    model = tf.keras.models.Model(input_layer, [CPs, K_and_alpha_and_class])
    return model