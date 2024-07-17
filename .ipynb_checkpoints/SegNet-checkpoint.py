import numpy as np
import tensorflow as tf

def BuildSegModel(seg_length, input_width=5):
    '''
    Build a WaveNet appropriate for the input segment lenght
    '''
    WaveParams = {'num_filters': 64,
                 'kernel_size': 3,
                 'num_dilated_conv_layers': int(np.log2(seg_length)-1),
                 'batch_norm': True}

    Output_Params = {'num_layers': 2,
                     'num_neurons': 64,
                     'num_outputs': 3}

    Inputs = tf.keras.Input(shape=(seg_length,input_width))
    L = WaveNet(WaveParams)(Inputs)
    L = tf.keras.layers.Flatten()(L)
    L = Output(Output_Params)(L)

    model = tf.keras.Model(inputs=Inputs, outputs=L)
    return model


class WaveNet():
    '''
    Based on WaveNet.
    Same padding instead of causal padding - i.e. sees forward and backwards in time
    
    dim (int): Dimensionality of input 

    WaveParams (dict):
        num_filters (int): Desired number of filters for convolutions 
        kernel_size (int): Desired kernel size for convolutions
        num_dilated_conv_layers (int): Desired number of layers for dilated convolutions
    '''
    def __init__(self, WaveParams):
        self.num_filters = WaveParams['num_filters']
        self.kernel_size = WaveParams['kernel_size']
        self.num_dilated_conv_layers = WaveParams['num_dilated_conv_layers']
        self.batch_norm = WaveParams['batch_norm']

    def __call__(self, input_layer):
        def dilated_conv_layer(dilation_input, dilation):
            filter_conv = tf.keras.layers.Conv1D(self.num_filters, kernel_size=self.kernel_size, padding='same', dilation_rate=dilation, kernel_initializer=tf.keras.initializers.GlorotNormal(), activation='tanh')(dilation_input)
            gate_conv = tf.keras.layers.Conv1D(self.num_filters, kernel_size=self.kernel_size, padding='same', dilation_rate=dilation, kernel_initializer=tf.keras.initializers.GlorotNormal(), activation='sigmoid')(dilation_input)

            gated_activation = tf.keras.layers.multiply([filter_conv, gate_conv])

            skip_out = tf.keras.layers.Conv1D(self.num_filters, kernel_size=1, kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(gated_activation)
            layer_out = tf.keras.layers.Conv1D(self.num_filters, kernel_size=1, kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(gated_activation)
            if self.batch_norm is True:
                layer_out = tf.keras.layers.BatchNormalization()(layer_out)

            layer_out = tf.keras.layers.Add()([layer_out, dilation_input])      # residual connection  

            return layer_out, skip_out

        conv = tf.keras.layers.Conv1D(self.num_filters, kernel_size=self.kernel_size, padding='same',  kernel_initializer=tf.keras.initializers.HeNormal())(input_layer)
        if self.batch_norm is True:
                conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.ReLU()(conv)

        dilation_input = conv
        skip_connections = []
        for layer in range(self.num_dilated_conv_layers):
            dilation_input, skip_out = dilated_conv_layer(dilation_input, 2**(layer+1))
            skip_connections.append(skip_out)

        sum_skip = tf.keras.layers.Add()(skip_connections)
        relu_sum_skip = tf.keras.layers.ReLU()(sum_skip)
        conv_1x1 = tf.keras.layers.Conv1D(self.num_filters, kernel_size=1, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(relu_sum_skip)
        if self.batch_norm is True:
                conv_1x1 = tf.keras.layers.BatchNormalization()(conv_1x1)
        conv_1x1 = tf.keras.layers.ReLU()(conv_1x1)
        
        return conv_1x1
    
    
class Output():
    '''
    A simple MLP for final processing before output.

    Output_Params (dict):
        num_layers (int): The number of feedforward layers, discounting the final output layer
        num_neurons (int): Number of neurons per layer
        num_outputs (int): Number of outputs
    '''
    def __init__(self, Output_Params):
        self.num_layers = Output_Params['num_layers']
        if self.num_layers > 0:
            self.num_neurons = Output_Params['num_neurons']
        self.num_outputs = Output_Params['num_outputs']
    
    def __call__(self, L):
        for layer in range(self.num_layers):
            L = tf.keras.layers.Dense(self.num_neurons, activation='relu')(L)
        
        L = tf.keras.layers.Dense(self.num_outputs)(L)
        
        return L