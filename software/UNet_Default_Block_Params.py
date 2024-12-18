WaveBlockParams = {'num_filters': None,
                   'kernel_size': 3,
                   'num_dilated_conv_layers': 3,
                   'batch_norm': True}

ConvBlockParams = {'num_filters': None,
                   'kernel_size' :3,
                   'strides': 1,
                   'padding': "same"}

DownSamplingParams = {'pool_size': 2,
                      'strides': 2}

UpSamplingParams = {'num_filters': None,
                    'kernel_size': 2,
                    'strides': 2,
                    'padding': "same"}

FinalTransformParams = {'num_filters': 1,
                        'kernel_size': 1,
                        'padding': "same",
                        'activation': "linear"}
