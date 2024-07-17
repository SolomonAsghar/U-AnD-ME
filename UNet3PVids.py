import tensorflow as tf

def VideoUNet3P_ConvertAtOutput(filters, VidConvBlock, BlockParams, VideoToTimeseries, SkipBlockParams, DecoderBlockParams): 
    '''
    Create a U-Net 3+ network.
    '''
    
    input_shape = (200, 128, 128, 1)    # must have 1 at the end as video data exepcts num channels to be explicit 
    U_depth = len(filters)

    # Set up the encoder and populate it with the input
    input_layer = tf.keras.layers.Input(input_shape)
    Encoder = [input_layer]

    # Set up the decoder
    Decoder = [[] for level in range(U_depth)]

    # Create the encoder and pass on any required skip connections
    for a in range(U_depth):
        # Apply Convolutional block for this encoder layer
        BlockParams['num_filters']  = filters[a]
        Encoder[-1] = (VidConvBlock(Encoder[-1], BlockParams))

        # For all but the bridge...
        if a+1 < U_depth:
            # Pass on skip connections
            for b in range(U_depth-1)[a:]:
                skip_connection = Encoder[-1]
                pool_size = 2**(b-a)
                if pool_size > 1:    # 1x1 maxpool is identity
                    pool_size_3d = [pool_size]*3
                    skip_connection = tf.keras.layers.MaxPool3D(pool_size_3d,pool_size_3d)(skip_connection)    # downsample
                skip_connection = VidConvBlock(skip_connection, SkipBlockParams)
                Decoder[b].append(skip_connection)

            # Downsample
            Encoder.append(tf.keras.layers.MaxPool3D((2,2,2),(2,2,2))(Encoder[-1]))

    # Bridge the encoder and decoder        
    Decoder[-1] = Encoder[-1]

    # Create the decoder, starting from the deepest layer
    decoder_range = range(U_depth)[::-1]    # flip the iterator, build from bottom up
    for b in decoder_range:
        # For all the but the bridge...
        if b+1 < U_depth:
            Decoder[b] = tf.keras.layers.Concatenate()(Decoder[b])
            Decoder[b] = VidConvBlock(Decoder[b], DecoderBlockParams)
            Decoder[b] = tf.keras.layers.BatchNormalization()(Decoder[b])
            Decoder[b] = tf.keras.layers.ReLU()(Decoder[b])

        # Upsample to higher decoder levels, if there are any
        if b > 0:
            for b_2 in decoder_range[-b:]:
                upscaling_factor = 2**(b-b_2)
                kernel_size = stride = [upscaling_factor]*3    # This is the upscaling factor
                upsampled = tf.keras.layers.Conv3DTranspose(SkipBlockParams['num_filters'], 
                                                            kernel_size, 
                                                            stride, 
                                                            SkipBlockParams['padding'])(Decoder[b])
                upsampled = VidConvBlock(upsampled, SkipBlockParams)
                Decoder[b_2].append(upsampled)

    x = VideoToTimeseries(Decoder[b], {'num_filters': 6})
    
    CPs = x[:,:,:1]
    CPs = tf.keras.layers.Activation('sigmoid', name='CPs')(CPs)
    x_y_K_alpha_class = x[:,:,1:]
    x_y_K_alpha_class = tf.keras.layers.Activation('linear', name='x_y_K_alpha_class')(x_y_K_alpha_class)

    model = tf.keras.models.Model(input_layer, [CPs, x_y_K_alpha_class])
    
    return model


def VideoUNet3P_ConvertAtSkipAndBridge(filters, VidConvBlock, ConvBlock, BlockParams, VideoToTimeseries, SkipBlockParams, DecoderBlockParams): 
    '''
    Create a U-Net 3+ network.
    '''

    input_shape = (200, 128, 128, 1)    # must have 1 at the end as video data exepcts num channels to be explicit 
    U_depth = len(filters)

    # Set up the encoder and populate it with the input
    input_layer = tf.keras.layers.Input(input_shape)
    Encoder = [input_layer]

    # Set up the decoder
    Decoder = [[] for level in range(U_depth)]

    # Create the encoder and pass on any required skip connections
    for a in range(U_depth):
        # Apply Convolutional block for this encoder layer
        BlockParams['num_filters']  = filters[a]
        Encoder[-1] = (VidConvBlock(Encoder[-1], BlockParams))

        # For all but the bridge...
        if a+1 < U_depth:
            # Pass on skip connections
            for b in range(U_depth-1)[a:]:
                skip_connection = Encoder[-1]
                pool_size = 2**(b-a)
                if pool_size > 1:    # 1x1 maxpool is identity
                    pool_size_3d = [pool_size]*3
                    skip_connection = tf.keras.layers.MaxPool3D(pool_size_3d,pool_size_3d)(skip_connection)    # downsample
                skip_connection = VidConvBlock(skip_connection, SkipBlockParams)
                Decoder[b].append(VideoToTimeseries(skip_connection, {'num_filters': filters[a]}))

            # Downsample
            Encoder.append(tf.keras.layers.MaxPool3D((2,2,2),(2,2,2))(Encoder[-1]))

    # Bridge the encoder and decoder        
    Decoder[-1] = Encoder[-1]
    Decoder[-1] = VideoToTimeseries(Decoder[-1], {'num_filters': filters[a]})

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
    x = tf.keras.layers.Conv1D(6, 1, activation='linear')(Decoder[b])

    CPs = x[:,:,:1]
    CPs = tf.keras.layers.Activation('sigmoid', name='CPs')(CPs)
    x_y_K_alpha_class = x[:,:,1:]
    x_y_K_alpha_class = tf.keras.layers.Activation('linear', name='x_y_K_alpha_class')(x_y_K_alpha_class)

    model = tf.keras.models.Model(input_layer, [CPs, x_y_K_alpha_class])

    return model