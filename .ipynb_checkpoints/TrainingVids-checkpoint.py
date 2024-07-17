import numpy as np
import tensorflow as tf
import sys
sys.path.append('/home/cs-solomon.asghar/AnDi_2024/software/')
from DataGenVid import GenerateVideoData


def ExplicitCPsVids(Labels):
    '''
    Accept labels, return expliticly labelled CPs + labels. For video format labels.
    '''
    # Work out CP locations
    labs_diff = Labels[:,1:,2:] - Labels[:,:-1,2:]
    labs_diff_sum = np.sum(abs(labs_diff), axis=2)
    CPs = np.where(labs_diff_sum != 0, 1, 0)
    # Prep to concatenate to labels
    zeros = np.zeros((np.shape(CPs)[0], 1))
    CPs = np.concatenate([zeros, CPs], axis=1)
    CPs = np.expand_dims(CPs, 2)
    # Concatenate to labels
    Labels = np.concatenate([CPs, Labels], axis=2)
    return Labels


def UnlimitedTrainVid(model, checkpoint_path, data_generator=GenerateVideoData, patience=3, data_gen_per_iteration=10000, prop_train=0.8, verbose=False):
    '''
    Use the function data_generator to keep generating new data and training network, for video data and networks.
    '''
    min_val_loss = np.inf
    p = patience
    while p > 0:
        # Generate new trajectories and labels
        Videos, Labels = data_generator(N=data_gen_per_iteration)
        Labels = ExplicitCPsVids(Labels)
        if verbose:
            print("New data generated.")
        
        Videos_trn = Videos[:int(len(Videos)*prop_train)]
        Labels_trn = Labels[:int(len(Labels)*prop_train)]
        Labels_trn_CPs = Labels_trn[:,:,:1]
        Labels_trn_pos_K_and_alpha_and_class = Labels_trn[:,:,1:]

        Videos_val = Videos[int(len(Videos)*prop_train):]
        Labels_val = Labels[int(len(Labels)*prop_train):]
        Labels_val_CPs = Labels_val[:,:,:1]
        Labels_val_pos_K_and_alpha_and_class = Labels_val[:,:,1:]
        
        # Use generated data until we overtraining
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1,
                                                          restore_best_weights=True)
        model.fit(x=Videos_trn,
                  y={"CPs": Labels_trn_CPs,
                     "x_y_K_alpha_class": Labels_trn_pos_K_and_alpha_and_class},
                  validation_data=(Videos_val, {"CPs": Labels_val_CPs,
                                               "x_y_K_alpha_class": Labels_val_pos_K_and_alpha_and_class}),
                  epochs=500,
                  callbacks=[early_stopping],
                  batch_size=32)

        # Store the validation loss with networks current state
        val_loss = model.evaluate(x=Videos_val, y={"CPs": Labels_val_CPs, "x_y_K_alpha_class": Labels_val_pos_K_and_alpha_and_class}, batch_size=16)[0]
        if verbose:
            print("New val loss:", val_loss)
        
        if val_loss < min_val_loss:
            p = patience
            min_val_loss = val_loss
            model.save_weights(checkpoint_path)
            if verbose:
                print("Network still learning.")
        else:
            p -= 1
            if verbose:
                print(f"Network not learning, patience at {p}.")
    
    model.load_weights(checkpoint_path)
    if verbose:
        print("Patience ran out. Best weights restored")
    return model  