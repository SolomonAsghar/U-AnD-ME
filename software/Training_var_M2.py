import numpy as np
import tensorflow as tf
import sys
from DataGen_var_M2 import *
from Evaluation import *

def UnlimitedTrain_M(model, checkpoint_path, data_generator=GenerateTrueData_M, data_post_process=None, patience=3, data_gen_per_iteration=10000, prop_train=0.8, verbose=False, max_traj_len=200):
    '''
    Use the function data_generator to keep generating new data and training network.
    '''
    min_val_loss = np.inf
    p = patience
    while p > 0:
        # Generate new trajectories and labels
        Trajs, Labels = data_generator(N=data_gen_per_iteration, max_traj_len=max_traj_len)
        Trajs = data_post_process(Trajs)
        Labels = ExplicitCPs(Labels)
        if verbose:
            print("New data generated.")
        
        Trajs_trn = Trajs[:int(len(Trajs)*prop_train)]
        Labels_trn = Labels[:int(len(Labels)*prop_train)]
        Labels_trn_CPs = Labels_trn[:,:,:1]
        Labels_trn_K_and_alpha_and_class = Labels_trn[:,:,1:4]
        Labels_trn_M = Labels_trn[:,:,4:]

        Trajs_val = Trajs[int(len(Trajs)*prop_train):]
        Labels_val = Labels[int(len(Labels)*prop_train):]
        Labels_val_CPs = Labels_val[:,:,:1]
        Labels_val_K_and_alpha_and_class = Labels_val[:,:,1:4]
        Labels_val_M = Labels_val[:,:,4:]
        
        # Use generated data until we overtraining
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1,
                                                          restore_best_weights=True)
        model.fit(x=Trajs_trn,
                  y={"CPs": Labels_trn_CPs,
                     "K_and_alpha_and_class": Labels_trn_K_and_alpha_and_class,
                     "Model": Labels_trn_M},
                  validation_data=(Trajs_val, {"CPs": Labels_val_CPs,
                                               "K_and_alpha_and_class": Labels_val_K_and_alpha_and_class,
                                               "Model": Labels_val_M}),
                  epochs=500,
                  callbacks=[early_stopping],
                  batch_size=512)

        # Store the validation loss with networks current state
        val_loss = model.evaluate(x=Trajs_val,
                                  y={"CPs": Labels_val_CPs, 
                                     "K_and_alpha_and_class": Labels_val_K_and_alpha_and_class,
                                     "Model": Labels_val_M},
                                  batch_size=512)[0]
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
