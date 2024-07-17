import numpy as np
import tensorflow as tf
import sys
sys.path.append(r'/home/cs-solomon.asghar/AnDi_2024/software/')
from DataGen import *
from Evaluation import *

def UnlimitedTrain(model, checkpoint_path, data_generator=GenerateTrueData, patience=3, data_gen_per_iteration=10000, prop_train=0.8, verbose=False):
    '''
    Use the function data_generator to keep generating new data and training network.
    '''
    min_val_loss = np.inf
    p = patience
    while p > 0:
        # Generate new trajectories and labels
        Trajs, Labels = data_generator(N=data_gen_per_iteration)
        Labels = ExplicitCPs(Labels)
        if verbose:
            print("New data generated.")
        
        Trajs_trn = Trajs[:int(len(Trajs)*prop_train)]
        Labels_trn = Labels[:int(len(Labels)*prop_train)]
        Labels_trn_CPs = Labels_trn[:,:,:1]
        Labels_trn_K_and_alpha_and_class = Labels_trn[:,:,1:]

        Trajs_val = Trajs[int(len(Trajs)*prop_train):]
        Labels_val = Labels[int(len(Labels)*prop_train):]
        Labels_val_CPs = Labels_val[:,:,:1]
        Labels_val_K_and_alpha_and_class = Labels_val[:,:,1:]
        
        # Use generated data until we overtraining
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1,
                                                          restore_best_weights=True)
        model.fit(x=Trajs_trn,
                  y={"CPs": Labels_trn_CPs,
                     "K_and_alpha_and_class": Labels_trn_K_and_alpha_and_class},
                  validation_data=(Trajs_val, {"CPs": Labels_val_CPs,
                                               "K_and_alpha_and_class": Labels_val_K_and_alpha_and_class}),
                  epochs=500,
                  callbacks=[early_stopping],
                  batch_size=512)

        # Store the validation loss with networks current state
        val_loss = model.evaluate(x=Trajs_val, y={"CPs": Labels_val_CPs, "K_and_alpha_and_class": Labels_val_K_and_alpha_and_class}, batch_size=512)[0]
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


def TrainAndEvaluate(model, data_generator, checkpoint_path, patience=3, data_gen_per_iteration=100000, prop_train=0.8, verbose=False, num_testing_samples=100000):
    '''
    Take a model, compile and train it, and then evaluate its performance. 
    '''
    # Compile model
    losses = {"CPs": "binary_crossentropy",
              "K_and_alpha": "MSE"}
    model.compile(optimizer="adam", loss=losses, metrics=["acc"])
    
    # Train
    model = UnlimitedTrain(model, data_generator, checkpoint_path, patience=patience, data_gen_per_iteration=data_gen_per_iteration, verbose=verbose)
    
    # Evaluate
    Trajs, Target_Labels = GenerateMSMData(N=num_testing_samples)
    Predicted_Labels = np.concatenate(model.predict(Trajs), axis=2)
    
    CP_MSE, dM_MSE, alpha_MSE, K_MSE = EvaluateLabels(Predicted_Labels, Target_Labels)    
    if verbose:
        print(f"Change point MSE: {CP_MSE}")
        print(f"Num seg. MSE:     {dM_MSE}")
        print(f"Alpha MSE:        {alpha_MSE}")
        print(f"K MSE:            {K_MSE}")
    return CP_MSE, dM_MSE, alpha_MSE, K_MSE