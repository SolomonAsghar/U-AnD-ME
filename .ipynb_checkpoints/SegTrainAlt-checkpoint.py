import numpy as np
import tensorflow as tf
from scipy import signal

from andi_datasets.models_phenom import models_phenom
from andi_datasets.datasets_challenge import challenge_phenom_dataset
import sys
sys.path.append(r'/home/cs-solomon.asghar/AnDi_2024/software/')
from DataGen import *

def LabelToCP(CP_labels, from_network=True):
    '''
    Convert from labels to change points
    '''
    # Get the locations of each changepoint 
    if from_network:
        CPs = signal.find_peaks(CP_labels, 0.25)[0]
    else:
        CPs = np.nonzero(CP_labels)[0]
    CPs = CPs[CPs > 2]    # Ignore CPs on first few time-step, min seg lenght is 3 anyways!
    CPs = CPs[(len(CP_labels) - CPs) > 2]    # Ignore CPs on final few time-step
    
    return CPs


def ExplicitCPs_SingleTraj(Labels):
    '''
    Accept labels, return expliticly labelled CPs + labels.
    Single traj version
    '''
    # Work out CP locations
    labs_diff = Labels[1:] - Labels[:-1]
    labs_diff_sum = np.sum(abs(labs_diff), axis=1)
    CPs = np.where(labs_diff_sum != 0, 1, 0)
    # Prep to concatenate to labels
    CPs = np.concatenate([[0], CPs], axis=0)
    CPs = np.expand_dims(CPs, 1)
    # Concatenate to labels
    Labels = np.concatenate([CPs, Labels], axis=1)
    return Labels


def GenerateTrueData_NoPadding(N=10000):
    all_trajs = [[]] * N
    all_labels = [[]] * N
    traj_idx = -1

    break_flag = False
    while True:
        ## Generate FOVs, convert to list of numpy arrays
        FOVs, _, _ = challenge_phenom_dataset(dics=Gen_Random_Model_Param_Dict(N=100), return_timestep_labs=True)
        FOVs = [FOV.to_numpy() for FOV in FOVs]

        ## Split FOVs up into different trajs, and add padding as appropriate ##
        for fov_idx, FOV in enumerate(FOVs):
            mean_xy = np.mean(FOV[:, 2:4], axis=0)    # zero mean for this FOV
            FOV[:, 2:4] = FOV[:, 2:4] - mean_xy
            _, first_idx = np.unique(FOV[:,0], return_index=True)
            split_trajs = np.split(FOV, first_idx[1:])    # split into diff trajs
            for traj in split_trajs:
                traj_idx += 1    # Keep track of how many trajs we have
                if traj_idx == N:    # If we have enough trajs, stop storing more
                    break_flag = True
                    break
                all_trajs[traj_idx] = traj[:,2:4]    # drop in the traj
                all_labels[traj_idx] = ExplicitCPs_SingleTraj(traj[:,4:])    # drop in the label
            if break_flag:
                break
        if break_flag:
            break
    
    return all_trajs, all_labels


def GenerateSegmentTrainingData(seg_len, data_gen_target, data_gen_per_iteration, data_generator=GenerateTrueData_NoPadding):
    '''
    Generates training data for the specified segment size.
    '''
    Generated_Segments = np.zeros((data_gen_target, seg_len, 2))
    Generated_Labels = np.zeros((data_gen_target,3))
    data_generated = 0
    
    while data_generated < data_gen_target:
        ## Generate data for this iteration
        Trajs, Labels = data_generator(data_gen_per_iteration)
        
        ## These containers will hold the segments, based on their lenght
        Segments_by_Size = [[]] * (200 - seg_len + 1)
        Labels_by_Segment_Size = [[]] * (200 - seg_len + 1)
        
        ## For each traj, split up the segment 
        for traj_idx, (traj, label) in enumerate(zip(Trajs, Labels)):
            ## Split up CPs and Alpha, K, class
            CP_labels = label[:,0]
            alpha_and_K_and_class_each_timestep = label[:,1:]
        
            ## Get Changepoints ##
            CPs = LabelToCP(CP_labels)
            ## /Get Changepoints ##
        
            ## Split according to changepoints ##
            traj_segments = np.split(traj, CPs)
            true_label_segments = np.split(label, CPs)
        
            for traj_seg, lab_seg in zip(traj_segments, true_label_segments):
                seg_idx = len(traj_seg) - (seg_len)
                if seg_idx >= 0:    # if this segment is at least as long as the desired lenght, save it
                    Segments_by_Size[seg_idx] = Segments_by_Size[seg_idx] + [traj_seg]
                    Labels_by_Segment_Size[seg_idx] = Labels_by_Segment_Size[seg_idx] + [lab_seg[0,1:]]
            ## /Split according to changepoitns ##
        ## /For each traj, split up the segment 
        
        ## Save all the trajs to the numpy array
        if len(Segments_by_Size[0]) > 0:
            old_data_generated = data_generated 
            data_generated = data_generated + len(Segments_by_Size[0])
            Generated_Segments[old_data_generated:data_generated] = Segments_by_Size[0][:data_gen_target - old_data_generated]
            Generated_Labels[old_data_generated:data_generated] = Labels_by_Segment_Size[0][:data_gen_target - old_data_generated]
        
        Segments_by_Size = Segments_by_Size[1:]
        Labels_by_Segment_Size = Labels_by_Segment_Size[1:]
        
        ## Fill from longer segments, if available
        if seg_len < 200:
            for seg_by_size, lab_by_size in zip(Segments_by_Size, Labels_by_Segment_Size):    # for each longer segment group
                if data_generated < data_gen_target:    # if we still need segments
                    segments_available = len(seg_by_size)    # work out how many we can take
                    seg_required = data_gen_target - data_generated
                    seg_to_take = min(segments_available, seg_required)
        
                    if seg_to_take:    # if we have segments to take
                        current_segment_lengths = np.shape(seg_by_size)[1]
                        max_proposed_start = current_segment_lengths - seg_len
                        proposed_random_starts = np.random.randint(0, max_proposed_start+1, seg_to_take)
                        
                        for rand_start_idx, (seg_b, lab_b) in enumerate(zip(seg_by_size[:seg_to_take], lab_by_size[:seg_to_take])):
                            # Create the new trajectory
                            start_idx = proposed_random_starts[rand_start_idx]
                            end_idx = start_idx + seg_len
                            seg_b = seg_b[start_idx:end_idx]
                
                            # Append to list of existing trajs
                            Generated_Segments[data_generated] = seg_b
                            Generated_Labels[data_generated] = lab_b
                            data_generated += 1

    Generated_Segments = np.diff(Generated_Segments, axis=1)
    
    return Generated_Segments, Generated_Labels


def UnlimitedTrainSegmentModels(model, seg_len, checkpoint_file_path, 
                                data_generator, data_per_iteration=1000, 
                                max_patience=3, prop_train=0.8, verbose=False):
    '''
    Use the function data_generator to keep generating new data and the desired segment model. 
    '''
    min_val_loss = np.inf
    patience = max_patience 
    
    while patience > 0:    # while there's patience left
        # Prepare training data
        Generated_Segments, Generated_Labels = data_generator(seg_len, data_per_iteration, data_per_iteration*2)
        data_idxs = np.arange(data_per_iteration)
        np.random.shuffle(data_idxs)
        trn_idxs = data_idxs[:int(data_per_iteration*prop_train)]
        val_idxs = data_idxs[int(data_per_iteration*prop_train):]
        Data_trn = Generated_Segments[trn_idxs]
        Labels_trn = Generated_Labels[trn_idxs]
        Data_val = Generated_Segments[val_idxs]
        Labels_val = Generated_Labels[val_idxs]
        
        # Train the network until training stagnates
        print('...training')
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1,
                                                          restore_best_weights=False)
        model.fit(x=Data_trn,
                  y=Labels_trn,
                  validation_data=(Data_val,Labels_val),
                  epochs=500,
                  callbacks=[early_stopping],
                  batch_size=512,
                  verbose=False)
        
        # Store the validation loss with the networks current state
        val_loss = model.evaluate(x=Data_val, y=Labels_val, batch_size=512)[0]
        
        if val_loss < min_val_loss:
            patience = max_patience
            min_val_loss = val_loss
            model.save_weights(checkpoint_file_path)
            
        else:
            patience -= 1
            if verbose:
                print(f"Network not learning, patience at {p}.")
        
        print(f'Finished with p={patience}')            
        print('------------------------------')