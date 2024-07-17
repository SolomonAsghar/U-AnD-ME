import os
import numpy as np
from scipy import signal
import tensorflow as tf

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


def modal_characteristics(array):
    "Returns the mode of the input array. Returns a random mode if multiple exist."
    unique_values, unique_value_counts = np.unique(array, return_counts=True, axis=0)    
    max_count = np.max(unique_value_counts)    # find modal value(s)
    max_count_idxs = np.argwhere(unique_value_counts == max_count).flatten()
    if len(max_count_idxs > 1):
        random_max_count_idx = np.random.choice(max_count_idxs)   # if there are multiple, pick a random one
        random_modal_value = unique_values[random_max_count_idx]
        return random_modal_value
    
    return unique_values[max_count_idxs[0]]


def PredictAndSplit_Training(model, padded_trajs, padding_mask, True_Labs):
    '''
    Takes prepared trajectories and makes predictions on them. 
    Splits up the trajs according to CP preds and makes input for next network.
    Returns the split trajectories along with appropriate labels.
    
    model - The network
    padded_trajs - The padded input trajectories 
    Labs - All true segment labels
    padding_mask - A mask showing which parts of the trajectories are just padding
    '''
    ## Make predictions ##
    Pred_Labs = model.predict(padded_trajs)
    Pred_Labs = np.concatenate(Pred_Labs, axis=2)
    ## /Make predictions ##

    ## Split up the segments, keeping track of where each came from ##
    segments = []    # we will collect all the segments across the dataset into this one list
    labels = []
    for traj_idx, (traj, pred_lab, true_lab) in enumerate(zip(padded_trajs, Pred_Labs, True_Labs)):
        ## Undo the padding ##
        traj = traj[padding_mask[traj_idx]]
        pred_lab = pred_lab[padding_mask[traj_idx]]
        true_lab = true_lab[padding_mask[traj_idx]]
        ## /Undo the padding ##

        CP_labels = pred_lab[:,0]
        alpha_and_K_and_class_each_timestep = pred_lab[:,1:]

        ## Get Changepoints ##
        CPs = LabelToCP(CP_labels)
        ## /Get Changepoints ##

        ## Split according to changepoitns ##
        traj_segments = np.split(traj, CPs)
        pred_label_segments = np.split(alpha_and_K_and_class_each_timestep, CPs)
        true_label_segments = np.split(true_lab, CPs)

        for seg_idx, (traj_seg, pred_lab_seg, true_lab_seg) in enumerate(zip(traj_segments, pred_label_segments, true_label_segments)):
            full_seg = np.concatenate([traj_seg, pred_lab_seg], axis=1)
            segments += [full_seg]
            labels += [true_lab_seg]
        ## /Split according to changepoitns ##
    ## /Split up the segments, keeping track of where each came from ##

    ## Collect segments based on their lenght
    Segments_by_Size = [[]] * 200
    Labels_by_Segment_Size = [[]] * 200
    for segment, label in zip(segments, labels):
            seg_idx = len(segment) - 1
            Segments_by_Size[seg_idx] = Segments_by_Size[seg_idx] + [segment]
            Labels_by_Segment_Size[seg_idx] = Labels_by_Segment_Size[seg_idx] + [label]
    
    return Segments_by_Size, Labels_by_Segment_Size


def CreateSplitTrainingData(model, data_generator, data_gen_target, data_gen_per_iteration):
    '''
    Uses a data generator and CP predicting model to create data for training segment analysis models.
    '''
    Generated_Data_by_Seg = [np.zeros((data_gen_target, seg_len, 5)) for seg_len in range(3,201)]
    Target_Labels_by_Seg = [np.zeros((data_gen_target, 3)) for seg_len in range(3,201)] 
    num_segs_by_seg_len = np.zeros(198, dtype='int')
    
    while np.any(num_segs_by_seg_len < data_gen_target):    # until we have data_gen_target trajs of each lenght
        Trajs, Labels, Padding_Mask = data_generator(N=data_gen_per_iteration, return_padding_mask=True)    # generate new trajs
        Segments_by_Size, Labels_by_Segment_Size = PredictAndSplit_Training(model, Trajs, Padding_Mask, Labels)

        for seg_len_idx, (seg_by_size, lab_by_size) in enumerate(zip(Segments_by_Size[2:], Labels_by_Segment_Size[2:])):
            segments_available = len(seg_by_size)
            seg_required = data_gen_target - num_segs_by_seg_len[seg_len_idx]
            seg_to_take = min(segments_available, seg_required)
            ## Fill in segments from the correct lenght 
            for seg, lab in zip(seg_by_size[:seg_to_take], lab_by_size[:seg_to_take]):
                Generated_Data_by_Seg[seg_len_idx][num_segs_by_seg_len[seg_len_idx]] = seg
                Target_Labels_by_Seg[seg_len_idx][num_segs_by_seg_len[seg_len_idx]] = modal_characteristics(lab)
                num_segs_by_seg_len[seg_len_idx] += 1

            # Create segments from the sizes up
            for seg_len_idx_b, (seg_by_size_b, lab_by_size_b) in enumerate(zip(Segments_by_Size[3+seg_len_idx:], Labels_by_Segment_Size[3+seg_len_idx:])):
                segments_available = len(seg_by_size_b)
                seg_required = data_gen_target - num_segs_by_seg_len[seg_len_idx]
                seg_to_take = min(segments_available, seg_required)

                desired_segment_length = seg_len_idx + 3
                current_segment_lenghts = desired_segment_length + 1 + seg_len_idx_b
                max_proposed_start = current_segment_lenghts-desired_segment_length
                proposed_random_starts = np.random.randint(0, max_proposed_start+1, seg_to_take)

                for rand_start_idx, (seg_b, lab_b) in enumerate(zip(seg_by_size_b[:seg_to_take], lab_by_size_b[:seg_to_take])):
                    # Create the new trajectory
                    start_idx = proposed_random_starts[rand_start_idx]
                    end_idx = start_idx + desired_segment_length
                    seg_b = seg_b[start_idx:end_idx]
                    lab_b = lab_b[start_idx:end_idx]

                    # Append to list of existing trajs
                    Generated_Data_by_Seg[seg_len_idx][num_segs_by_seg_len[seg_len_idx]] = seg_b
                    Target_Labels_by_Seg[seg_len_idx][num_segs_by_seg_len[seg_len_idx]] = modal_characteristics(lab_b)
                    num_segs_by_seg_len[seg_len_idx] += 1
        
    return Generated_Data_by_Seg, Target_Labels_by_Seg


def UnlimitedTrainSegmentModels(SegModelBuilder, checkpoints_folder_path, data_generator, segmentation_model, patience=3, data_per_iteration=1000, prop_train=0.8, verbose=False):
    '''
    Use the function data_generator to keep generating new data and training each of the segment models. 
    '''
    min_seg = 3
    max_seg = 200
    Seg_Sizes = np.arange(min_seg, max_seg + 1)
    Min_Val_Losses = np.full(max_seg - min_seg + 1, np.inf)
    P = np.full(max_seg - min_seg + 1, patience)
    
    while np.any(P > 0):    # while some networks have patience left
        
        Generated_Data_by_Seg, Target_Labels_by_Seg = data_generator[0](segmentation_model, data_generator[1], data_per_iteration, data_per_iteration*5)
        
        Seg_Sizes = Seg_Sizes[P > 0]
        Generated_Data_by_Seg = [b for a, b in zip(P>0, Generated_Data_by_Seg) if a]
        Target_Labels_by_Seg = [b for a, b in zip(P>0, Target_Labels_by_Seg) if a]
        
        for seg_size, data, targets in zip(Seg_Sizes, Generated_Data_by_Seg, Target_Labels_by_Seg):    # for any networks with patience left
            print(f'Current Net: {seg_size}')
            # Spawn and compile the network
            print('...building')
            model = SegModelBuilder(seg_size)
            model.compile(optimizer='adam', loss='MSE', metrics='acc')
            
            # Load in weights if they exist
            checkpoint_file_path = checkpoints_folder_path + f'/Net_{seg_size}/model_checkpoint'
            if os.path.isfile(checkpoint_file_path):
                model.load_weights(checkpoint_file_path)
            
            # Prepare training data
            data_idxs = np.arange(data_per_iteration)
            np.random.shuffle(data_idxs)
            trn_idxs = data_idxs[:int(data_per_iteration*prop_train)]
            val_idxs = data_idxs[int(data_per_iteration*prop_train):]
            Data_trn = data[trn_idxs]
            Labels_trn = targets[trn_idxs]
            Data_val = data[val_idxs]
            Labels_val = targets[val_idxs]
            
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
            
            seg_idx = seg_size - 3
            
            if val_loss < Min_Val_Losses[seg_idx]:
                P[seg_idx] = patience
                Min_Val_Losses[seg_idx] = val_loss
                model.save_weights(checkpoint_file_path)
                
            else:
                P[seg_idx] -= 1
                if verbose:
                    print(f"Network {seg_size} not learning, patience at {p}.")
            
            print(f'Finished with p={P[seg_idx]}')            
            print('------------------------------')