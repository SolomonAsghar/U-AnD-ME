import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(r'/home/cs-solomon.asghar/AnDi_2024/software/')
from Evaluation import *
from SegNet import *


def LabelToCP(CP_labels, from_network=True, min_peak_height=0.25):
    '''
    Convert from labels to change points
    '''
    # Get the locations of each changepoint 
    if from_network:
        CPs = signal.find_peaks(CP_labels, min_peak_height, distance=2)[0]
    else:
        CPs = np.nonzero(CP_labels)[0]
    CPs = CPs[CPs > 2]    # Ignore CPs on first few time-step, min seg lenght is 3 anyways!
    CPs = CPs[(len(CP_labels) - CPs) > 2]    # Ignore CPs on final few time-step
    
    return CPs


def UnshuffleSegmentLabels(Segment_Labels_by_Size, Segment_Addresses_by_Size):
    "Uses segment addresses to reshuffle segment labels into thier original order"
    ### Flatten label list and address list ###
    Segment_Labels = []
    for seg_lab_size in Segment_Labels_by_Size:
        for seg_lab in seg_lab_size:
            Segment_Labels = Segment_Labels + [seg_lab]

    Segment_Addresses = []
    for seg_add_size in Segment_Addresses_by_Size:
        for seg_add in seg_add_size:
            Segment_Addresses = Segment_Addresses + [seg_add]
    Segment_Addresses = np.array(Segment_Addresses)
    ### /Flatten label list and address list ###

    ### Collect all trajs from the same trajectory together ###
    maximum_traj_index = np.max(Segment_Addresses[:,0])
    Segment_Labels_True_Order = [[]] * (maximum_traj_index + 1)
    Updated_Segment_Addresses = [[]] * (maximum_traj_index + 1)
    for segment_label, segment_address  in zip(Segment_Labels, Segment_Addresses):
        Segment_Labels_True_Order[segment_address[0]] = Segment_Labels_True_Order[segment_address[0]] + [segment_label]
        Updated_Segment_Addresses[segment_address[0]] = Updated_Segment_Addresses[segment_address[0]] + [segment_address[1]]
    ### /Collect all trajs from the same trajectory together ###

    ### For each traj, get the segments in the correct order ###
    for traj_idx, (traj_segment_labels, traj_segment_addresses) in enumerate(zip(Segment_Labels_True_Order, Updated_Segment_Addresses)):
        true_ordering = np.argsort(traj_segment_addresses)
        Segment_Labels_True_Order[traj_idx] = [traj_segment_labels[true_index] for true_index in true_ordering]
    ### \For each traj, get the segments in the correct order ###
    
    return Segment_Labels_True_Order 


def MakePredictions_withSegNets(model, seg_model_folder_path, data_path, SegModelBuilder, mode='starting_k'):
    '''
    Make AnDi predictions, for Track 2 Single Traj only. 
    '''
    ### Load data and prepare it for network ###
    All_Trajs = []    # stores all the trajs across all exps and fovs!
    All_Padding_Masks = []    # stores all the padding masks across all exps and fovs!
    All_Traj_Addresses = []    # for each traj, stores what exp and fov its from!

    num_exps = len(os.listdir(data_path + '/ref/track_2/'))
    for exp in range(num_exps):
        all_files = os.listdir(data_path + f'/ref/track_2/exp_{exp}/')
        num_fovs = len([fov for fov in all_files if fov.startswith('trajs_fov')])
        for fov in range(num_fovs):
            FOV_df = pd.read_csv(data_path + f'ref/track_2/exp_{exp}/trajs_fov_{fov}.csv')
            FOV = FOV_df.to_numpy()

            num_trajs = int(FOV[-1,0]) + 1
            all_trajs = np.zeros((num_trajs,200,2))   # prepare a container for all the trajs
            traj_idx = -1
            padding_mask = np.full((num_trajs,200), True)    # keeps track of what is padded vs authentic data

            mean_xy = np.mean(FOV[:, 2:4], axis=0)    # zero mean for this FOV
            FOV[:, 2:4] = FOV[:, 2:4] - mean_xy

            _, first_idx = np.unique(FOV[:,0], return_index=True)    # split into diff trajs
            split_trajs = np.split(FOV, first_idx[1:])    

            for traj in split_trajs:
                traj_idx += 1
                first_frame, last_frame = int(traj[0,1]), int(traj[-1,1])

                all_trajs[traj_idx][first_frame:last_frame+1] = traj[:,2:4]    # drop in the traj
                all_trajs[traj_idx][:first_frame] = traj[0,2:4]    # pad the traj!
                all_trajs[traj_idx][last_frame+1:] = traj[-1,2:4]

                padding_mask[traj_idx][:first_frame] = False    # keep track of what values are padding
                padding_mask[traj_idx][last_frame+1:] = False

                All_Traj_Addresses = All_Traj_Addresses + [[exp, fov]]

            All_Trajs = All_Trajs + [all_trajs]
            All_Padding_Masks = All_Padding_Masks + [padding_mask]

    All_Trajs = np.concatenate(All_Trajs, axis=0)        
    All_Padding_Masks = np.concatenate(All_Padding_Masks, axis=0)        
    All_Traj_Addresses = np.array(All_Traj_Addresses)
    ### /Load data and prepare it for network ###

    ### Make predictions ###        
    Segments_by_Size, Segment_Addresses_by_Size, CPs = PredictAndSplit_Inference(model, All_Trajs, All_Padding_Masks, return_CPs=True)

    Segments_by_Size = [sbs for sbs in Segments_by_Size if sbs != []]    # remove empty elements
    Segment_Addresses_by_Size = [sabs for sabs in Segment_Addresses_by_Size if sabs != []]
    Segment_Labels_by_Size = []

    for segments_by_size in tqdm(Segments_by_Size):           
        seg_size = np.shape(segments_by_size)[1]
        ### if the segsize is too small, just use the old scheme i.e. average the values!
        if seg_size < 3:
            Segment_Labels_by_Size = Segment_Labels_by_Size + [np.mean(segments_by_size, axis=1)[:,2:]]
        else:
            seg_model = SegModelBuilder(seg_size)
            seg_checkpoint_path = seg_model_folder_path + f'/Net_{seg_size}/model_checkpoint'
            seg_model.load_weights(seg_checkpoint_path).expect_partial()
            Segment_Labels_by_Size = Segment_Labels_by_Size + [seg_model.predict(np.array(segments_by_size), verbose=0)]
    Segment_Labels_True_Order = UnshuffleSegmentLabels(Segment_Labels_by_Size, Segment_Addresses_by_Size)
    ### /Make predictions ###

    ### Convert output to AnDi format, restore EXP and FOV structure ###
    for exp in range(num_exps):
        ### collect all addresses, labels, and cps for this exp ###
        exp_mask = All_Traj_Addresses[:,0] == exp
        exp_Traj_Addresses = All_Traj_Addresses[exp_mask]
        exp_Labels = [seg_lab for e_mask, seg_lab in zip(exp_mask, Segment_Labels_True_Order) if e_mask]
        exp_CPs = [cp for e_mask, cp in zip(exp_mask, CPs) if e_mask]
        ### /collect all addresses, trajs,and cps for this exp ###

        ### create the correct file structure ###
        if mode == 'starting_k':
            results_dir_path = data_path + f'res/track_2/exp_{exp}/'
        elif mode == 'public_data':
            results_dir_path = os.getcwd() + f'/public_data_preds/track_2/exp_{exp}/'
        ### /create the correct file structure ###

        ### loop over all fovs in this experiment and write their info to different files ###
        fovs = np.unique(exp_Traj_Addresses[:,1])
        for fov in fovs:
            ### collect all info for this FOV ###
            fov_mask = exp_Traj_Addresses[:,1] == fov
            fov_Labels = [seg_lab for f_mask, seg_lab in zip(fov_mask, exp_Labels) if f_mask]
            fov_CPs = [cp for f_mask, cp in zip(fov_mask, exp_CPs) if f_mask]
            ### /collect all info for this FOV ###

            ### write to file ###
            Path(results_dir_path).mkdir(parents=True, exist_ok=True)    # make parent dir if needed
            file = open(results_dir_path + f'fov_{fov}.txt', 'w')
            for traj_idx, (Traj_Labels, Traj_CPs) in enumerate(zip(fov_Labels, fov_CPs)):
                prediction_string = str(traj_idx)
                for seg_label, cp in zip(Traj_Labels, Traj_CPs):
                    prediction_string = (prediction_string + ','  
                                        +str(seg_label[0]) + ',' 
                                        +str(seg_label[1]) + ',' 
                                        +str(round(seg_label[1])) + ','
                                        +str(cp))
                prediction_string = prediction_string + '\n'
                file.write(prediction_string)
            file.close()
            ### /write to file ###
        ### loop over all fovs in this experiment and write their info to diff files ###
 
        
def PredictAndSplit_Inference(model, padded_trajs, padding_mask, return_CPs=False, min_peak_height=0.25):
    '''
    Takes prepared trajectories and makes predictions on them.
    Splits up the trajs according to CP preds and makes input for next network.
    Returns the split trajectories along with instructuions to reconstruct original trajs.

    model - The network
    padded_trajs - The padded input trajectories
    padding_mask - A mask showing which parts of the trajectories are just padding
    '''
    ## Make predictions ##
    Pred_Labs = model.predict(padded_trajs)
    Pred_Labs = np.concatenate(Pred_Labs, axis=2)
    ## /Make predictions ##

    ## Split up the segments, keeping track of where each came from ##
    segments = []    # we will collect all the segments across the dataset into this one list
    segment_addresses = []    # this will allow us to assign the generated labs to the correct trajs
    if return_CPs == True:
        All_CPs = []
    for traj_idx, (traj, pred_lab) in enumerate(zip(padded_trajs, Pred_Labs)):
        ## Undo the padding ##
        traj = traj[padding_mask[traj_idx]]   
        pred_lab = pred_lab[padding_mask[traj_idx]]
        ## /Undo the padding ##

        CP_labels = pred_lab[:,0]
        alpha_and_K_and_class_each_timestep = pred_lab[:,1:]

        ## Get Changepoints ##
        CPs = LabelToCP(CP_labels, min_peak_height=min_peak_height)
        if return_CPs == True:
            All_CPs = All_CPs + [np.concatenate((CPs, [np.count_nonzero(padding_mask[traj_idx])]))]    # append lenght as a final CP
        ## /Get Changepoints ##

        ## Split according to changepoitns ##
        traj_segments = np.split(traj, CPs)
        pred_label_segments = np.split(alpha_and_K_and_class_each_timestep, CPs)

        for seg_idx, (traj_seg, pred_lab_seg) in enumerate(zip(traj_segments, pred_label_segments)):
            full_seg = np.concatenate([traj_seg, pred_lab_seg], axis=1)
            segments += [full_seg]
            segment_addresses += [[traj_idx, seg_idx]]
        ## /Split according to changepoitns ##
    ## /Split up the segments, keeping track of where each came from ##

    ## Collect segments based on their lenght
    Segments_by_Size = [[]] * 200
    Segment_Addresses_by_Size = [[]] * 200
    for segment, address in zip(segments, segment_addresses):
            seg_idx = len(segment) - 1
            Segments_by_Size[seg_idx] = Segments_by_Size[seg_idx] + [segment]
            Segment_Addresses_by_Size[seg_idx] = Segment_Addresses_by_Size[seg_idx] + [address]
    
    if return_CPs == True:
        return Segments_by_Size, Segment_Addresses_by_Size, All_CPs
    return Segments_by_Size, Segment_Addresses_by_Size        
