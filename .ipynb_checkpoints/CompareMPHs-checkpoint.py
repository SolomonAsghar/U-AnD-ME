import numpy as np
import sys
sys.path.append(r'/home/cs-solomon.asghar/AnDi_2024/software/')
from utils import DiffTrajs

from SmartEval import *
from MakePreds import *

def PhaseOnePredictions(model, data_path, max_traj_len=200, min_peak_height=0.25):
    '''
    Load all data and make predictions using a U-Net
    '''
    ### Load data and prepare it for network ###
    All_Trajs = []    # stores all the trajs across all exps and fovs!
    All_Padding_Masks = []    # stores all the padding masks across all exps and fovs!
    All_Traj_Addresses = []    # for each traj, stores what exp and fov its from!

    num_exps = len(os.listdir(data_path + '/track_2/'))
    for exp in range(num_exps):
        all_files = os.listdir(data_path + f'/track_2/exp_{exp}/')
        num_fovs = len([fov for fov in all_files if fov.startswith('trajs_fov')])
        for fov in range(num_fovs):
            FOV_df = pd.read_csv(data_path + f'track_2/exp_{exp}/trajs_fov_{fov}.csv')
            FOV = FOV_df.to_numpy()

            num_trajs = int(FOV[-1,0]) + 1
            all_trajs = np.zeros((num_trajs,max_traj_len,2))   # prepare a container for all the trajs
            traj_idx = -1
            padding_mask = np.full((num_trajs,max_traj_len), True)    # keeps track of what is padded vs authentic data

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
    All_Trajs = DiffTrajs(All_Trajs)
    All_Padding_Masks = np.concatenate(All_Padding_Masks, axis=0)        
    All_Traj_Addresses = np.array(All_Traj_Addresses)
    ### /Load data and prepare it for network ###
    
    return All_Traj_Addresses, All_Trajs, All_Padding_Masks, num_exps, num_fovs
    
    ## Make predictions ###        

    
def PredictAndSplit_Inference(Pred_Labs, model, padded_trajs, padding_mask, return_CPs=False, min_peak_height=0.25):
    '''
    Takes prepared trajectories and makes predictions on them.
    Splits up the trajs according to CP preds and makes input for next network.
    Returns the split trajectories along with instructuions to reconstruct original trajs.

    model - The network
    padded_trajs - The padded input trajectories
    padding_mask - A mask showing which parts of the trajectories are just padding
    '''
    ## Make predictions ##
    # Pred_Labs = model.predict(padded_trajs)
    # Pred_Labs = np.concatenate(Pred_Labs, axis=2)
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

    
    
    
def ComparePeakHeights(model, max_traj_len=224, output_codes=['a','b','c'], min_peak_heights=[0.25, 0.5, 0.75]):

    data_path = '/home/cs-solomon.asghar/AnDi_2024/Evaluation/StartingKitEval/ref/'

    # Make Preds
    All_Traj_Addresses, All_Trajs, All_Padding_Masks, num_exps, num_fovs = PhaseOnePredictions(model, data_path, max_traj_len=max_traj_len)
    
    Pred_Labs = model.predict(All_Trajs)
    Pred_Labs = np.concatenate(Pred_Labs, axis=2)
    
    for output_code, min_peak_height in zip(output_codes, min_peak_heights):
        output_name = f"/MPHs/res_sk_{output_code}/"
        eval_output_name = f'/MPHs/SK_{output_code}'
        
        Segments_by_Size, Segment_Addresses_by_Size, CPs = PredictAndSplit_Inference(Pred_Labs, model, All_Trajs, All_Padding_Masks, return_CPs=True, min_peak_height=min_peak_height)
    
        PhaseTwoPredictions(All_Traj_Addresses, Segments_by_Size, Segment_Addresses_by_Size, CPs, 
                            num_exps, num_fovs,
                            output_name=output_name, use_seg_nets=False, local_eval_mode=False)

        # Evaluate and Save results
        submit_dir = os.getcwd() + output_name
        output_path = os.getcwd() + eval_output_name
        CodaLab_Eval(submit_dir, data_path, output_path)