import os
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
sys.path.append(r'/home/cs-solomon.asghar/AnDi_2024/software/')
from utils import DiffTrajs
from SegEvaluate import *
from SegNet import *

def FixPredictions(prediction):
    "Enforce physical limits on the prediction"
    
    alphas = prediction[:,0]
    Ks = prediction[:,1]
    Ms = np.round(prediction[:,2])
    
    alphas = np.where(alphas > 0, alphas, 0)
    alphas = np.where(alphas < 1.999, alphas, 1.999)
    
    Ks = np.where(Ks > 1e-12, Ks, 1e-12)
    Ks = np.where(Ks < 1e6, Ks, 1e6)
    
    Ms = np.where(Ms > 0, Ms, 0)    
    Ms = np.where(Ms < 3, Ms, 3)
    Ms = np.where(alphas > 1.9, 3, Ms)    # if alpha is over 1.9, M must be 3
    
    fixed_prediction = np.stack([alphas, Ks, Ms], axis=1)
    return fixed_prediction

from sklearn import mixture

def AnalyseEnsembleProperties(Data):
    '''
    Returns optimal number of components, along with means and stds.
    '''
    n_components = np.arange(1, 15)[:len(Data)]
    models = [mixture.GaussianMixture(n, covariance_type='diag').fit(Data)
              for n in n_components]

    BICs = np.zeros(len(n_components))
    OverlapFree = np.zeros(len(n_components))

    for idx, model in enumerate(models):
        BICs[idx] = model.bic(Data) 
        OverlapFree[idx] = CheckGaussianOverlap(model.means_[:,:2], model.covariances_[:,:2], std_scale=1)
    
    best_model_index = np.argmin(BICs[OverlapFree!=0])
    best_model = models[best_model_index]
    
    opt_num_components = n_components[best_model_index]
    means = best_model.means_[:,:2]
    stds = best_model.covariances_[:,:2]
    weights = best_model.weights_
    
    return opt_num_components, means, stds, weights, best_model 


def CheckGaussianOverlap(means, covariances, std_scale=1):
    '''
    Checks if any Gaussians mean+- (std_scale) standard deviation(s) contains any other Gaussians' means.
    Returns True is the GMM is good, False if the GMM is bad.
    '''
    standard_deviations = np.sqrt(covariances)

    all_means_min_scaled_std = means - (standard_deviations*std_scale)
    all_means_plu_scaled_std = means + (standard_deviations*std_scale)

    for mean in means:
        others_mask = means != mean
        above_min_mask = all_means_min_scaled_std < mean
        below_max_mask = all_means_plu_scaled_std > mean

        in_std_range_mask = np.logical_and(above_min_mask, below_max_mask)
        other_means_in_std_range = in_std_range_mask[others_mask]

        if np.any(other_means_in_std_range):    # if any of the other Gaussians have means within one std of, this GMM is bad
            return False
    
    return True    # if not, this GMM is good!#

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

    ## Make predictions ###        
    Segments_by_Size, Segment_Addresses_by_Size, CPs = PredictAndSplit_Inference(model, All_Trajs, All_Padding_Masks, return_CPs=True, min_peak_height=min_peak_height)
    
    return All_Traj_Addresses, Segments_by_Size, Segment_Addresses_by_Size, CPs, num_exps, num_fovs

def PhaseTwoPredictions(All_Traj_Addresses, Segments_by_Size, Segment_Addresses_by_Size, CPs, 
                        num_exps, num_fovs,
                        output_name, use_seg_nets=True,
                        seg_model_folder_path='/home/cs-solomon.asghar/AnDi_2024/NetworkTesting/SegmentSpecificNets_Alt/',
                        local_eval_mode=False):
    '''
    Take the outputs from phase one and use them to make phase two predictions.
    '''
    available_networks = np.array([  3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
                                    16,  17,  18,  19,  20,  22,  23,  25,  27,  29,  31,  33,  35,
                                    36,  38,  40,  43,  44,  47,  50,  53,  54,  57,  59,  63,  64,
                                    68,  71,  75,  79,  80,  84,  87,  90,  95, 101, 105, 111, 116,
                                   123, 124, 128, 133, 141, 144, 150, 155, 162, 171, 181, 191, 200,
                                   207, 218, 229, 235, 241, 254, 267, 271, 286, 302, 306, 323, 341,
                                   342, 355, 364, 384, 405, 418, 441, 442, 462, 475])
    
    if use_seg_nets == True:
        min_seg_size = 5
    else:
        min_seg_size = np.inf
        
    Segments_by_Size = [sbs for sbs in Segments_by_Size if sbs != []]    # remove empty elements
    Segment_Addresses_by_Size = [sabs for sabs in Segment_Addresses_by_Size if sabs != []]
    Segment_Labels_by_Size = []

    for segments_by_size in tqdm(Segments_by_Size):           
        seg_size = np.shape(segments_by_size)[1] + 1    # accounts for the fact that we now start diffed
        if seg_size < min_seg_size: # if the seg is too small
            prediction = np.mean(segments_by_size, axis=1)[:,2:]
            Segment_Labels_by_Size = Segment_Labels_by_Size + [FixPredictions(prediction)]
        else:
            all_deltas = available_networks - seg_size
            first_positive_deltas_idx = np.argwhere(all_deltas<=0)[-1,0]
            closest_seg = available_networks[first_positive_deltas_idx]

            if seg_size != closest_seg:
                start = int((seg_size-closest_seg)/2)
                segments_by_size = np.array(segments_by_size)
                segments_by_size = segments_by_size[:,start:start+(closest_seg-1)]

            seg_model = BuildSegModel(closest_seg-1, input_width=2)
            seg_checkpoint_path = seg_model_folder_path + f'SegNet_{closest_seg}/Net_checkpoint'
            seg_model.load_weights(seg_checkpoint_path).expect_partial()
            segments_by_size = np.array(segments_by_size)[:,:,:2]
            prediction = seg_model.predict(segments_by_size, verbose=0)
            Segment_Labels_by_Size = Segment_Labels_by_Size + [FixPredictions(prediction)] 

    Segment_Labels_True_Order = UnshuffleSegmentLabels(Segment_Labels_by_Size, Segment_Addresses_by_Size)
    ## /Make predictions ###

    ### Convert output to AnDi format, restore EXP and FOV structure ###
    for exp in range(num_exps):
        ### collect all addresses, labels, and cps for this exp ###
        exp_mask = All_Traj_Addresses[:,0] == exp
        exp_Traj_Addresses = All_Traj_Addresses[exp_mask]
        exp_Labels = [seg_lab for e_mask, seg_lab in zip(exp_mask, Segment_Labels_True_Order) if e_mask]
        exp_CPs = [cp for e_mask, cp in zip(exp_mask, CPs) if e_mask]
        ### /collect all addresses, trajs,and cps for this exp ###

        ### create the correct file structure ###
        results_dir_path = os.getcwd() + f'/{output_name}/track_2/exp_{exp}/'
        ### /create the correct file structure ###
        
        if not local_eval_mode:
            ### Ensemble Level Analysis! ###
            flat_exp_labels = np.array([x for xs in exp_Labels for x in xs])
            num_components, means, stds, weights, GMM_model = AnalyseEnsembleProperties(flat_exp_labels)
            ### write to file ###
            Path(results_dir_path).mkdir(parents=True, exist_ok=True)
            file = open(results_dir_path + f'ensemble_labels.txt', 'w')
            prediction_string = f'model: multi_state; num_state: {num_components} \n'
            prediction_string += "; ".join(means[:,0].astype('str')) + '\n'    # all alpha means
            prediction_string += "; ".join(stds[:,0].astype('str')) + '\n'    # all alpha stds
            prediction_string += "; ".join(means[:,1].astype('str')) + '\n'    # all K means
            prediction_string += "; ".join(stds[:,1].astype('str')) + '\n'    # all K stds
            prediction_string += "; ".join(weights.astype('str'))    # weights
            file.write(prediction_string)
            file.close()
            ### write to file ###  
            ### Ensemble Level Analysis! ###


        # loop over all fovs in this experiment and write their info to different files ###
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
                                        +str(seg_label[1]) + ','    # Ks 
                                        +str(seg_label[0]) + ','    # alphas
                                        +str(seg_label[2]) + ','    # Ms
                                        +str(cp))
                prediction_string = prediction_string + '\n'
                file.write(prediction_string)
            file.close()
            # /write to file ###
        # loop over all fovs in this experiment and write their info to diff files ###