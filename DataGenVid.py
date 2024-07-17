########## Modify some andi_datasets functions ##########
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import csv
import shutil
from pathlib import Path

from DataGen import Gen_Random_Model_Param_Dict

from andi_datasets.utils_challenge import segs_inside_fov, label_continuous_to_list, extract_ensemble, label_filter, df_to_array, file_nonOverlap_reOrg
from andi_datasets.datasets_phenom import datasets_phenom
from andi_datasets.datasets_theory import datasets_theory
from andi_datasets.utils_trajectories import normalize
from andi_datasets.utils_videos import transform_to_video, psf_width
from scipy.spatial import distance

def get_max_VIP(array_trajs, min_distance_part = 2, pad = -1, 
                boundary = False, boundary_origin = (0,0), min_distance_bound = 0,
                sort_length = True):
    '''
    Given an array of trajectories, finds max possible VIP particles that participants will
    need to characterize in the video track.
    '''
    if not boundary:
        candidates_vip = np.argwhere(array_trajs[0,:,0] != pad).flatten()
    else:
        # Define masks
        boundary_x0 = array_trajs[0,:,0] > (boundary_origin[0] + min_distance_bound)
        boundary_xL = array_trajs[0,:,0] < (boundary_origin[0] + boundary - min_distance_bound)
        boundary_y0 = array_trajs[0,:,1] > (boundary_origin[1] + min_distance_bound)
        boundary_yL = array_trajs[0,:,1] < (boundary_origin[1] + boundary - min_distance_bound)
        padding = array_trajs[0,:,0] != pad
        
        candidates_vip = np.argwhere(boundary_x0 & boundary_xL & boundary_y0 & boundary_yL & padding).flatten()        

    num_vip = len(candidates_vip)    # hacky way to force it to give us all possible VIPs    

    elected = []
    count_while = 0    
    
    if sort_length:
        array_candidates = array_trajs[:, candidates_vip, :]
        lengths = np.ones(array_candidates.shape[1])*array_candidates.shape[0]
        where_pad = np.argwhere(array_candidates[:,:,0] == pad)
        lengths[where_pad[:,1]] = where_pad[:,0]
        # We sort the particle by their lenghts (note the minus for descending order)
        candidates_vip = candidates_vip[np.argsort(-lengths)]
        
    while len(elected) < num_vip:
        
        if sort_length and count_while == 0: 
            # if we already did a while loop, we start with a random candidate even
            # when sorting
            elected = [candidates_vip[0]]
        else:
            elected = [np.random.choice(candidates_vip)]

        for c_idx in candidates_vip:
            if c_idx == elected[0]:
                continue
            if len(array_trajs[0, elected,:].shape) < 2:
                all_rest = np.expand_dims(array_trajs[0, elected,:], 0)
            else:
                all_rest = array_trajs[0, elected,:]

            dist = distance.cdist(np.expand_dims(array_trajs[0,c_idx,:], 0), all_rest, metric='euclidean').transpose()

            if dist.min() > min_distance_part:
                elected.append(c_idx)
            else:
                num_vip = num_vip - 1

            if len(elected) == num_vip:
                break

        count_while += 1
        if count_while > 100: 
            raise ValueError('Could not find suitable VIP particles. This is due to either having to few particles or them being too close')
            
    return elected


class _defaults_andi2: 
    '''
    This class defines the default values set for the ANDI 2 challenge.
    '''
    def __init__(self):        
        # General parameters

        self.T = 500                   # Length of simulated trajectories
        self._min_T = 20               # Minimal length of output trajectories
        self.FOV_L = 128               # Length side of the FOV (px)
        self.L = 1.8*self.FOV_L          # Length of the simulated environment
        self.D = 1                     # Baseline diffusion coefficient (px^2/frame)
        self.density = 2               # Particle density   
        self.N = 50                    # Number of particle in the whole experiment
        self.sigma_noise = 0.12        # Variance of the localization noise

        self.label_filter = lambda x: label_filter(x, window_size = 5, min_seg = 3) 


def challenge_phenom_dataset_videos_custom(experiments = 5,
                                           dics = None,
                                           repeat_exp = True,
                                           num_fovs = 1,
                                           return_timestep_labs = False,
                                           get_video = False, num_vip = None, get_video_masks = False):
    ''' 
    Creates a datasets with same structure as ones given in the ANDI 2 challenge. 
    This is a custom function with different behaviour to the official one. 
    '''
    # Set prefixes for saved files
    if return_timestep_labs:
        df_list = []

    # Sets the models of the experiments that will be output by the function
    if dics is None:
        if isinstance(experiments, int):
            if repeat_exp: # If experiments can be repeated, we just sample randomly
                model_exp = np.random.randint(len(datasets_phenom().avail_models_name), size = experiments)
            else: # If not, we sampled them in an ordered way
                if experiments >= len(datasets_phenom().avail_models_name):
                    num_repeats = (experiments % len(datasets_phenom().avail_models_name))+1
                else:
                    num_repeats = 1
                model_exp = np.tile(np.arange(len(datasets_phenom().avail_models_name)), num_repeats)[:experiments]
            # We add one to get into non-Python numeration
            model_exp += 1
        else:
            model_exp = experiments
    # If list of dics is given, then just create a list of length = len(dics)
    else: 
        model_exp = [0]*len(dics)

    # Output lists
    trajs_out, labels_traj_out, labels_ens_out = [], [], []
    for idx_experiment, model in enumerate(tqdm(model_exp)):       

        ''' Generate the trajectories '''
        if dics is None:
            dic = _get_dic_andi2(model)
        else:
            dic = dics[idx_experiment]
            # Overide the info about model
            model = datasets_phenom().avail_models_name.index(dic['model'])+1    
        print(f'Creating dataset for Exp_{idx_experiment} ('+dic['model']+').')
        trajs, labels = datasets_phenom().create_dataset(dics = dic)            
        
        
        ''' Apply the FOV '''
        for fov in range(num_fovs):           

            # We take as min/max for the fovs a 5 % distance of L
            dist = 0.05
            min_fov = int(dist*_defaults_andi2().L)
            max_fov = int((1-dist)*_defaults_andi2().L)-_defaults_andi2().FOV_L
            # sample the position of the FOV
            fov_origin = (np.random.randint(min_fov, max_fov), np.random.randint(min_fov, max_fov))
            
            ''' Go over trajectories in FOV (copied from utils_trajectories for efficiency) '''
            trajs_fov, array_labels_fov, list_labels_fov, idx_segs_fov, frames_fov = [], [], [], [], []
            idx_seg = -1

            # Total frames
            frames = np.arange(trajs.shape[0])
            # We save the correspondance between idx in FOV and idx in trajs dataset
            for idx, (traj, label) in enumerate(zip(trajs[:, :, :].transpose(1,0,2),
                                                    labels[:, :, :].transpose(1,0,2))):
                                
                nan_segms = segs_inside_fov(traj[:,:2], # take only the 2D projection of the traj
                                            fov_origin = fov_origin,
                                            fov_length = _defaults_andi2().FOV_L,
                                            cutoff_length = _defaults_andi2()._min_T)

                if nan_segms is not None:
                    for idx_nan in nan_segms:  
                        idx_seg+= 1  
                        
                        trajs_fov.append(traj[idx_nan[0]:idx_nan[1]])
                        frames_fov.append(frames[idx_nan[0]:idx_nan[1]])

                        lab_seg = []
                        for idx_lab in range(labels.shape[-1]):
                            lab_seg.append(_defaults_andi2().label_filter(label[idx_nan[0]:idx_nan[1], idx_lab]))
                        lab_seg = np.vstack(lab_seg).transpose()                    
                        array_labels_fov.append(lab_seg)

                        # Tranform continuous labels to list for correct output
                        if model == 2 or model == 4: 
                            # if multi-state or dimerization, we get rid of the label of state numbering
                            CP, alphas, Ds, states = label_continuous_to_list(lab_seg[:, :-1])
                        else:
                            CP, alphas, Ds, states = label_continuous_to_list(lab_seg)
                        
                        # Extract final point of trajectory 
                        T = CP[-1]
                        CP = CP[:-1]
                        list_gt = [idx_seg, Ds[0], alphas[0], states[0]]
                        for gtc, gta, gtd, gts in zip(CP, alphas[1:], Ds[1:], states[1:]):
                            list_gt += [gtc, gtd, gta, gts]
                        # Add end point of trajectory
                        list_gt.append(T)
                        list_labels_fov.append(list_gt)     

                        # Save index of segment with its length to latter append in the dataframe              
                        idx_segs_fov.append(np.ones(trajs_fov[-1].shape[0])*idx_seg)             
            
            '''Extract ensemble trajectories''' 
            ensemble_fov = extract_ensemble(np.concatenate(array_labels_fov)[:, -1], dic)

            df_data = np.hstack((np.expand_dims(np.concatenate(idx_segs_fov), axis=1),
                                 np.expand_dims(np.concatenate(frames_fov), axis=1).astype(int),
                                 np.concatenate(trajs_fov)))            
            
            if 'dim' in dic.keys() and dic['dim'] == 3:
                df_traj = pd.DataFrame(df_data, columns = ['traj_idx', 'frame', 'x', 'y','z'])
            else:                
                df_traj = pd.DataFrame(df_data, columns = ['traj_idx', 'frame', 'x', 'y'])
            
            
            if get_video:
                print(f'Generating video for EXP {idx_experiment} FOV {fov}')               
                
                pad = -20 #  padding has to be further enough from the FOV so that the PSF 
                          # of particles does not enter in the videos
                array_traj_fov = df_to_array(df_traj.copy(), pad = pad)
                min_distance = psf_width()
                idx_vip = get_max_VIP(array_traj_fov,
                                      min_distance_part = min_distance, 
                                      min_distance_bound = min_distance,
                                      boundary_origin = fov_origin,
                                      boundary = _defaults_andi2().FOV_L,                    
                                      pad = pad)  
                
                pf_videos = ''                                
                
                video_fov = transform_to_video(array_traj_fov, # see that we insert the trajectories without noise!
                                               optics_props={
                                                   "output_region":[fov_origin[0], fov_origin[1],
                                                                    fov_origin[0] + _defaults_andi2().FOV_L, fov_origin[1] + _defaults_andi2().FOV_L]
                                                },
                                               get_vip_particles=idx_vip,
                                               with_masks = get_video_masks,
                                               save_video = False) 
                
                try:
                    videos_out.append(video_fov)
                except:
                    videos_out = [video_fov] 
                                        
            # Zero to the fov origin, scaled down to [0,4] so that values are same oom as other labels
            df_traj.x = (df_traj.x - fov_origin[0])/32   
            df_traj.y = (df_traj.y - fov_origin[1])/32
                    
            if return_timestep_labs:
                array_labels_fov = np.concatenate(array_labels_fov)
                df_traj['alpha'] = array_labels_fov[:, 0]
                df_traj['D'] = array_labels_fov[:, 1]
                df_traj['state'] = array_labels_fov[:, 2]
            
            # Add data to main lists (trajectories and lists with labels)
            trajs_out.append(df_traj)
            labels_traj_out.append(list_labels_fov)
            labels_ens_out.append(ensemble_fov)
            
    # If asked, create a reorganized version of the folders
    if get_video:
        return trajs_out, videos_out, labels_traj_out, labels_ens_out
    else:
        return trajs_out, labels_traj_out, labels_ens_out
##########\ Modify some andi_datasets functions ##########


########## Main ##########
def NormalisePerFrame(video):
    '''
    Normalise the intensity over a video such that each frame is [0,1]
    '''
    min_intensity_by_frame = np.tile(np.expand_dims(np.min(video, axis=(1,2,3)), (1,2,3)), (1,128,128,1))
    video = video - min_intensity_by_frame
    max_intensity_by_frame = np.tile(np.expand_dims(np.max(video, axis=(1,2,3)), (1,2,3)), (1,128,128,1))
    video = video / max_intensity_by_frame
    return video 


def GenerateVideoData(N=10000):
    '''
    Generate N trajectories (as videos) with accompanying labels.
    '''
    all_videos = np.zeros((N,200,128,128,1))
    all_labels = np.zeros((N,200,5)) # x,y,a,k,s
    traj_idx = -1
    
    while True:
        # Generate videos with lots of VIPs
        ### BAD HACKY CODE ### Check here to confirm that the videos are correct len ### BAD HACKY CODE ###
        num_ts_vids = 0
        while num_ts_vids != 201:
            trajs_by_fov, videos_by_fov, _, _ = challenge_phenom_dataset_videos_custom(dics=Gen_Random_Model_Param_Dict(N=100), return_timestep_labs=True, get_video=True)
            num_ts_vids = np.shape(videos_by_fov)[1]
        ### BAD HACKY CODE ### Check here to confirm that the videos are correct len ### BAD HACKY CODE ###
        
        trajs_by_fov = [trajs.to_numpy() for trajs in trajs_by_fov]
        
        # Split the diff FOVs into diff videos
        for video, trajs in zip(videos_by_fov, trajs_by_fov):
            # Get all the possible 
            possible_traj_idxs, idx_timesteps = np.unique(trajs[:,0], return_index=True)
            idx_timesteps = np.concatenate([idx_timesteps, [len(trajs)]])
    
            ### BAD HACKY CODE ### Constrain VIP idxs to those that are in possible_traj_idxs ### BAD HACKY CODE ###
            vip_idxs = np.unique(video[0])[1:]
            vip_idxs = np.intersect1d(vip_idxs, possible_traj_idxs)
            ### BAD HACKY CODE ### Constrain VIP idxs to those that are in possible_traj_idxs ### BAD HACKY CODE ###
            
            for i, vip_idx in enumerate(vip_idxs):
                traj_idx += 1    # we have a new traj
                if traj_idx == N:
                    break
                vip_mask = np.where(video[0] == vip_idx, 1, 0)    # create a mask for this traj
                all_videos[traj_idx,0] = vip_mask    # store this mask
                all_videos[traj_idx,1:] = NormalisePerFrame(video[1:-1])
    
                first_timestep = idx_timesteps[int(vip_idx)]
                last_timestep = idx_timesteps[int(vip_idx)+1]
    
                vip_labels = trajs[first_timestep:last_timestep][:,1:]
                vip_first_frame = int(vip_labels[0,0])
                vip_final_frame = int(vip_labels[-1,0])
                
                all_labels[traj_idx][vip_first_frame:vip_final_frame+1] = vip_labels[:,1:]
                all_labels[traj_idx,:vip_first_frame,:2] = vip_labels[0,1:3]    
                all_labels[traj_idx,vip_final_frame:,:2] = vip_labels[-1,1:3]
            
            if traj_idx == N:
                break
        if traj_idx == N:
                break
    
    return all_videos, all_labels
########## \Main ##########e