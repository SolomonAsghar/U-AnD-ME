import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal, optimize
from pathlib import Path
import sys
sys.path.append(r'/home/cs-solomon.asghar/AnDi_2024/software/')
from utils import ExplicitCPs

def CompareLabs(PredLabs, TrueLabs, First_and_Last_Timesteps):
    pred_CPs = PredLabs[:,0]
    pred_Ks = PredLabs[:,1]
    pred_alphas = PredLabs[:,2]
    pred_models = PredLabs[:,3]

    true_CPs = TrueLabs[:,0]
    true_Ks = TrueLabs[:,1]
    true_alphas = TrueLabs[:,2]
    true_models = TrueLabs[:,3]

    plt.plot(true_CPs, color='black', label='True')
    plt.plot(pred_CPs, color='red', label='Pred', marker='o')
    plt.xlim(0,200)
    plt.ylim(-0.05,1.05)
    plt.title('CP')
    plt.axvline(First_and_Last_Timesteps[0], color='green')
    plt.axvline(First_and_Last_Timesteps[1], color='royalblue')
    plt.legend()
    plt.show()
    
    plt.plot(true_Ks, color='black', label='True')
    plt.plot(pred_Ks, color='red', label='Pred')
    plt.xlim(0,200)
    plt.ylim(-0.05,2.05)
    plt.title('K')
    plt.axvline(First_and_Last_Timesteps[0], color='green')
    plt.axvline(First_and_Last_Timesteps[1], color='royalblue')
    plt.legend()
    plt.show()
    
    plt.plot(true_alphas, color='black', label='True')
    plt.plot(pred_alphas, color='red', label='Pred')
    plt.xlim(0,200)
    plt.ylim(-0.05,2.05)
    plt.title(r'$\alpha$')
    plt.axvline(First_and_Last_Timesteps[0], color='green')
    plt.axvline(First_and_Last_Timesteps[1], color='royalblue')
    plt.legend()
    plt.show() 
    
    plt.plot(true_models, color='black', label='True')
    plt.plot(pred_models, color='red', label='Pred')
    plt.xlim(0,200)
    plt.ylim(-0.05,3.05)
    plt.title('model')
    plt.axvline(First_and_Last_Timesteps[0], color='green')
    plt.axvline(First_and_Last_Timesteps[1], color='royalblue')
    plt.legend()
    plt.show() 
    
    
def LabelToPrediction(Labs, from_network=True, min_peak_height=0.25):
    '''
    Convert a list of labels describing properties at each timestep into predictions about the CP positions, and alpha and K for each segment.
    '''
    CP_labels = Labs[:,0]
    alpha_and_K_each_timestep = Labs[:,1:]

    # Get the locations of each changepoint 
    if from_network:
        CPs = signal.find_peaks(CP_labels, min_peak_height, distance=2)[0]
    else:
        CPs = np.nonzero(CP_labels)[0]

    # If we have changepoints, segment the labels
    if len(CPs) == 0:
        mean_alpha, mean_K = np.mean(alpha_and_K_each_timestep, axis=0)
        return CPs, np.array([mean_alpha]), np.array([mean_K])
    segments = np.split(alpha_and_K_each_timestep, CPs)

    # Calculate mean values for of each segment type 
    mean_alpha_a, mean_K_a = np.mean(np.concatenate(segments[::2]), axis=0)
    mean_alpha_b, mean_K_b = np.mean(np.concatenate(segments[1::2]), axis=0)

    return CPs, np.array([mean_alpha_a, mean_alpha_b]), np.array([mean_K_a, mean_K_b])


def LabelToPredictionString(Labs, traj_idx, from_network=True, min_peak_height=0.25):
    '''
    Convert a list of labels describing properties at each timestep into predictions about the CP positions, and alpha and K and class for each segment.
    '''
    CP_labels = Labs[:,0]
    alpha_and_K_and_class_each_timestep = Labs[:,1:]

    # Get the locations of each changepoint 
    if from_network:
        CPs = signal.find_peaks(CP_labels, min_peak_height, distance=2)[0]
    else:
        CPs = np.nonzero(CP_labels)[0]
    CPs = CPs[CPs > 2]    # Ignore CPs on first few time-step, min seg lenght is 3 anyways!
    CPs = CPs[(len(Labs) - CPs) > 2]    # Ignore CPs on final few time-step
    
    # If we have changepoints, segment the labels
    if len(CPs) == 0:
        mean_alpha, mean_K, mean_class = np.mean(alpha_and_K_and_class_each_timestep, axis=0)
        prediction_string = str(traj_idx) + ',' + str(mean_K) + ',' + str(mean_alpha) + ',' + str(round(mean_class))
    
    else:
        segments = np.split(alpha_and_K_and_class_each_timestep, CPs)
        mean_alpha, mean_K, mean_class = np.mean(segments[0], axis=0)
        prediction_string = str(traj_idx) + ',' + str(mean_K) + ',' + str(mean_alpha) + ',' + str(round(mean_class))    # deal with the first segment
        for CP, segment in zip(CPs, segments[1:]):    # loop over each remaining segment and work out its parameters
            mean_alpha, mean_K, mean_class = np.mean(segment, axis=0)
            prediction_string += ',' + str(CP) + ',' + str(mean_K) + ',' + str(mean_alpha) + ',' + str(round(mean_class))

    return prediction_string + ',' + str(len(Labs))


def EvaluateCPs(Predictions, Targets):
    '''
    Evaluate quality of CP predictions
    '''
    # Assign using Hungarian algorithm
    dist = lambda cp_1, cp_2: np.sqrt(((cp_1-cp_2)**2).sum())
    distance_matrix = np.asarray([[dist(cp_1, cp_2) for cp_2 in Targets] for cp_1 in Predictions])
    Pred_indicies, Target_indicies = optimize.linear_sum_assignment(distance_matrix)
    
    # Work out MSE on assigned CPs
    CP_E = np.array([Predictions[i_p] - Targets[i_t] for i_p, i_t in zip(Pred_indicies, Target_indicies)])
    CP_MSE = np.mean(CP_E**2)
    
    # Work out delta_M
    dM = len(Predictions) - len(Targets)
    
    return CP_MSE, dM


def EvaluateLabels(Predictions, Targets):
    '''
    Evaluate the predictions of a network 
    '''
    num_trajs = len(Predictions)
    num_CPs = num_trajs
    CP_MSEs = np.zeros(num_trajs)
    dMs = np.zeros(num_trajs)
    alphas_MSEs = np.zeros(num_trajs)
    Ks_MSEs = np.zeros(num_trajs)
    
    for i, (prediction, target) in enumerate(zip(Predictions, Targets)):
        CPs_pred, alphas_pred, Ks_pred = LabelToPrediction(prediction)
        CPs_target, alphas_target, Ks_target = LabelToPrediction(target, from_network=False)
        
        # Evaluate CPs
        if len(CPs_target) > 0 and len(CPs_pred) > 0:
            CP_MSEs[i], dMs[i] = EvaluateCPs(CPs_pred, CPs_target)
        else:
            CP_MSEs[i] = 0
            num_CPs -= 1
            dMs[i] = len(CPs_pred) - len(CPs_target)
        
        # Evaluate alphas and Ks
        alphas_MSEs[i] = np.mean((alphas_pred - Ks_pred)**2)
        Ks_MSEs[i] = np.mean((alphas_target - Ks_target)**2)
    
    return np.sum(CP_MSEs)/num_CPs, np.mean(dMs**2), np.mean(alphas_MSEs), np.mean(Ks_MSEs)


def MakePredictions(model, data_path, mode='starting_k'):
    '''
    Make AnDi predictions, for Track 2 Single Traj only. 
    '''
    num_exps = len(os.listdir(data_path + '/ref/track_2/'))
    for exp in range(num_exps):
        all_files = os.listdir(data_path + f'/ref/track_2/exp_{exp}/')
        num_fovs = len([fov for fov in all_files if fov.startswith('trajs_fov')])
        for fov in range(num_fovs):
            ## Load data and prepare it for network ##
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
            ## /Load data and prepare it for network ##

            ## Make predictions ##
            Labels_pred = model.predict(all_trajs)
            Labels_pred = np.concatenate(Labels_pred, axis=2)
            ## /Make predictions ##

            ## Convert network output to AnDi format and save ##
            if mode == 'starting_k':
                results_dir_path = data_path + f'res/track_2/exp_{exp}/'
            elif mode == 'public_data':
                results_dir_path = os.getcwd() + f'/public_data_preds/track_2/exp_{exp}/'
            Path(results_dir_path).mkdir(parents=True, exist_ok=True)    # make parent dir if needed
            file = open(results_dir_path + f'fov_{fov}.txt', 'w')
            for traj_idx, traj_labels in enumerate(Labels_pred):
                traj_labels = traj_labels[padding_mask[traj_idx]]
                prediction_string = LabelToPredictionString(traj_labels, traj_idx)
                file.write(prediction_string + '\n')
            file.close()        
            ## /Convert network output to AnDi format and save ##
            
            print(f"FOV {fov+1}/{num_fovs} complete.")
        print(f"Experiment {exp+1}/{num_exps} complete.")

        
def CompareTimestepLabels(model, data_path):
    '''
    Compare the predicted labels for each time-step with the true labels for each time-step. For Track 2 Single Traj only. 
    '''
    ALL_true_labels = []
    ALL_first_and_last_timesteps = []
    ALL_pred_labels = []
    
    num_exps = len(os.listdir(data_path + '/ref/track_2/'))
    for exp in range(num_exps):
        all_files = os.listdir(data_path + f'/ref/track_2/exp_{exp}/')
        num_fovs = len([fov for fov in all_files if fov.startswith('trajs_fov')])
        for fov in range(num_fovs):
            ## Load data and prepare it for network ##
            FOV_df = pd.read_csv(data_path + f'ref/track_2/exp_{exp}/trajs_fov_{fov}.csv')
            FOV = FOV_df.to_numpy()

            num_trajs = int(FOV[-1,0]) + 1
            all_trajs = np.zeros((num_trajs,200,2))   # prepare a container for all the trajs
            all_true_labels = np.zeros((num_trajs,200,3))
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
                
                all_true_labels[traj_idx][first_frame:last_frame+1] = traj[:,4:]    # drop in the label
                ALL_first_and_last_timesteps += [[first_frame, last_frame+1]]
            ## /Load data and prepare it for network ##

            ## Make predictions ##
            all_pred_labels = model.predict(all_trajs)
            all_pred_labels = np.concatenate(all_pred_labels, axis=2)
            ## /Make predictions ##
            
            ## Store everything ##
            ALL_true_labels += [ExplicitCPs(all_true_labels)]
            ALL_pred_labels += [all_pred_labels]
            ## \Store everything ##
            
            print(f"FOV {fov+1}/{num_fovs} complete.")
        print(f"Experiment {exp+1}/{num_exps} complete.")
        
    return np.concatenate(ALL_true_labels, axis=0), np.array(ALL_first_and_last_timesteps), np.concatenate(ALL_pred_labels, axis=0)