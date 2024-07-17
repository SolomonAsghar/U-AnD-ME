import numpy as np
import sys
sys.path.append(r'/home/cs-solomon.asghar/AnDi_2024/software/')
from utils import *
from andi_datasets.models_phenom import models_phenom
from andi_datasets.datasets_challenge import challenge_phenom_dataset
from DataGen import *

FOVs, videos_out, labels, _ = challenge_phenom_dataset(dics=Gen_Random_Model_Param_Dict(N=100), return_timestep_labs=True)
FOVs = [FOV.to_numpy() for FOV in FOVs]


def GenerateTrueData(N=10000):
    all_trajs = np.zeros((N,200,2))
    all_labels = np.zeros((N,200,3))
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
                traj_idx += 1
                if traj_idx == N:
                    break_flag = True
                    break
                first_frame, last_frame = int(traj[0,1]), int(traj[-1,1])

                all_trajs[traj_idx][first_frame:last_frame+1] = traj[:,2:4]    # drop in the traj
                all_trajs[traj_idx][:first_frame] = traj[0,2:4]    # Pad the traj!
                all_trajs[traj_idx][last_frame:] = traj[-1,2:4]

                all_labels[traj_idx][first_frame:last_frame+1] = traj[:,4:]    # drop in the label

            if break_flag:
                break
        if break_flag:
            break
            
    return all_trajs, all_labels