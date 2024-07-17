import numpy as np
import sys
from utils import *
from andi_datasets.models_phenom import models_phenom
from andi_datasets.datasets_challenge import challenge_phenom_dataset
import random

def PrepareLabels(Labels, label_CPs=True):
    '''
    Go from the labels outputted by andi-datasets to labels our network will train on.
    '''  
    if label_CPs is True:
        # Work out CP locations
        labs_diff = Labels[1:] - Labels[:-1]
        labs_diff_sum = np.sum(abs(labs_diff), axis=2)
        CPs = np.where(labs_diff_sum != 0, 1, 0)
        # Prep to concatenate to labels
        zeros = np.zeros([1, np.shape(CPs)[1]])
        CPs = np.concatenate([zeros, CPs], axis=0)
        CPs = np.expand_dims(CPs, 2)
        # Concatenate to labels
        Labels = np.concatenate([CPs, Labels], axis=2)
    
    # Prune the model information
    Labels = Labels[:,:,:-1]
    return Labels

##########################################################################

def GenerateMSMData(N=100000, T=128, L=1.5*128, D=0.1):
    '''
    Generate MSM data and prepare labels.
    '''    
    trajs, labels = models_phenom().multi_state(N = N, 
                                                L = L,
                                                T = T,
                                                alphas = [1.2, 0.7],
                                                Ds = [[10*D, 0.1], [0.1*D, 0.0]], 
                                                M = [[0.98, 0.02], [0.02, 0.98]])
    labels =  PrepareLabels(labels)
    return np.swapaxes(trajs, 0,1), np.swapaxes(labels, 0,1)

##########################################################################
########## MAKE DICTIONARIES ##########
MSM_param_bounds = {'num_segs': (2,5),
                    'Pii': (0.9,0.999)} 
                            
DIM_param_bounds = {'r': (0.1,2.0),
                    'N': (50,150),
                    'Pb': 1,
                    'Pu': (0,0.1)}

TCM_param_bounds = {'Nc': (10,100),
                    'rc': (1,15),
                    'T': (0,0.5)}

QTM_param_bounds = {'Nt': (100,500),
                    'rt': (0.1,2.0),
                    'Pb': 1,
                    'Pu': (0,0.1)}

def Gen_Random_Model_Param_Dict(ensemble_models, ensemble_distributions, N=100,T=200, L=230.4):
    '''
    Use the param_bounds dictionaries to generate a set of random model dictionaries
    '''
    def gen_param(param, param_type='float'):
        if type(param) == tuple:
            if param_type == 'float':
                return np.random.uniform(*param)
            else:
                return np.random.randint(*param)
        else:
            return param
    
    OUTPUT_DICTS = []
    
    # Single state
    if 'single_state' in ensemble_models:     
        ens_dist = random.choice(ensemble_distributions)
        Ds = [ens_dist["D_mean"], ens_dist["D_var"]]
        alphas = [ens_dist["alpha_mean"], ens_dist["alpha_var"]]

        dict_single_state = {'model': 'single_state',
                             'T': T,
                             'N': N,
                             'Ds': np.array(Ds),
                             'alphas': np.array(alphas),
                             'dim': 2,
                             'L': L}
        OUTPUT_DICTS = OUTPUT_DICTS + [dict_single_state]
    
    # Multi state
    if 'multi_state' in ensemble_models: 
        num_segs = gen_param(MSM_param_bounds['num_segs'], 'int')
        Pii = gen_param(MSM_param_bounds['Pii'])
        Pij = (1 - Pii)/(num_segs-1)
        M = np.full([num_segs, num_segs], Pij)
        shuffled_ens_dist = random.sample(ensemble_distributions, len(ensemble_distributions))    # shuffle the ensemble distributions
        Ds = []
        alphas = []
        for i in range(num_segs):
            M[i,i] = Pii
            ens_dist = shuffled_ens_dist[i%len(shuffled_ens_dist)]
            Ds = Ds + [[ens_dist["D_mean"], ens_dist["D_var"]]]
            alphas = alphas + [[ens_dist["alpha_mean"], ens_dist["alpha_var"]]]

        dict_multi_state = {'model': 'multi_state',
                            'T': T,
                            'N': N,
                            'Ds': np.array(Ds),
                            'alphas': np.array(alphas),
                            'M': M,
                            'L': L,
                            'return_state_num': True}
        OUTPUT_DICTS = OUTPUT_DICTS + [dict_multi_state]

    # Dimerization
    if 'dimerization' in ensemble_models: 
        shuffled_ens_dist = random.sample(ensemble_distributions, len(ensemble_distributions))    # shuffle the ensemble distributions
        Ds = []
        alphas = []
        for i in range(2):
            ens_dist = shuffled_ens_dist[i%len(shuffled_ens_dist)]
            Ds = Ds + [[ens_dist["D_mean"], ens_dist["D_var"]]]
            alphas = alphas + [[ens_dist["alpha_mean"], ens_dist["alpha_var"]]]

        dict_dimerization = {'model': 'dimerization',
                             'T': T,
                             'N': gen_param(DIM_param_bounds['N'], 'int'),
                             'Ds': np.array(Ds),
                             'alphas': np.array(alphas),
                             'r': gen_param(DIM_param_bounds['r']),
                             'Pb': gen_param(DIM_param_bounds['Pb']),
                             'Pu': gen_param(DIM_param_bounds['Pu']),
                             'L': L,
                             'return_state_num': True}
        OUTPUT_DICTS = OUTPUT_DICTS + [dict_dimerization]
    
    # Confinement
    if 'confinement' in ensemble_models: 
        if len(ensemble_distributions) > 1:
            Ds = []
            alphas = []
            for i in range(2):
                ens_dist = ensemble_distributions[i]
                Ds = Ds + [[ens_dist["D_mean"], ens_dist["D_var"]]]
                alphas = alphas + [[ens_dist["alpha_mean"], ens_dist["alpha_var"]]]
        else:
            ens_dist = ensemble_distributions[0]
            D_scaling = np.random.uniform()
            Ds = [[ens_dist["D_mean"], ens_dist["D_var"]], [D_scaling*ens_dist["D_mean"], D_scaling*ens_dist["D_var"]]]
            alpha_scaling = np.random.uniform()
            alphas = [[ens_dist["alpha_mean"], ens_dist["alpha_var"]], [alpha_scaling*ens_dist["alpha_mean"], alpha_scaling**ens_dist["alpha_var"]]]

        dict_confinement = {'model': 'confinement',
                            'T': T,
                            'N': N,
                            'Ds': np.array(Ds),
                            'alphas': np.array(alphas),
                            'Nc': gen_param(TCM_param_bounds['Nc'], 'int'),
                            'r': gen_param(TCM_param_bounds['rc']),
                            'trans': gen_param(TCM_param_bounds['T']),
                            'L': L}
        OUTPUT_DICTS = OUTPUT_DICTS + [dict_confinement]
    
    # Immobile traps
    if 'immobile_traps' in ensemble_models: 
        ens_dist = random.choice(ensemble_distributions)
        Ds = [ens_dist["D_mean"], ens_dist["D_var"]]
        alphas = [ens_dist["alpha_mean"], ens_dist["alpha_var"]]

        dict_immobile_traps = {'model': 'immobile_traps',
                               'T': T,
                               'N': N,
                               'Ds': np.array(Ds),
                               'alphas': np.array(alphas),
                               'Nt': gen_param(QTM_param_bounds['Nt'], 'int'), 
                               'r': gen_param(QTM_param_bounds['rt']),
                               'Pb': gen_param(QTM_param_bounds['Pb']),
                               'Pu': gen_param(QTM_param_bounds['Pu']),
                               'L': L}
        OUTPUT_DICTS = OUTPUT_DICTS + [dict_immobile_traps]
    
    return OUTPUT_DICTS
########## \MAKE DICTIONARIES ##########

########## MAKE DATASET ##########
def GenerateTrueData_EMS(ensemble_models, ensemble_distributions, N=10000, return_padding_mask=False, max_traj_len=200):
    '''
    Generate data that will be used for training    
    '''
    all_trajs = np.zeros((N,max_traj_len,2))
    if return_padding_mask:
        padding_mask = np.full((N,max_traj_len), True)
    all_labels = np.zeros((N,max_traj_len,3))
    traj_idx = -1

    break_flag = False
    while True:
        
        ### WEIRD CODE WEIRD CODE WEIRD CODE WEIRD CODE ###  --> here as the AnDi code sometimes glitches!
        while True:
            try:
                ## Generate FOVs, convert to list of numpy arrays
                FOVs, _, _ = challenge_phenom_dataset(dics=Gen_Random_Model_Param_Dict(ensemble_models, ensemble_distributions, N=100), return_timestep_labs=True)
            except:
                print('FOV gen failure; trying again.')
                continue
            else:
                break
        ### WEIRD CODE WEIRD CODE WEIRD CODE WEIRD CODE ###
        FOVs = [FOV.to_numpy() for FOV in FOVs]
        
        ## Split FOVs up into different trajs, and add padding as appropriate ##
        for fov_idx, FOV in enumerate(FOVs):
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
                
                if return_padding_mask:
                    padding_mask[traj_idx][:first_frame] = False    # keep track of what values are padding
                    padding_mask[traj_idx][last_frame+1:] = False
                    
            if break_flag:
                break
        if break_flag:
            break
    if return_padding_mask:
        return all_trajs, all_labels, padding_mask
    else:
        return all_trajs, all_labels
########## \MAKE DATASET ##########
