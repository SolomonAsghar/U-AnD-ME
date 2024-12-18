import numpy as np
import sys
from utils import *
from andi_datasets.models_phenom import models_phenom
from andi_datasets.datasets_challenge import challenge_phenom_dataset
import random

def OneHotEncoder(input_class, num_classes=5):
    "Returns simple one hot encoded array"
    label = np.zeros(5)    
    label[input_class] = 1
    return label

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

def Gen_Random_Model_Param_Dict(N=100,T=200, L=230.4):
    '''
    Use the param_bounds dictionaries to generate a set of random model dictionaries
    '''
    # All Ensemble Dists #
    exp0_param = [{'D_mean': 0.104229422,
                   'D_var': 0.001166479,
                   'alpha_mean': 0.970239179,
                   'alpha_var': 0.026103678}]

    exp1_param = [{'D_mean': 0.103780753,
                   'D_var': 0.001157472,
                   'alpha_mean': 0.975343424,
                   'alpha_var': 0.026023816}]

    exp2_param = [{'D_mean': 1.173273976,
                   'D_var': 0.159293209,
                   'alpha_mean': 0.996544889,
                   'alpha_var': 0.025733102},

                  {'D_mean': 0.018891323,
                   'D_var': 0.000229975,
                   'alpha_mean': 0.01533281,
                   'alpha_var': 0.000458715}]

    exp3_param = [{'D_mean': 0.84815965,
                   'D_var': 0.395201726,
                   'alpha_mean': 0.798281856,
                   'alpha_var': 0.102034826},

                 {'D_mean': 0.179309144,
                   'D_var': 0.074173715,
                   'alpha_mean': 0.310325879,
                   'alpha_var': 0.035854887},

                  {'D_mean': 0.024845327,
                   'D_var': 0.000584181,
                   'alpha_mean': 0.035221736,
                   'alpha_var': 0.001891723}]

    exp4_param = [{'D_mean': 1.008246831,
                   'D_var': 0.288532552,
                   'alpha_mean': 0.896964821,
                   'alpha_var': 0.068297407},  

                  {'D_mean': 0.386410813,
                   'D_var': 0.250142469,
                   'alpha_mean': 0.438131816,
                   'alpha_var': 0.084724388},

                  {'D_mean': 0.025417135,
                   'D_var': 0.001219208,
                   'alpha_mean': 0.023823026,
                   'alpha_var': 0.001463282}]

    exp5_param = [{'D_mean': 0.620735179,
                   'D_var': 0.073157356,
                   'alpha_mean': 0.972160622,
                   'alpha_var': 0.013363848}]

    exp6_param = [{'D_mean': 0.55313097,
                   'D_var': 0.126512872,
                   'alpha_mean': 0.988764897,
                   'alpha_var': 0.032290914}]

    exp7_param = [{'D_mean': 0.550102248,
                   'D_var': 0.036625373,
                   'alpha_mean': 1.399221747,
                   'alpha_var': 0.040378079},

                  {'D_mean': 0.353859405,
                   'D_var': 0.007802151,
                   'alpha_mean': 0.923866077,
                   'alpha_var': 0.077178163},

                  {'D_mean': 0.074279597,
                   'D_var': 0.011913283,
                   'alpha_mean': 0.209474589,
                   'alpha_var': 0.081382054}]

    exp8_param = [{'D_mean': 0.505607985,
                   'D_var': 0.135533248,
                   'alpha_mean': 0.763779936,
                   'alpha_var': 0.112317361}]

    exp9_param = [{'D_mean': 0.650941998,
                   'D_var': 0.296086579,
                   'alpha_mean': 0.852957992,
                   'alpha_var': 0.115184626}]

    exp10_param = [{'D_mean': 7.124892254,
                    'D_var': 21.49053148,
                    'alpha_mean': 0.891913826,
                    'alpha_var': 0.234179804}]

    exp11_param = [{'D_mean': 2.280235516,
                    'D_var': 1.328984755,
                    'alpha_mean': 1.931062842,
                    'alpha_var': 0.002604257},

                   {'D_mean': 1.256172632,
                    'D_var': 0.297456058,
                    'alpha_mean': 1.7350064,
                    'alpha_var': 0.015110158},

                  {'D_mean': 0.080768361,
                   'D_var': 0.018090337,
                   'alpha_mean': 0.142352564,
                   'alpha_var': 0.077370859}]
    
    ALL_ensemble_distributions = [exp0_param, exp1_param, exp2_param, exp3_param, exp4_param, exp5_param, exp6_param, exp7_param, exp8_param, exp9_param, exp10_param, exp11_param]
    # All Ensemble Dists #
    
    def gen_param(param, param_type='float'):
        if type(param) == tuple:
            if param_type == 'float':
                return np.random.uniform(*param)
            else:
                return np.random.randint(*param)
        else:
            return param
    
    # Single state     
    ensemble_distributions = random.choice(ALL_ensemble_distributions)    # pick a random ens dist
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
    
    # Multi state
    ensemble_distributions = random.choice(ALL_ensemble_distributions)    # pick a random ens dist
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
    
    # Dimerization
    ensemble_distributions = random.choice(ALL_ensemble_distributions)    # pick a random ens dist
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
    
    # Confinement
    ensemble_distributions = random.choice(ALL_ensemble_distributions)    # pick a random ens dist
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
    
    # Immobile traps
    ensemble_distributions = random.choice(ALL_ensemble_distributions)    # pick a random ens dist
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
    
    return [dict_single_state, dict_multi_state, dict_dimerization, dict_confinement, dict_immobile_traps]
########## \MAKE DICTIONARIES ##########

########## MAKE DATASET ##########
def GenerateTrueData_M(N=10000, return_padding_mask=False, max_traj_len=200):
    all_trajs = np.zeros((N,max_traj_len,2))
    if return_padding_mask:
        padding_mask = np.full((N,max_traj_len), True)
    all_labels = np.zeros((N,max_traj_len,8))
    traj_idx = -1

    break_flag = False
    while True:
        
        ### WEIRD CODE WEIRD CODE WEIRD CODE WEIRD CODE ###  --> here as the AnDi code sometimes glitches!
        while True:
            try:
                ## Generate FOVs, convert to list of numpy arrays
                FOVs, _, _ = challenge_phenom_dataset(dics=Gen_Random_Model_Param_Dict(N=100), return_timestep_labs=True)
            except:
                print('FOV gen failure; trying again.')
                continue
            else:
                break
        ### WEIRD CODE WEIRD CODE WEIRD CODE WEIRD CODE ###
        FOVs = [FOV.to_numpy() for FOV in FOVs]
        FOV_model_labels = np.arange(5)    # models have labels 0,1,2,3,4
        
        ## Split FOVs up into different trajs, and add padding as appropriate ##
        for fov_idx, (FOV, FOV_model_label) in enumerate(zip(FOVs, FOV_model_labels)):
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

                all_labels[traj_idx][first_frame:last_frame+1, :3]= traj[:,4:]    # drop in the labels for alpha, K and motion type
                all_labels[traj_idx][first_frame:last_frame+1, 3 + FOV_model_labels[fov_idx]]= 1    # drop in the labels for model
                
                all_labels[traj_idx][:first_frame, -1] = 1    # pad the model labels so out of FOV = trapped at edge
                all_labels[traj_idx][last_frame:, -1] = 1    # pad the model labels so out of FOV = trapped at edge
                
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
