import numpy as np

def ZeroOneNorm(data, min_bound, max_bound, reverse=False):
    data_range = max_bound - min_bound
    if reverse is True:
        return (data*data_range) + min_bound
    return (data - min_bound)/data_range

def ZeroOneNormLabels(Labels, K_min=10e-12, K_max=10e6, alpha_min=0, alpha_max=2, reverse=False):
    Labels = Labels.copy()
    Labels[:,:,0] = ZeroOneNorm(Labels[:,:,0], 10e-12, 10e6, reverse)
    Labels[:,:,1] = ZeroOneNorm(Labels[:,:,1], 0, 2, reverse)
    return Labels

def ExplicitCPs(Labels):
    '''
    Accept labels, return expliticly labelled CPs + labels
    '''
    # Work out CP locations
    labs_diff = Labels[:,1:] - Labels[:,:-1]
    labs_diff_sum = np.sum(abs(labs_diff), axis=2)
    CPs = np.where(labs_diff_sum != 0, 1, 0)
    # Prep to concatenate to labels
    zeros = np.zeros((np.shape(CPs)[0], 1))
    CPs = np.concatenate([zeros, CPs], axis=1)
    CPs = np.expand_dims(CPs, 2)
    # Concatenate to labels
    Labels = np.concatenate([CPs, Labels], axis=2)
    
    return Labels


def ScaleAndZeroTrajs(Trajs):
    "Get all trajs to start from 0 and scale them all to the interval [-4,4]"
    ZeroedTrajs = Trajs - Trajs[:,:1]
    ScaledAndZeroedTrajs = ZeroedTrajs/32    # [-128,128]/32 -> [-4,4]
    return ScaledAndZeroedTrajs

def DiffTrajs(Trajs):
    "Diff and add zeros to the end to preserve shape"
    DiffedTrajs = np.diff(Trajs, axis=1)
    DiffedTrajs = np.concatenate([DiffedTrajs, np.zeros_like(DiffedTrajs[:,-1:])], axis=1)
    return DiffedTrajs

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