import os
import re
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import stats
import pandas
from tqdm.auto import tqdm
import warnings
from pathlib import Path
from andi_datasets.models_phenom import models_phenom
from andi_datasets.utils_challenge import *
import sys
sys.path.append('/home/cs-solomon.asghar/AnDi_2024/software/')
from MakePreds import *

def CodaLab_Eval(submit_dir, truth_dir, output_path):
    '''
    Better version of AnDi scoring function.
    submit_dir: where the res file is
    truth_dir: where the ref file is
    output_path: desired names and locs of outputs 
    '''
    # Starting the HMTL file
    htmlOutputDir = f"{output_path}_html"
    if not os.path.exists(htmlOutputDir):
            os.makedirs(htmlOutputDir)
    html_filename = os.path.join(htmlOutputDir, 'scores.html')
    html_file = open(html_filename, 'a', encoding="utf-8")
    html_file.write('<h1>Submission detailed results </h1>')

    if not os.path.isdir(submit_dir):
        print( "%s doesn't exist", submit_dir)
        

    output_filename = f'{output_path}.txt'
    output_file = open(output_filename, 'w')
       
    # Track 1: videos
    # Track 2: trajectories
    for track, name_track in zip([1,2], ['videos', 'trajectories']):
        
        ##### ----- In case the whole track is missing, give Nones to both tasks ----- #####
        path_preds = f'{submit_dir}/track_{track}'
        
        if not os.path.exists(path_preds):
            wrn_str = f'No submission for track {track} found.'
            warnings.warn(wrn_str)
            
            for task in enumerate(['single', 'ensemble']): 
                # Codalab naming:
                # task 1 : single traj
                # task 2: ensemble
                idx_task = 1 if task == 'single' else 2
                
                # single trajectories
                if task == 'single':                    
                    for name, max_error in zip(['alpha','D','state', 'cp','JI'], list(_get_error_bounds()[:-2])+[0]): # This names must be the same as used in the yaml leaderboard                  
                        output_file.write(f'tr{track}.ta1{idx_task}.'+name+': '+str(max_error) +'\n')
                elif task == 'ensemble':
                    for name, max_error in zip(['alpha','D'], _get_error_bounds()[-2:]): # This names must be the same as used in the yaml leaderboard
                        output_file.write(f'tr{track}.ta{idx_task}.'+name+': '+str(max_error) +'\n')
            continue
        ##### ------------------------------------------------------------------------ #####
        
        
        html_file.write(f'<h2> Track {track}: '+name_track+' </h2>')

        for task in ['ensemble', 'single']: 

            # Codalab naming:
            # task 1 : single traj
            # task 2: ensemble
            idx_task = 1 if task == 'single' else 2
            
            if task == 'single':
                html_file.write(f'<h3> Single Trajectory Task </h3>')
            elif task == 'ensemble':
                html_file.write(f'<h3> Ensemble Task </h3>')


            # Get the number of experiments from the true directory
            exp_folders = sorted(list(listdir_nohidden(truth_dir+f'/track_{track}')))
            exp_nums = [int(re.findall(r'\d+', name)[0]) for name in exp_folders]

            if task == 'single':  

                avg_metrics, df = run_single_task(exp_nums, track, submit_dir, truth_dir )

                for name, res in zip(['cp','JI','alpha','D','state'], avg_metrics): # This names must be the same as used in the yaml leaderboard                  
                    output_file.write(f'tr{track}.ta{idx_task}.'+name+': '+str(res) +'\n')

                ''' To keep consistency with leaderboard display, we swap the K and alpha columns that
                get printed in the detailed results.
                Moreover, we change the names to match leaderboard. '''
                df_swapped = df.iloc[:,[0,1,2,3,5,4,6]]
                df_swapped = df_swapped.rename(columns = {'alpha': 'MAE (alpha)', 'K': 'MSLE (K)',
                                                          'RMSE CP': 'RMSE (CP)', 'JSC CP': 'JSC (CP)',
                                                          'state': 'F1 (diff. type)'})
                # Changing the name of JI to JSC to match paper nomenclature
                html_file.write(df_swapped.to_html(index = False).replace('\n',''))
              

            if task == 'ensemble':

                avg_metrics, df = run_ensemble_task(exp_nums, track, submit_dir, truth_dir)
                
                ''' There was a problem with the leaderboard labels and we had to SWAP alpha and D in the 
                first element of the zip, i.e. the list is now ['D', 'alpha'] but avg_metrics is [alpha, D] '''
                for name, res in zip(['D','alpha'], avg_metrics):                
                    output_file.write(f'tr{track}.ta{idx_task}.'+name+': '+str(res) +'\n')      

                ''' To keep consistency with leaderboard display, we swap the K and alpha columns that
                get printed in the detailed results.
                Moreover, we change the names to match leaderboard. '''
                df_swapped = df.iloc[:,[0,2,1]]                
                df_swapped = df_swapped.rename(columns = {'alpha': r'W1 (alpha)', 'K': 'W1 (K)'})
                
                html_file.write(df_swapped.to_html(index = False).replace('\n',''))
   

    html_file.close()
    output_file.close()


def EvaluateModel(model, mode, max_traj_len=224, min_peak_height=0.25, use_seg_nets=False):
    if mode == 'SK':
        data_path = '/home/cs-solomon.asghar/AnDi_2024/Evaluation/StartingKitEval/ref/'
        output_name_template = "/res_sk/"
        eval_output_name = '/SK'
        local_eval_mode = False
    elif mode == 'LE':
        data_path = '/home/cs-solomon.asghar/AnDi_2024/Evaluation/LocalEval/ref/'
        output_name_template = "/res_le/"
        eval_output_name = '/LE'
        local_eval_mode = True
    elif mode == 'LCD':
        data_path = '/home/cs-solomon.asghar/AnDi_2024/Evaluation/LocalEval_ES/ref/'
        output_name_template = "/res_LCD/"
        eval_output_name = '/LCD'
        local_eval_mode = True

    # Make Preds
    All_Traj_Addresses, Segments_by_Size, Segment_Addresses_by_Size, CPs, num_exps, num_fovs = PhaseOnePredictions(model, data_path, max_traj_len=max_traj_len, min_peak_height=min_peak_height)
    
    if use_seg_nets:
        output_name_template = output_name_template + '_segs'
    
    PhaseTwoPredictions(All_Traj_Addresses, Segments_by_Size, Segment_Addresses_by_Size, CPs, 
                        num_exps, num_fovs,
                        output_name=output_name_template, use_seg_nets=use_seg_nets, local_eval_mode=local_eval_mode)

    # Evaluate and Save results
    submit_dir = os.getcwd() + output_name_template
    output_path = os.getcwd() + eval_output_name
    CodaLab_Eval(submit_dir, data_path, output_path)