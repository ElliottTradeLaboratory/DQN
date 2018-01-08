import os
import sys
import re
import time
from datetime import datetime
import numpy as np
from collections import OrderedDict

import tensorflow as tf

import torch
from torch.utils.serialization import load_lua
sys.path.append('./')
from dqn.visualizer import *

config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

logdir = '/home/deeplearning/temp/etl_dqn/dqn3.0/{}'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
param_dir = '/home/deeplearning/work/dqn3.0_work/logs'
layer_names = ['Conv1', 'relu1', 'Conv2', 'relu2', 'Conv3', 'relu3', 'Linear', 'relu4', 'q_all']
p_names = ['weight','bias'] * (len(layer_names))

last_numSteps = 0
last_output_numSteps = 0
last_eval_numSteps = 0

def get_last_number(cur_last_number, prefix):
    files = os.listdir(param_dir)
    files.sort()

    print(prefix + ' last numbrt:',  cur_last_number)
    numSteps_list = []
    for file_name in files:
        if re.search('{}.*'.format(prefix), file_name):
            num = int(re.search(r"\d+",file_name).group())
            if num > cur_last_number:
                numSteps_list.append(num)
            
        if len(numSteps_list) > 10:
            break

    numSteps_list = np.array(numSteps_list)

    after_cur_last_num_steps = numSteps_list[numSteps_list > cur_last_number]

    after_cur_last_num_steps.sort()

    return after_cur_last_num_steps

def remove(path, prefix, remove_steps):
    files = os.listdir(path)
    for file_name in files:
        if re.search('{}.*20000000'.format(prefix), file_name):

def get_params_outputs():

    global last_numSteps, last_eval_numSteps, last_output_numSteps

    #
    # Find to first filename for visualization(current_state, params, outputs)
    #

    batch = OrderedDict()
    getQUpdate = OrderedDict()
    outputs = OrderedDict()
    optimizes = OrderedDict()
    current_state = OrderedDict()
    evaluation = OrderedDict()
    
    #
    # data loading
    # 
    numSteps_list = get_last_number(last_numSteps, 'current_state')
    if len(numSteps_list) > 0:
        last_numSteps = numSteps_list[-1]
        for numSteps in numSteps_list:
            cur_filename = os.path.join(param_dir, 'current_state_{:010d}.dat'.format(numSteps))
            print('load:'+cur_filename)
            current_state[numSteps] = load_lua(cur_filename)
        if last_numSteps > 20000000:
            


    numSteps_list = get_last_number(last_output_numSteps, 'batch_')
    if len(numSteps_list) > 0:
        last_output_numSteps = numSteps_list[-1]
        for numSteps in numSteps_list:
            params_filename = os.path.join(param_dir, 'batch_{:010d}.dat'.format(numSteps))
            if os.path.exists(params_filename):
                print('load:'+params_filename)
                batch[numSteps] = load_lua(params_filename)

            params_filename = os.path.join(param_dir, 'getQUpdate_{:010d}.dat'.format(numSteps))
            if os.path.exists(params_filename):
                print('load:'+params_filename)
                values= load_lua(params_filename)
                values = [v.numpy() for v in values]
                getQUpdate[numSteps] = values

            

            network = {}
            target_network = {}
            for lnm in layer_names:
                outputs_filename = os.path.join(param_dir, 'network_vars_{:010d}_{}.dat'.format(numSteps, lnm))
                outputs_filename2 = os.path.join(param_dir, 'network_vars_{:010d}_{}.dat'.format(numSteps, lnm.lower()))
                if os.path.exists(outputs_filename):
                    print('load:'+outputs_filename)
                    params = load_lua(outputs_filename)
                    network[lnm] = [v.numpy() for v in params]
                elif os.path.exists(outputs_filename2):
                    print('load:'+outputs_filename2)
                    params = load_lua(outputs_filename)
                    network[lnm] = [v.numpy() for v in params]
                os.remove(outputs_filename)
                outputs_filename = os.path.join(param_dir, 'target_network_vars_{:010d}_{}.dat'.format(numSteps, lnm))
                outputs_filename2 = os.path.join(param_dir, 'target_network_vars_{:010d}_{}.dat'.format(numSteps, lnm.lower()))
                if os.path.exists(outputs_filename):
                    print('load:'+outputs_filename)
                    params = load_lua(outputs_filename)
                    target_network[lnm] = [v.numpy() for v in params]
                elif os.path.exists(outputs_filename2):
                    print('load:'+outputs_filename2)
                    params = load_lua(outputs_filename)
                    target_network[lnm] = [v.numpy() for v in params]

            if len(network) >0:
                outputs[numSteps] = (network, target_network)

    #
    # Find to first filename for visualization(evaluation)
    #
    numSteps_list = get_last_number(last_eval_numSteps, 'evaluation')
    if len(numSteps_list) > 0:
        last_eval_numSteps = numSteps_list[-1]
        for numSteps in numSteps_list:
            eval_filename = os.path.join(param_dir, 'evaluation_{:010d}.dat'.format(numSteps))
            print('load:'+eval_filename)
            evaluation[numSteps] = load_lua(eval_filename)
    else:
        evaluation = {}


    print('exit', 'current_state:', len(current_state), 'batch:', len(batch), 'getQUpdate', len(getQUpdate), 'outputs:', len(outputs), 'numSteps_list:', len(numSteps_list))
    return current_state, batch, getQUpdate, outputs, evaluation, numSteps_list



def main():

    current_state_logger = CurrentStateVisualizer(logdir)
    learning_logger = LearningVisualizer(logdir, layer_names)
    eval_logger = EvaluationVisualizer(logdir)
    
    while True:

        time.sleep(5)

        current_state, step_batch, step_getQUpdate, step_outputs, evaluation, numSteps_list = get_params_outputs()

        print('-----curstate----')
        for numSteps, states in current_state.items():
            print('global_step', numSteps)
            current_state_logger.addCurrentState(states)
            current_state_logger.flush(numSteps)


        for numSteps, outputs in step_outputs.items():
            print('outputs', numSteps)
            batch = step_batch[numSteps]
            s = batch[0].numpy().reshape(32, 4, 84, 84, 1)
            s2 = batch[3].numpy().reshape(32, 4, 84, 84, 1)
            learning_logger.addInputImages(2, s[0])
            learning_logger.addInputImages(3, s2[0])

            network_vars, target_network_vars = outputs

            learning_logger.addNetworkParameters(network_vars)
            learning_logger.addTargetNetworkParameters(target_network_vars)
            
            getQUpdate = step_getQUpdate[numSteps]
            dict_getQUpdate = dict(zip(learning_logger.GET_Q_UPDATE_VALUE_NAMES, getQUpdate))
            learning_logger.addGetQUpdateValues(dict_getQUpdate)

            learning_logger.flush(numSteps)

        for numSteps, evals in evaluation.items():

            print('evaluation', numSteps)

            values = {'episode_score':np.array(evals['episode_scores']),
                       'episode_count':len(evals['episode_scores']) }
            eval_logger.addEvaluation(values)
            eval_logger.addValidation({'TDerror':evals['TD_error'], 'V':evals['V']})
            eval_logger.flush(numSteps)

if __name__=='__main__':
    main()
