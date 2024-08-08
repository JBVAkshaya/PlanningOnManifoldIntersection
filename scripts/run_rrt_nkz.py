#! /usr/bin/python3

from algorithms.rrt_nkz import RRTStarManifold as rrt
import numpy as np
import plotly.graph_objects as go
import time
import logging
from joblib import Parallel, delayed

from glob import glob
import json
import os

import argparse
import importlib

logging.basicConfig(level=logging.INFO)
np.random.seed(0)

argParser = argparse.ArgumentParser()
argParser.add_argument("-i",
                       "--import_module",
                       help="experiment config file",
                       default="I_config")

args = argParser.parse_args()

config = importlib.import_module(".".join(
    ["experiment_configs", args.import_module]))

file_names = glob('results/maps/*.json')
# file_names = [file_names[-1]]
output_folder = f'results/RAL/NKZ/{config.struct_type}_{config.num_robots}'


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

side_len = 2 * ((config.rob_specs['b_x'] / 2.0) + config.rob_specs['l_1'] +
                config.rob_specs['l_2'])


def process(f_name):
    try:
        print(f_name)
        data = {}
        with open(f_name, 'r') as f:
            data = json.load(f)

        env_params = {
            'obstacles': data['obstacles'],
            'start': data['start'],
            'goal': data['goal'],
            'struct_size': [side_len, side_len, side_len],
            'obs_size': [1.5, 1.5, 1.5],
            'obs_z_translation': 0.0  #data['bounds']['z_bounds'][1]/2.0
        }

        #### Set this based on Start and Goal cells ####
        robot_limits = {
            'pos': {
                'min': [
                    float(env_params['start'][0] - 4 * side_len),  #x
                    float(env_params['start'][1] - 4 * side_len),
                    float(data['bounds']['z_bounds'][0])
                ],  #y
                'max': [
                    float(env_params['goal'][0] + 4 * side_len),
                    float(env_params['goal'][1] + 4 * side_len),
                    float(data['bounds']['z_bounds'][1])
                ]
            },
            'angles': {
                'min': [
                    np.round(0 * 180. / np.pi, 0),  #yaw: 0
                    np.round(0. * 180. / np.pi, 0),  #q1:0
                    np.round(0. * 180. / np.pi, 0)
                ],  #q2:0
                'max': [
                    np.round(1.57 * 180. / np.pi, 0),  # 90
                    np.round(0.78 * 180. / np.pi, 0),  # 45
                    np.round(0.78 * 180. / np.pi, 0)  #45
                ]
            }
        }

        output_file = output_folder + '/' + f_name.split('/')[-1]
        # print('out: ', output_file)
        if not os.path.exists(output_file):
            try:
                data = {
                    'num_robots': config.robot_params['num_bots'],
                    'struct_type': config.robot_params['struct_type'],
                    'n_target_nodes': config.rrt_params['n_samples'],
                    'dofs': config.robot_params['dof'],
                    'proj_eps': config.rrt_params['proj_eps'],
                    'obstacles': env_params['obstacles'],
                    'robot_size': env_params['struct_size'],
                    'obs_size': env_params['obs_size'],
                    'min_length': config.robot_params['min_len'],
                    'beta': config.rrt_params['beta'],
                    'path': [],
                    'n_actual_nodes': 0,
                    'time_taken': 0
                }
                rand_num = np.random.randint(10)
                # print(str(i), "th time")

                np.random.seed(rand_num)
                t_bots = rrt(rob_specs=config.rob_specs,
                             robot_limits=robot_limits,
                             robot_params=config.robot_params,
                             constraint_params=config.constraint_params,
                             rrt_params=config.rrt_params,
                             env_params=env_params,
                             plot_params=config.plot_params)
                # print("t bots")
                data_final = {}
                data_final['results'] = t_bots.task_space_run(data)

                if len(data_final['results']['path']) > 0:
                    data = json.dumps(data_final, cls=NumpyEncoder)
                    with open(output_file, "w") as outfile:
                        outfile.write(data)
                        outfile.close()
                else:
                    return False
                return True
            except:
                return False
        else:
            return False
    except:
        logging.error('data file invalid')
        return False


if config.debug_flag == False:
    results = Parallel(n_jobs=13)(delayed(process)(file_names[i])
                                  for i in range(0, len(file_names)))
else:
    results = [process(file_names[i]) for i in range(0, len(file_names))]
