""" random_env_gen.py

Use cellular automata to generate a series of mazes for planning on

"""
import importlib
import os
import struct
import sys
import shutil
import secrets

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from cellular_automata.cellular_automata import CellAutomata
from cellular_automata.random_env import RandomEnv
from tqdm import tqdm
import json
import cellular_automata.env_utils as ca_utils
from lib.util import ee_position, ee_orient
import lib.plot as plt

from lib.environment import Environment
import fcl


sys.setrecursionlimit(1500)

def generate_environment(seed: int,
                         world_size:int,
                         cell_size:float,
                         birth:int,
                         death:int,
                         init_prob:float,
                         steps:int,
                         neighbor_type:str,
                         robot_params:dict,
                         output_folder:str):
    """
    Generates a random environment given the parameters

    Parameters
    ----------
    seed: int
        the random seed used to initialize the random number generator
    world_size: int
        size of the world, assuming cube world with each axis the same
    cell_size: float
        the size of each cell
    birth: int
        if there are at least this number of wall neighbors 
        a wall tile will be 'born'
    death: int
        if there are less than this number of wall neighbors 
        the wall tile will 'die' and become free space
    init_prob: float
        probability of a cell being a wall initially
    steps: int
        number of steps to run the cellular automata
    neighbor_type: str
        type of neighboorhood used for CA rules
            'moore' - use Moore Neighborhood of 26 surrounding tiles
            'neumann' - use von Neumann neighborhood of adjacent 6 tiles
    robot_params: dict
        parameter dictionary for robot shape/size
    output_folder: str
        the path of the output folder
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    gen_params = dict(seed=seed,
                      world_size = (world_size, world_size, world_size),
                      cell_size = cell_size,
                      ca_prop = dict(
                        init_prob=init_prob,
                        neighbor_type=neighbor_type,
                        birth = birth, # form a wall if cell is surrounded by >birth neighbors
                        death = death, # tear down a wall if a cell is surround by <death neighbors
                        steps = steps,
                      ))
    ca = CellAutomata(gen_params, show_tqdm=False, debug=False)
    map_gen_result = ca.generate_map()
    # print (map_gen_result)
    env = ca.create_env(map_gen_result, robot_params)

    if len(env['obstacles']) > 9:
        print(len(env['obstacles']),len(env['obstacles'])/(world_size/cell_size)**3)
        env['map_complexity'] = len(env['obstacles'])/((world_size/cell_size)**3 - len(env['obstacles']))
        
        # save_img(map_gen_result, ca, f'{output_folder}/{seed}.png',)
        robot_params['type']='UVMS'
        fig, in_collision = save_plotly_img(env['obstacles'],env['start'],env['goal'], robot_params=robot_params)
        if in_collision==False:
            fig.write_image(f'{output_folder}/{seed}.png')
            with open(f'{output_folder}/{seed}_env.json', "w") as outfile:
                json.dump(env, outfile)


def save_plotly_img(obstacles,start,goal, robot_params):
    start_conf_state = ca_utils.get_state(cell_pose=start, num_rob= robot_params['num_robots'], struct_type = robot_params['struct_type'], min_len=0.5)
    start_task_state = ee_position(dofs=robot_params['dof'],q=start_conf_state, rob_specs=robot_params)
    goal_conf_state = ca_utils.get_state(cell_pose=goal, num_rob= robot_params['num_robots'], struct_type = robot_params['struct_type'], min_len=0.5)
    goal_task_state = ee_position(dofs=robot_params['dof'],q=goal_conf_state, rob_specs=robot_params)
    
    data = [{'task_conf': np.reshape(start_task_state,(robot_params['num_robots'],3)), 
            'orient': ee_orient(robot_params['dof'],start_conf_state,rob_specs=robot_params)},
            {'task_conf': np.reshape(goal_task_state,(robot_params['num_robots'],3)), 
            'orient': ee_orient(robot_params['dof'],goal_conf_state,rob_specs=robot_params)}]
    
    ## fcl environment
    fcl_env = Environment()
    mesh_list = fcl_env.get_obstacle_objs(obstacles, obs_three_d = True, z_translation = 0.0)

    start_struct = fcl_env.get_structure_objs(np.reshape(start_task_state,(robot_params['num_robots'],3)))
    goal_struct = fcl_env.get_structure_objs(np.reshape(goal_task_state,(robot_params['num_robots'],3)))
    
    request_collide = fcl.CollisionRequest()
    result_collide = fcl.CollisionResult()
    in_collision = False
    for _,obs in mesh_list.items():
        for _,struct_piece in start_struct.items():
            ret_1 = fcl.collide(obs, struct_piece, request_collide, result_collide)
            print(f'ret_1: status: ', ret_1, result_collide.is_collision)
            if result_collide.is_collision == True:
                in_collision = True
                break
    
    if in_collision==False:
        for _,obs in mesh_list.items():
            for _,struct_piece in goal_struct.items():
                ret_1 = fcl.collide(obs, struct_piece, request_collide, result_collide)
                print(f'ret_1: status: ', ret_1, result_collide.is_collision)
                if result_collide.is_collision == True:
                    in_collision = True
                    break
    print('in_collision:', in_collision)

    fig = plt.plot(env=fcl_env, 
            data = data, 
            mesh_list = mesh_list, 
            obs_size = robot_params['obs_size'], 
            path=True, 
            num_robots = robot_params['num_robots'], 
            struct_type =robot_params['struct_type'])

    return fig, in_collision

def save_img(map_gen_result:dict,
             ca: RandomEnv,
             output_file : str,
             color : tuple = (0.0, 0.7, 0.7, 0.8)):
    """
    Saves the image of a generated environment to a file

    Parameters
    ----------
    map_gen_results: dict
        result dictionary that is the output of generate_map()
        with 'final_cells' key that maps to cells with obstacles
    ca: RandomEnv
        random environment generator object that created map_gen_results
        in this case, a CellAutomata object
    output_file: str
        the path/filename of the output image
    color: tuple
        rgba color of the obstacle in render
    """
    cells = map_gen_result['final_cells']
    color = (0.0, 0.7, 0.7, 0.8)
    fig_num = 0
    if plt.fignum_exists(fig_num):
        plt.close()
    fig = plt.figure(fig_num)
    ax = fig.add_subplot(projection='3d')
    ca.plot_cells_plt(cells, ax, color)
    plt.show()
    plt.savefig(output_file)


if __name__=="__main__":


    start_seed = secrets.randbelow(1_000_000)
    world_size = 18   
    cell_size = 1.5
    birth = 12
    death = 2
    init_prob = 4.5
    steps = 1
    neighbor_type = 'moore'
    num_maps = 100

    rng = np.random.default_rng(start_seed)
    seeds = rng.integers(low=0, high=1_000_000_000, size=num_maps)

    output_folder = 'results/random_worlds/'
    print(os.getcwd())
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    robot_params = dict(type='box',
                        sides = [1.0, 1.0, 1.0],
                        position = [0, 0, 0],
                        orient_rpy = [0, 0, 0],
                        num_robots = 3,
                        struct_type = 'S',
                        min_len = 0.5,
                        b_x=0.6,
                        b_y=0.6,
                        b_z=0.6,
                        l_1=0.3,
                        l_2=0.1,
                        dof= ['x', 'y', 'z', 'yaw', 'q1', 'q2'], # independent of algo hence to be edited here
                        obs_size= [1.5,1.5,1.5])

    for i in tqdm(range(num_maps)):

        generate_environment(seeds[i], world_size, cell_size,
                             birth, death, init_prob/100.0, steps,
                             neighbor_type, robot_params, output_folder)