""" random_env_gen.py

Use cellular automata to generate a series of mazes for planning on

"""
import importlib
import os
import sys
import shutil
import secrets

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from cellular_automata import CellAutomata
from random_env import RandomEnv
from tqdm import tqdm
import json

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
    print(len(env['obstacles']),len(env['obstacles'])/(world_size/cell_size)**3)
    env['map_complexity'] = len(env['obstacles'])/((world_size/cell_size)**3 - len(env['obstacles']))
    with open(f'{output_folder}/{seed}_env.json', "w") as outfile:
        json.dump(env, outfile)
    save_img(map_gen_result, ca, f'{output_folder}/{seed}.png',)

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
    birth = 10 
    death = 2
    init_prob = 5
    steps = 1
    neighbor_type = 'moore'
    num_maps = 3

    rng = np.random.default_rng(start_seed)
    seeds = rng.integers(low=0, high=1_000_000_000, size=num_maps)

    run_name = f'W{world_size}_C{cell_size}_B{str(birth).zfill(2)}_D{str(death).zfill(2)}_I{init_prob}_S{steps}_{"M" if neighbor_type=="moore" else "N"}'
    output_folder = 'SeqManifold/v7/PlanningOnManifolds/cellular_automata/random_worlds/' + run_name
    print(os.getcwd())
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # with open(output_folder + '/gen_info.txt', 'w') as f:
    #     f.write(f'start_seed: {start_seed}\nseeds: {seeds}')

    # shutil.copy(__file__, output_folder / os.path.basename(__file__))

    robot_params = dict(type='box',
                        sides = [1.0, 1.0, 1.0],
                        position = [0, 0, 0],
                        orient_rpy = [0, 0, 0])

    for i in tqdm(range(num_maps)):
        # run_folder = output_folder / str(i).zfill(len(str(num_maps)))

        generate_environment(seeds[i], world_size, cell_size,
                             birth, death, init_prob/100.0, steps,
                             neighbor_type, robot_params, output_folder)