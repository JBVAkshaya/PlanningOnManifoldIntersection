""" random_env.py

Base class for random environment generation

"""
from typing import List

import matplotlib
import plotly.graph_objects as go
import yaml
import numpy as np

class RandomEnv(object):

    def __init__(self, gen_params:dict,
                       show_tqdm : bool = False,
                       tqdm_offset : int = 0,
                       debug : bool = False):
        """
        Constructor for Cellular Automata

        Parameters
        ----------
        gen_params: dict
            dictionary of parameters used for generating a world
            'seed': int
                the random seed used to initialize the random number generator
            'world_size': tuple
                (x, y, z) the size of the world in each axis
            'cell_size': int
                the size of each cell
        show_tqdm: bool
            if True, shows intermediate steps of maze generation using tqdm
        tqdm_offset: int
            offset tqdm progress bar by this amount for nested progress bars
        debug: bool
            if True, print debug statements
        """
        self.debug = debug
        self.show_tqdm = show_tqdm
        self.tqdm_offset = tqdm_offset
        self.gen_params = gen_params
        self.seed = gen_params['seed']
        self.rng = np.random.default_rng(self.seed)

        self.world_size = gen_params['world_size']
        self.cell_size = gen_params['cell_size']
        self.cell_array_shape = np.ceil(np.array(self.world_size) / self.cell_size).astype('int')

        # self.cells is an array where True means there is a wall/obstacle at that location
        self.cells = self.init_cells()

    def init_cells(self):
        """
        Initializes self.cells using parameters in self.gen_params
        """
        raise NotImplementedError('Implement this in each respective child class')

    def generate_map(self):
        """
        Generates a map based on self.gen_params 
        Creates an environment object based on the map

        Returns
        -------
        result: dict
            dictionary containing results from maze generation
            'final_cells': np.ndarray
                final cell array
            'start_pos': np.ndarray (12,)
                vehicle starting pose
            'goal_pos': np.ndarray (12,)
                vehicle goal pose
        """
        raise NotImplementedError('Implement this in each respective child class')

    def create_env(self, map_gen_result:dict, robot_params:dict):
        """
        Creates an Environment object with obstacles based 
        results from 
        
        """
        cells = map_gen_result['final_cells']
        env_params = {}

        # load the robot params to dict
        env_params['robot'] = robot_params

        # load bounds to dict
        env_params['bounds'] = {}
        buffer = 0.0
        env_params['bounds']['x_bounds'] = [0.0-buffer, self.world_size[0]+buffer]
        env_params['bounds']['y_bounds'] = [0.0-buffer, self.world_size[1]+buffer]
        env_params['bounds']['z_bounds'] = [0.0-buffer, self.world_size[2]+buffer]
        env_params['bounds']['yaw_bounds'] = [-np.pi, np.pi]

        # load the filled cells of the environment as obstacles to dict
        obstacles = []
        for i in range(cells.shape[0]):
            for j in range(cells.shape[1]):
                for k in range(cells.shape[2]):
                    # If there is an obstacle at that cell location
                    # draw an obstacle there
                    if cells[i,j,k]:
                        pos = self.cell_index_to_world_pos(np.array([i, j, k]))
                        obs_dict = dict(
                            type = 'box',
                            sides = [self.gen_params['cell_size'] for _ in range(3)],
                            position = self.cell_index_to_world_pos(np.array([i, j, k])).tolist(),
                            orient_rpy = [0, 0, 0]
                        )
                        obstacles.append(obs_dict)
        env_params['obstacles'] = obstacles

        # load the start/goal to dict
        env_params['start'] = map_gen_result['start_pos'].tolist()
        env_params['goal'] = map_gen_result['goal_pos'].tolist()

        # Create an environment using this information
        # env = Environment.from_dict(env_params)
        return env_params


    def save_to_yaml(self, filename: str, map_gen_result:dict, robot_params:dict):
        """
        
        """
        # Create an environment using this information
        env = self.create_env(map_gen_result, robot_params)

        # Save this environment to file
        env.to_yaml(filename)

    def cell_index_to_world_pos(self, idx: np.ndarray) -> np.ndarray:
        """
        Computes the world position from index in cell array

        Parameters
        ----------
        idx: np.ndarray of np.int (3,)
            index of cell in array
        
        Returns
        -------
        world_pos: np.ndarray (3,)
            the position in the world corresponding to the cell
        """
        return (idx * self.cell_size) + np.ones(3)*self.cell_size/2.0

    def world_pos_to_cell_index(self, world_pose) -> np.ndarray:
        """
        Computes index in cell array from the world position 

        Parameters
        ----------
        world_pos: np.ndarray (3,)
            the position in the world corresponding to the cell
        
        Returns
        -------
        idx: np.ndarray of np.int (3,)
            index of cell in array
        """
        return np.floor(world_pose / self.cell_size)

    def plot_cells_plotly(self, cells: np.ndarray) -> List:
        """
        Plots the given cells using Plotly library

        Parameters
        ----------
        cells: np.ndarray of np.bool
            cells array where True represents there is a wall
            and False means it is empty
        
        Returns
        -------
        traces: List of plotly traces
            a list of plotly traces to be added using go.Figure.add_trace()
        """

        cell_size = self.gen_params['cell_size']
        sides = np.ones((3)) * cell_size

        obstacle_points = []
        for i in range(cells.shape[0]):
            for j in range(cells.shape[1]):
                for k in range(cells.shape[2]):
                    # If there is an obstacle at that cell location
                    # draw an obstacle there
                    if cells[i,j,k]:
                        pos = self.cell_index_to_world_pos(np.array([i, j, k]))
                        obstacle_points.append(gen_bb_points(sides=sides, orient=np.eye(3), pos=pos))
        return plot_obstacles(obstacle_points)

    def plot_cells_plt(self, cells: np.ndarray, ax, color=np.ndarray) -> List:
        """
        Plots the given cells using matplotlib

        Parameters
        ----------
        cells: np.ndarray of np.bool
            cells array where True represents there is a wall
            and False means it is empty

        ax: plt axis
            axis object to plot cells to
        
        color: np.ndarray or str
            color to color the obstacles
        
        Returns
        -------
        traces: List of plotly traces
            a list of plotly traces to be added using go.Figure.add_trace()
        """

        x,y,z = np.indices(np.array(cells.shape)+1)*self.gen_params['cell_size']

        ax.voxels(x, y, z, cells, facecolors=color)
        ax.set(xlabel='x', ylabel='y', zlabel='z')
        return ax

def gen_bb_points(sides : np.ndarray = np.array([1.0, 1.0, 1.0]),
                  orient : np.ndarray = None,
                  pos : np.ndarray = None):
    """
    Generates the bounding box points for a box with
    given side lengths centered at the origin

    Parameters
    ----------
    sides: np.ndarray (3,)
        the x, y, z side lengths
    pos: np.ndarray (3,)
        the position of the box in x,y,z
    orient: np.ndarray (3, 3)
        rotation matrix defining rotation of box

    Returns
    -------
    points: np.ndarray (8, 3)
        points defining the corners of the box

    """
    # Create box centered at the origin with given side lengths
    points = 0.5*np.array([[-1, -1, -1],
                           [ 1, -1, -1],
                           [ 1,  1, -1],
                           [-1,  1, -1],
                           [-1, -1,  1],
                           [ 1, -1,  1],
                           [ 1,  1,  1],
                           [-1,  1,  1]])
    points *= sides

    if orient is not None:
        points = (orient @ points.T).T
    if pos is not None:
        points += pos
    return points

def plot_obstacles(obstacle_points: List) -> List:
    """ 
    Creates plotly graph object traces that visualizes obstacle.

    Parameters
    ----------
    obstacle_points: List of np.ndarray (8, 3)
        List of x,y,z coordinates defining where obstacles are

    Returns
    -------
    traces: List
        a list of plotly graph traces to be added to a figure
    """

    traces = []
    for i, obs_pts in enumerate(obstacle_points):
        traces.append(go.Mesh3d(x= obs_pts[:,0],
                            y= obs_pts[:,1],
                            z= obs_pts[:,2],
                            # i, j and k give the vertices of triangles
                            i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                            j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                            k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                            opacity=0.6,
                            color='#DC143C',
                            flatshading=True,
                            name=f'obs_{i}'
                        ))
    return traces