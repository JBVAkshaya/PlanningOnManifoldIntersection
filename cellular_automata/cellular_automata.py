"""
cellular_automata.py

Generates a random maze using the cellular automata method


Sources used in this implementation
---
https://www.raywenderlich.com/2425-procedural-level-generation-in-games-using-a-cellular-automaton-part-1#toc-anchor-010
https://gamedevelopment.tutsplus.com/tutorials/generate-random-cave-levels-using-cellular-automata--gamedev-9664
https://softologyblog.wordpress.com/2019/12/28/3d-cellular-automata-3/

"""
from typing import List

import numpy as np

from cellular_automata.random_env import RandomEnv
from scipy import ndimage
from scipy.spatial import distance_matrix
from tqdm import tqdm


class CellAutomata(RandomEnv):
    """
        Class for generating a random 3D maze using Cellular Automata.
    """
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
            'ca_prop': dict
                dictionary of parameters for defining a cellular automata
                'init_prob': float
                    probability of a cell being a wall initially
                'neighbor_type': str
                    type of neighboorhood used for CA rules
                    'moore' - use Moore Neighborhood of 26 surrounding tiles
                    'neumann' - use von Neumann neighborhood of adjacent 6 tiles
                'death': int or tuple
                    if there are less than this number of wall neighbors 
                    the wall tile will 'die' and become free space
                'birth': int or tuple
                    if there are at least this number of wall neighbors 
                    a wall tile will be 'born'
                'steps': int
                    number of steps to run the cellular automata
        show_tqdm: bool
            if True, shows intermediate steps of maze generation using tqdm
        tqdm_offset: int
            offset tqdm progress bar by this amount for nested progress bars
        debug: bool
            if True, print debug statements
        """
        super().__init__(gen_params, show_tqdm=show_tqdm, tqdm_offset=tqdm_offset, debug=debug)
        self.ca_prop = gen_params['ca_prop']
        self.create_kernels()

    def init_cells(self):
        """
        Initializes self.cells using parameters in self.gen_params

        Returns
        -------
        cells: np.ndarray
            initial cell states 0 = free, 1 = wall

        """
        return self.rng.random(self.cell_array_shape) < self.gen_params['ca_prop']['init_prob']

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
        # self.start_cell_idx = self.world_pos_to_cell_index(self.start[0,0:3])
        # self.goal_cell_idx = self.world_pos_to_cell_index(self.goal[0,0:3])
        if self.show_tqdm:
            pbar = tqdm(total=self.ca_prop['steps'], desc='Gen Map - Iter', position=self.tqdm_offset)
        for iter in range(self.ca_prop['steps']):
            self.run_iter()
            if self.debug:
                tqdm.write(f'Iter: {iter}, Wall Cells {np.sum(self.cells)}/{self.cells.size}')
            if self.show_tqdm:
                pbar.update(1)
        if self.show_tqdm:
            pbar.close()

        if self.debug:
            tqdm.write('Completed evolving cellular automata')
            tqdm.write('Identifying Caverns')
        flood_fill_map = self.identify_caverns()
        if self.debug:
            tqdm.write(f'Found {np.max(flood_fill_map)} caverns')
            tqdm.write('Removing caverns')
        cells_removed = self.remove_caverns(flood_fill_map)
        if self.debug:
            tqdm.write(f'Final array has {np.sum(np.invert(cells_removed))}/{cells_removed.size} free spaces')
            tqdm.write(f'Finding max dist start/end positions')
        start_cell, goal_cell = self.choose_start_and_goal(cells_removed)

        start_pos = np.zeros(12)
        goal_pos =  np.zeros(12)
        start_pos[0:3] = self.cell_index_to_world_pos(start_cell)
        start_pos[5] = self.rng.uniform(-np.pi, np.pi)
        goal_pos[0:3] = self.cell_index_to_world_pos(goal_cell)
        goal_pos[5] = self.rng.uniform(-np.pi, np.pi)
        if self.debug:
            tqdm.write(f'start: {start_pos}, goal: {goal_pos}')

        result = dict(final_cells=cells_removed, start_pos=start_pos, goal_pos=goal_pos)

        return result

    def run_iter(self):
        """
        Runs a single iteration of cell automata birth/death
        """

        # Perform convolution to count the number of wall cells around each cell
        neighbor_count = ndimage.convolve(self.cells.astype('int'), self.kernel, mode='constant', cval=0.0)

        # Use birth/death logic to update self.cells based on neighboring count
        # 1) if a wall cell is surrounded by < self.ca_prop['death'] neighbors, the wall is destroyed
        # 2) if a floor cell is surrounded by > self.ca_prop['birth] neighbors, the wall is formed
        # We also have support for providing a range of values for birth/death to occur
        for i in range(self.cells.shape[0]):
            for j in range(self.cells.shape[1]):
                for k in range(self.cells.shape[2]):
                    # if we are at a wall, check to see if the wall should 'die'
                    if self.cells[i,j,k]:
                        if isinstance(self.ca_prop['death'], tuple):
                            dead = self.ca_prop['death'][0] < neighbor_count[i,j,k] < self.ca_prop['death'][1]
                        else:
                            dead = neighbor_count[i,j,k] < self.ca_prop['death']
                        if dead:
                            self.cells[i,j,k] = False
                    # if we are at an empty tile, check to see if a wall should be born
                    else:
                        if isinstance(self.ca_prop['birth'], tuple):
                            born = self.ca_prop['birth'][0] < neighbor_count[i,j,k] < self.ca_prop['birth'][1]
                        else:
                            born = neighbor_count[i,j,k] > self.ca_prop['birth']
                        if born:
                            self.cells[i,j,k] = True

    def _flood_fill_cavern(self, x: int, y: int, z: int, fill_num: int):
        """
        A recursive function that marks a cell with fill_number in
        self.flood_fill and then fills adjacents cells

        Parameters
        ----------
        x, y, z: int
            (x,y,z) index of the cell we are examining
        fill_num: int
            the number to fill the cell with
        """

        # If the current cell is a wall,
        # or has been flood filled before
        if self.flood_fill[x, y, z] != -1:
            return
        
        self.flood_fill[x, y, z] = fill_num

        adjacent_indices = []
        if x > 0:
            adjacent_indices.append([x-1, y, z])
        if x < self.cells.shape[0] - 1:
            adjacent_indices.append([x+1, y, z])
        if y > 0:
            adjacent_indices.append([x, y-1, z])
        if y < self.cells.shape[1] - 1:
            adjacent_indices.append([x, y+1, z])
        if z > 0:
            adjacent_indices.append([x, y, z-1])
        if z < self.cells.shape[2] - 1:
            adjacent_indices.append([x, y, z+1])
        
        for x, y, z in adjacent_indices:
            self._flood_fill_cavern(x, y, z, fill_num)

    def identify_caverns(self):
        """
        Creates a flood_fill array where:
        cells indicated filled blocks are marked by 0
        free spaces are indexed from 1-n where a cell with index i
        belongs to the i-th cavern

        Returns
        -------
        flood_fill: np.ndarray (self.cells.shape)

        
        """
        # Flood fill array is 0 for walls, -1 for free space,
        # positive int represents index of flood fill region
        self.flood_fill = -1 *np.invert(np.copy(self.cells)).astype('int')

        fill_num = 1
        if self.show_tqdm:
            pbar = tqdm(total = self.cells.size, desc='Gen Map - Flood Fill', position=self.tqdm_offset+1)
        for i in range(self.flood_fill.shape[0]):
            for j in range(self.flood_fill.shape[1]):
                for k in range(self.flood_fill.shape[2]):
                    # If we have not flood filled this cell
                    # flood fill it
                    if self.flood_fill[i, j, k] == -1:
                        self._flood_fill_cavern(i, j, k, fill_num)
                        fill_num += 1
                    if self.show_tqdm:
                        pbar.update(1)
        if self.show_tqdm:
            pbar.close()
        return self.flood_fill    

    def remove_caverns(self, flood_fill: np.ndarray):
        """
        Destructively remove unconnected caverns from the main room
        """
        # find the largest cavern in the map
        cavern_ind, counts = np.unique(flood_fill, return_counts=True)

        # If there are 0 or 1 caverns, then just stop
        if len(cavern_ind) < 2:
            return np.copy(self.cells)
        
        counts_ind_sorted = np.argsort(counts)
        most_cells = counts[counts_ind_sorted[-1]]
        
        # Grab the flood_fill index with the highest occurence
        main_cavern_ind = cavern_ind[counts_ind_sorted[-1]]
        # If that index corresponds to the wall index
        # skip and grab the second highest, which should correspond to free space
        if main_cavern_ind == 0:
            main_cavern_ind = cavern_ind[counts_ind_sorted[-2]]
            most_cells = counts[counts_ind_sorted[-2]]

        if self.debug:
            tqdm.write(f'Largest Cavern Index {main_cavern_ind}, with {most_cells} cells')

        cells_filled = np.copy(self.cells)

        if self.show_tqdm:
            pbar = tqdm(total = self.cells.size, desc='Gen Map - Filling Cells', position=self.tqdm_offset+1)
        for i in range(self.cells.shape[0]):
            for j in range(self.cells.shape[1]):
                for k in range(self.cells.shape[2]):
                    # If the current cell is not part of the main cavern
                    # fill it in with a wall
                    flood_val = self.flood_fill[i,j,k]
                    if (flood_val > 0) and (flood_val != main_cavern_ind):
                        cells_filled[i,j,k] = True
                    if self.show_tqdm:
                        pbar.update(1)
        if self.show_tqdm:
            pbar.close()
        return cells_filled

    def connect_caverns(self):
        """
        Connect caverns together using 
        """
        pass

    def choose_start_and_goal(self, cells):
        """
        Parameters
        ----------
        cells: np.ndarray 
            cells array where 0 represents free space and 1 represents walls
        """
        free_cells = []
        free_spaces = []
        for i in range(cells.shape[0]):
            for j in range(cells.shape[1]):
                for k in range(cells.shape[2]):
                    if cells[i,j,k] == 0:
                        cell_index = np.array([i,j,k])
                        world_pos = self.cell_index_to_world_pos(cell_index)
                        free_cells.append(cell_index)
                        free_spaces.append(world_pos)

        # If there are no free cells, just return origin
        if len(free_cells) == 0:
            return np.zeros(3), np.zeros(3)
        
        free_cells = np.array(free_cells)
        free_spaces = np.array(free_spaces)
        dist_mat = distance_matrix(free_spaces, free_spaces, p=2)
        ind = np.unravel_index(np.argmax(dist_mat), dist_mat.shape)
        return free_cells[ind[0]], free_cells[ind[1]]

    def create_kernels(self):
        """
        Create kernels for computing convolutions for fast neighbor counts.
        """
        # kernel for counting number of 'alive' wall cells
        # By looking at all 26 neighbors of a cell,
        # including the diagonals
        self.moore_kernel = np.zeros((3,3,3))
        self.moore_kernel[:] = 1
        # Don't look at the current cell's value
        # to determine number of alive neighbors
        self.moore_kernel[1,1,1] = 0

        # Kernel for looking at only immediate 6 neighbors of a cell
        self.neumann_kernel = np.zeros((3,3,3))
        self.neumann_kernel[1,:,1] = 1
        self.neumann_kernel[:,1,1] = 1
        self.neumann_kernel[1,1,:] = 1
        self.neumann_kernel[1,1,1] = 0

        if self.ca_prop['neighbor_type'] == 'moore':
            self.kernel = self.moore_kernel
        elif self.ca_prop['neighbor_type'] == 'neumann':
            self.kernel = self.neumann_kernel
        else:
            raise ValueError(f'Unknown type {self.ca_prop["neighbor_type"]} for neighbor_type. Expected "moore" or "neumann".')

if __name__=="__main__":
    pass