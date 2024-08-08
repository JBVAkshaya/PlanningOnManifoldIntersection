import json
import logging
import glob
import numpy as np

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)
if __name__=="__main__":
    
    file_names = glob.glob('random_worlds/*/*.json')
    logging.debug(file_names[0])
    with open(file_names[0],'r') as f:
        data = json.load(f)
    world_size = int(data['bounds']['x_bounds'][1])
    obstacles = data['obstacles']
    cell_size = obstacles[0]['sides']

    arr = np.zeros((world_size, world_size, world_size))
    for i in range(0,len(obstacles)):
        logging.debug(len(obstacles))
