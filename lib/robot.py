import numpy as np
import logging

# logging.basicConfig(level=logging.DEBUG, format=("%(levelname)s: %(asctime)s: %(message)s"))

### All angles in radians

class Robot:
    def __init__(self, limits): 
        self.limits = limits


# ToDo: change so that q1 and q2 are in terms of pi.
    def sample_rand_config(self):
        
        pos = np.random.randint(self.limits['pos']['min'], self.limits['pos']['max']).astype(np.float16) 
        angles = np.random.randint(self.limits['angles']['min'], self.limits['angles']['max']).astype(np.float16)                           
        for i in range (0,angles.shape[0]):
            angles[i] = np.float(angles[i])*np.pi/180.

        # tmp = np.insert(tmp,[2], self.rov_z)
        config = np.append(pos,angles,0)
        return np.around(config, decimals=2)
        # return dict(zip(self.dofs, np.around(tmp, decimals=2)))