import logging
from time import time
import numpy as np
import autograd.numpy as gnp
from autograd import jacobian

from lib import cnkz, nkz

#TODO: Remove getStartState and getGoalState functions

logging.basicConfig(level=logging.DEBUG)
try:
    from lib.robot import Robot
    from lib.constraint_manifolds import ConstraintManifolds
except ImportError as e:
    logging.error('import failed in configuration space module')
    logging.error(e)
except:
    logging.error("Unknown error in Configuration space module.")

class Qspace:
    def __init__(self, rob_specs, robot_limits, robot_params, constraint_params, env_params):
        #### Set number of robots (It does not incorporate variable number of robots yet.)
        self.num_rob = robot_params['num_bots']
        self.z = robot_params['base_z'] #### Seems not used anywhere
        self.robots = []
        self.min_len = robot_params['min_len']
        self.struct_type = robot_params['struct_type']
        #### Create Robot objects and assign their limits 

        for  _ in range (0,self.num_rob):
            self.robots.append(Robot(robot_limits))
            

        #### Configuration for each robot
        self.conf_rob = robot_params['dof']
        self.rob_specs = rob_specs

        #### generate the complete list of configuration in the form [r1_conf_dict,r2_conf_dict,r2_conf_dict]
        # self.robs_vars = list(util.rob_vars(self.conf_rob, self.num_rob).values())

        #### generate start and goal state [x,y,z,yaw,q1,q2]*3
        self.start_state = self.get_state(env_params['start'])
        self.goal_state = self.get_state(env_params['goal'])

        #TODO
        #### This needs to be checked and moved while generating dictionary
        self.z_plane = util.ee_position(self.conf_rob, self.goal_state, rob_specs=rob_specs)[2]
        
        #### Adding constraint param for m4
        constraint_params['m4']['z'] = self.z_plane

        #### Initialize all the manifolds
        self.mfolds = ConstraintManifolds(params=constraint_params, rob_specs=rob_specs)

    def get_rand_config(self):
        samp = []
        sort_element = []
        
        for i in range (0, self.num_rob):
            tmp = list(self.robots[i].sample_rand_config())
            samp.append(tmp)
            sort_element.append(tmp[1]) ### y axis value

        samp = np.array(samp)
        sort_element = np.array(sort_element)
        samp_arr = np.array([(samp[i]) for i in np.argsort(sort_element)])
        samp_arr = samp_arr.flatten()

        return samp_arr

    def get_biased_rand_config(self):
        samp = []
        sort_element = []
        
        for i in range (0, self.num_rob):
            tmp = list(self.robots[i].sample_rand_config())
            tmp_base = np.random.randint([self.goal_state[i*len(self.conf_rob)], self.goal_state[i*len(self.conf_rob)+1], self.goal_state[i*len(self.conf_rob)+2]],
                                    [self.goal_state[i*len(self.conf_rob)]+ 3.0, self.goal_state[i*len(self.conf_rob)+1]+ 3.0, self.goal_state[i*len(self.conf_rob)+2]+ 2.0]).astype(np.float16) 
            
            tmp[0:3] = tmp_base.copy()
            samp.append(tmp)
            sort_element.append(tmp[1]) ### y axis value

        samp = np.array(samp)
        sort_element = np.array(sort_element)
        samp_arr = np.array([(samp[i]) for i in np.argsort(sort_element)])
        samp_arr = samp_arr.flatten()

        return samp_arr

    def get_state(self, cell_pose): ###### Change this to be robust
        """to generate the start and target states

        Args:
            cell_pose (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.rob_specs['type']=='UVMS':
            if self.struct_type =='S':
                tmp = np.array([cell_pose[0], cell_pose[1], cell_pose[2], np.float(self.robots[0].limits['angles']['max'][0])*np.pi/180., 
                        np.float(self.robots[0].limits['angles']['max'][1])*np.pi/180., 
                        np.float(self.robots[0].limits['angles']['max'][2])*np.pi/180.])                         
                r1 = np.around(tmp, decimals=2)
                arr = r1.copy()
                for i in range(1, self.num_rob):
                    tmp = np.array([cell_pose[0], cell_pose[1] + i * self.min_len, cell_pose[2], np.float(self.robots[i].limits['angles']['max'][0])*np.pi/180., 
                        np.float(self.robots[i].limits['angles']['max'][1])*np.pi/180., 
                        np.float(self.robots[i].limits['angles']['max'][2])*np.pi/180.])                         
                    r1 = np.around(tmp, decimals=2)
                    arr = np.append(arr,r1,0)

                return arr
            elif self.struct_type =='T':
                tmp = np.array([cell_pose[0], cell_pose[1], cell_pose[2], np.float(self.robots[0].limits['angles']['max'][0])*np.pi/180., 
                        np.float(self.robots[0].limits['angles']['max'][1])*np.pi/180., np.float(self.robots[0].limits['angles']['max'][2])*np.pi/180.])                         
                r1 = np.around(tmp, decimals=2)
                arr = r1.copy()
                
                tmp = np.array([cell_pose[0], cell_pose[1] + 2.0 * self.min_len, cell_pose[2], np.float(self.robots[1].limits['angles']['max'][0])*np.pi/180., 
                    np.float(self.robots[1].limits['angles']['max'][1])*np.pi/180., np.float(self.robots[1].limits['angles']['max'][2])*np.pi/180.])                         
                r1 = np.around(tmp, decimals=2)
                arr = np.append(arr,r1,0)

                tmp = np.array([cell_pose[0] + self.min_len, cell_pose[1] + self.min_len, cell_pose[2], np.float(self.robots[1].limits['angles']['max'][0])*np.pi/180., 
                    np.float(self.robots[1].limits['angles']['max'][1])*np.pi/180., np.float(self.robots[1].limits['angles']['max'][2])*np.pi/180.])                         
                r1 = np.around(tmp, decimals=2)
                arr = np.append(arr,r1,0)
                return arr
            
            elif self.struct_type =='I':
                tmp = np.array([cell_pose[0], cell_pose[1], cell_pose[2], np.float(self.robots[0].limits['angles']['max'][0])*np.pi/180., 
                        np.float(self.robots[0].limits['angles']['max'][1])*np.pi/180., np.float(self.robots[0].limits['angles']['max'][2])*np.pi/180.])                         
                r1 = np.around(tmp, decimals=2)
                arr = r1.copy()
                
                tmp = np.array([cell_pose[0], cell_pose[1] + 2.0 * self.min_len, cell_pose[2], np.float(self.robots[1].limits['angles']['max'][0])*np.pi/180., 
                    np.float(self.robots[1].limits['angles']['max'][1])*np.pi/180., np.float(self.robots[1].limits['angles']['max'][2])*np.pi/180.])                         
                r1 = np.around(tmp, decimals=2)
                arr = np.append(arr,r1,0)

                tmp = np.array([cell_pose[0] + self.min_len, cell_pose[1] + self.min_len, cell_pose[2], np.float(self.robots[1].limits['angles']['max'][0])*np.pi/180., 
                    np.float(self.robots[1].limits['angles']['max'][1])*np.pi/180., np.float(self.robots[1].limits['angles']['max'][2])*np.pi/180.])                         
                r1 = np.around(tmp, decimals=2)
                arr = np.append(arr,r1,0)

                tmp = np.array([cell_pose[0] + 2.0 * self.min_len, cell_pose[1], cell_pose[2], np.float(self.robots[0].limits['angles']['max'][0])*np.pi/180., 
                        np.float(self.robots[0].limits['angles']['max'][1])*np.pi/180., np.float(self.robots[0].limits['angles']['max'][2])*np.pi/180.])                         
                r1 = np.around(tmp, decimals=2)
                arr = np.append(arr,r1,0)
                
                tmp = np.array([cell_pose[0] + 2.0 * self.min_len, cell_pose[1] + 2.0 * self.min_len, cell_pose[2], np.float(self.robots[1].limits['angles']['max'][0])*np.pi/180., 
                    np.float(self.robots[1].limits['angles']['max'][1])*np.pi/180., np.float(self.robots[1].limits['angles']['max'][2])*np.pi/180.])                         
                r1 = np.around(tmp, decimals=2)
                arr = np.append(arr,r1,0)
                return arr
        elif self.rob_specs['type']=='TB3OM':
            if self.struct_type =='S':
                tmp = np.array([cell_pose[0], cell_pose[1], np.float(self.robots[0].limits['angles']['max'][0])*np.pi/180., 
                        np.float(self.robots[0].limits['angles']['max'][1])*np.pi/180., np.float(self.robots[0].limits['angles']['max'][2])*np.pi/180., 
                        np.float(self.robots[0].limits['angles']['max'][3])*np.pi/180.,np.float(self.robots[0].limits['angles']['max'][3])*np.pi/180.])                         
                r1 = np.around(tmp, decimals=2)
                arr = r1.copy()
                for i in range(1, self.num_rob):
                    tmp = np.array([cell_pose[0], cell_pose[1] + i * self.min_len, np.float(self.robots[i].limits['angles']['max'][0])*np.pi/180., 
                        np.float(self.robots[i].limits['angles']['max'][1])*np.pi/180., np.float(self.robots[i].limits['angles']['max'][2])*np.pi/180., 
                        np.float(self.robots[0].limits['angles']['max'][3])*np.pi/180.,np.float(self.robots[0].limits['angles']['max'][3])*np.pi/180.])                         
                    r1 = np.around(tmp, decimals=2)
                    arr = np.append(arr,r1,0)

                return arr
        

    def get_kacz(self, samp_state, proj_eps, proj_max_iter):
        _,proj_state,residuals,_,status,_= (nkz.kaczmarz(samp_state, self.mfolds, proj_max_iter, eps=proj_eps ))
        return status,residuals, proj_state
    
    def get_projection_cnkz(self, nearest_state, samp_state, proj_eps, proj_max_iter, factor = 1.0):
        d = np.subtract(samp_state, nearest_state)
        unit_state = np.around(np.asarray(nearest_state) + d * (factor * np.float(self.num_rob) / np.linalg.norm(d)), decimals=2)
        #### add projection
        _,proj_state,residual,_,status,time_taken= (cnkz.kaczmarz(unit_state, self.mfolds, proj_max_iter, eps=proj_eps ))
        return status, proj_state, residual, time_taken
    
    # TODO: get_projection single method. Pass Kaczmarz, cimminos. Send dictionary for more than 2 parameters
    
    def get_projection_nkz(self, nearest_state, samp_state, proj_eps, proj_max_iter, factor = 1.0):
        d = np.subtract(samp_state, nearest_state)
        unit_state = np.around(np.asarray(nearest_state) + d * (factor * np.float(self.num_rob) / np.linalg.norm(d)), decimals=2)
        #### add projection
        _,proj_state,residual,_,status,time_taken= (nkz.kaczmarz(unit_state, self.mfolds, proj_max_iter, eps=proj_eps ))
        return status, proj_state, residual, time_taken