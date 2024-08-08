import autograd.numpy as gnp
from robot_configs.robot_config import RobotConfig

class Uvms(RobotConfig):
    def __init__(self):
        pass
    
    @staticmethod
    def forward_kinematics(dict_var, rob_specs):
        position = gnp.array([[0.5*rob_specs['b_x']*gnp.cos(dict_var['yaw']) + 1.0*rob_specs['l_1']*gnp.cos(dict_var['q1'])*gnp.cos(dict_var['yaw']) + rob_specs['l_2']*gnp.cos(dict_var['yaw'])*gnp.cos(dict_var['q1'] + dict_var['q2']) + 1.0*dict_var['x']], 
                    [0.5*rob_specs['b_x']*gnp.sin(dict_var['yaw']) + 1.0*rob_specs['l_1']*gnp.sin(dict_var['yaw'])*gnp.cos(dict_var['q1']) + rob_specs['l_2']*gnp.sin(dict_var['yaw'])*gnp.cos(dict_var['q1'] + dict_var['q2']) + 1.0*dict_var['y']], 
                    [0.5*rob_specs['b_z'] - 1.0*rob_specs['l_1']*gnp.sin(dict_var['q1']) - rob_specs['l_2']*gnp.sin(dict_var['q1'] + dict_var['q2']) + 1.0*dict_var['z']], 
                    [1.]])
        return position
    
    @staticmethod
    def ee_orient(dict_var, rob_specs):
        orient = gnp.array([[gnp.cos(dict_var['yaw'])*gnp.cos(dict_var['q1'] + dict_var['q2'])],
                            [gnp.sin(dict_var['yaw'])*gnp.cos(dict_var['q1'] + dict_var['q2'])],
                            [-gnp.sin(dict_var['q1'] + dict_var['q2'])],
                            [1.0]])
        return orient