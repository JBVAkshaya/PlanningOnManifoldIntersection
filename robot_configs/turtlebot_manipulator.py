import autograd.numpy as gnp
from robot_configs.robot_config import RobotConfig


class TurtlebotManipulator(RobotConfig):

    @staticmethod
    def forward_kinematics(dict_var, rob_specs):

        # Tranfer to base frame
        T_base = gnp.array([[gnp.cos(dict_var['yaw']), gnp.sin(dict_var['yaw']), 0., dict_var['x']],
                                [gnp.sin(dict_var['yaw']), gnp.cos(dict_var['yaw']), 0., dict_var['y']],
                                [0.,              0.,        1, rob_specs['base_z']],
                                [0.,              0.,        0., 1   ]])
        
        base_to_link5 = gnp.array([[gnp.cos(dict_var['q0'])*gnp.cos(dict_var['q1'] + dict_var['q2'] + dict_var['q3']), -gnp.sin(dict_var['q0']), gnp.sin(dict_var['q1'] + dict_var['q2'] + dict_var['q3'])*gnp.cos(dict_var['q0']), 0.128*gnp.sin(dict_var['q1'])*gnp.cos(dict_var['q0']) + 0.024*gnp.cos(dict_var['q0'])*gnp.cos(dict_var['q1']) + 0.124*gnp.cos(dict_var['q0'])*gnp.cos(dict_var['q1'] + dict_var['q2']) - 0.08], 
                         [gnp.sin(dict_var['q0'])*gnp.cos(dict_var['q1'] + dict_var['q2'] + dict_var['q3']), gnp.cos(dict_var['q0']), gnp.sin(dict_var['q0'])*gnp.sin(dict_var['q1'] + dict_var['q2'] + dict_var['q3']), (0.128*gnp.sin(dict_var['q1']) + 0.024*gnp.cos(dict_var['q1']) + 0.124*gnp.cos(dict_var['q1'] + dict_var['q2']))*gnp.sin(dict_var['q0'])], 
                         [-gnp.sin(dict_var['q1'] + dict_var['q2'] + dict_var['q3']), 0., gnp.cos(dict_var['q1'] + dict_var['q2'] + dict_var['q3']), -0.024*gnp.sin(dict_var['q1']) - 0.124*gnp.sin(dict_var['q1'] + dict_var['q2']) + 0.128*gnp.cos(dict_var['q1']) + 0.176], 
                         [0., 0., 0., 1.0]])
        ee = gnp.array([
                        [rob_specs['ee']],
                        [0.0],
                        [0.0],
                        [1.0]])# edit this according to ee position vector in link5 frame
        return gnp.matmul(gnp.matmul(T_base, base_to_link5), ee)
    
    @staticmethod
    def ee_orient(dict_var, rob_specs):
        return gnp.array([[gnp.cos(dict_var['q0'] - dict_var['yaw'])*gnp.cos(dict_var['q1'] + dict_var['q2'] + dict_var['q3'])], 
                [gnp.sin(dict_var['q0'] + dict_var['yaw'])*gnp.cos(dict_var['q1'] + dict_var['q2'] + dict_var['q3'])], 
                [-gnp.sin(dict_var['q1'] + dict_var['q2'] + dict_var['q3'])],
                [1.0]])