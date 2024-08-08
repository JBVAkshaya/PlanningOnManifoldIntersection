from abc import ABC, abstractmethod

class RobotConfig(ABC):
    
    @staticmethod
    @abstractmethod
    def forward_kinematics(dict_var, rob_specs):
        pass

    @staticmethod
    @abstractmethod
    def transform_matrix_arm0(dict_var, rob_specs):
        pass

    @staticmethod
    @abstractmethod
    def ee_orient(dict_var, rob_specs):
        pass
