import autograd.numpy as gnp
from autograd import jacobian
import logging

try:
    from lib import manifolds
except ImportError as e:
    logging.error('import failed in configuration space module')
    logging.error(e)
except:
    logging.error("Unknown error in Configuration space module.")

class ConstraintManifolds:
    def __init__(self, params, rob_specs):
        #### Create object for individual manifolds
        self.m1 = manifolds.UVMSonRodManifold(params=params['m1'], rob_specs=rob_specs)
        self.m2 = manifolds.UVMSEquidistanctOnRodManifold(params=params['m2'], rob_specs=rob_specs)
        self.m3 = manifolds.UVMStoRodOrientManifold(params=params['m3'], rob_specs=rob_specs)
        
        self.m4 = manifolds.UvmsEeOnSameZPlanedManifold(params=params['m4'], rob_specs=rob_specs)
        # self.m4 = manifolds.UvmsEeOnFixedZPlanedManifold(params=params['m4'], rob_specs=rob_specs)
        self.counter = 0
        
    def y(self, q):
        ### Append all the eqs into 1 array.
        # logging.debug(f"In ConstMani y:")
        # logging.debug(f"In ConstMani y: {gnp.concatenate((self.m1.y(q), self.m2.y(q), self.m3.y(q), self.m4.y(q)),axis=0)}")
        return gnp.concatenate((self.m1.y(q), self.m2.y(q), self.m3.y(q), self.m4.y(q)),axis=0)
    
    def get_sub_y(self, q):
        ### Append all the eqs into 1 array.
        # logging.debug(f"In ConstMani y:")
        # logging.debug(f"In ConstMani y: {gnp.concatenate((self.m1.y(q), self.m2.y(q), self.m3.y(q), self.m4.y(q)),axis=0)}")
        return gnp.concatenate((self.m1.y(q), self.m2.y(q), self.m3.y(q), self.m4.y(q)),axis=0)[self.counter]

    def J(self, q):
        jacobian_cost = jacobian(self.y)
        return jacobian_cost(q)
    
    def get_sub_j (self, q):
        jacobian_cost = jacobian(self.get_sub_y)
        return jacobian_cost(q)