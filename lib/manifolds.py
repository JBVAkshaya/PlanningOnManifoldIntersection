from lib import util
import autograd.numpy as gnp
from abc import ABC, abstractmethod
import time

import logging
'''
y takes a single numpy array of robots configuration. 
'''
class Manifold(ABC):
    """Abstract class to represent a manifold."""
    @abstractmethod
    def __init__(self,
                 params, rob_specs):
        '''
        Parameters
        ----------
        params = {
            'min_len': length between two consecutive robots
            'struct_type': Type of structure: S-> Straight, T-> T-shaped, I-> I-shaped
            'num_robots': number of robots carrying the structure
            'penalty_factor': linear Penalty factor for constraint
        }
        '''
        self.min_len = params['min_len']
        self.struct_type = params['struct_type']
        self.num_robots = params['num_robots']
        self.dofs = rob_specs['dof']
        self.num_dofs = len(self.dofs)
        self.name = params['name']
        self.rob_specs = rob_specs

    """This function describes the level set of the manifold y(x)=0."""
    @abstractmethod
    def y(self, x: gnp.ndarray) -> gnp.ndarray:
        pass


class UvmsEeOnSameZPlanedManifold(Manifold):
    def __init__(self, params, rob_specs):
        '''
        Generates manifold for a fixed plane. 

        Parameters
        ----------
        params = {
            'z': value corresponding to val in z = val plane,
            'min_len': length between two consecutive robots
            'struct_type': Type of structure: S-> Straight, T-> T-shaped, I-> I-shaped
            'num_robots': number of robots carrying the structure
            'penalty_factor': linear Penalty factor for constraint
        }
        '''

        self.z_coord = params['z']
        self.penalty = params['penalty_factor']
        params['name'] = "UvmsEeOnSameZPlanedManifold"
        Manifold.__init__(self, params = params, rob_specs=rob_specs)
        

    def y(self, q):
        '''
        Returns the manifold for a configuration q

        Parameters
        ----------
        q = concatenated configuration of n robots holding the structure

        Returns
        -------
        Array -> array of constraints forming fixed plane manifold
        '''
        
        p = {}
        for i in range (0,self.num_robots):
            dict_Q = dict(zip(self.dofs,q[i*self.num_dofs:(i+1)*self.num_dofs]))
            p[str(i)] = util.trans_mat_single_bot_ee_position(dict_Q, rob_specs=self.rob_specs)
            p[str(i)] = p[str(i)][0:3]
        
        m = {}
        for i in range (0,self.num_robots -1):
            m[f'{i}'] = gnp.array([self.penalty*(p[str(i)][2,0] - p[str(i+1)][2,0])])
        
        mfolds = gnp.array(list(m.values()))
        return mfolds


class UvmsEeOnFixedZPlanedManifold(Manifold):
    def __init__(self, params, rob_specs):
        '''
        Generates manifold for a fixed plane. 

        Parameters
        ----------
        params = {
            'z': value corresponding to val in z = val plane,
            'min_len': length between two consecutive robots
            'struct_type': Type of structure: S-> Straight, T-> T-shaped, I-> I-shaped
            'num_robots': number of robots carrying the structure
            'penalty_factor': linear Penalty factor for constraint
        }
        '''

        self.z_coord = params['z']
        self.penalty = params['penalty_factor']
        params['name'] = "UvmsEeOnFixedZPlanedManifold"
        Manifold.__init__(self, params = params, rob_specs=rob_specs)
        

    def y(self, q):
        '''
        Returns the manifold for a configuration q

        Parameters
        ----------
        q = concatenated configuration of n robots holding the structure

        Returns
        -------
        Array -> array of constraints forming fixed plane manifold
        '''
        
        p = {}
        for i in range (0,self.num_robots):
            dict_Q = dict(zip(self.dofs,q[i*self.num_dofs:(i+1)*self.num_dofs]))
            p[str(i)] = util.trans_mat_single_bot_ee_position(dict_Q, rob_specs=self.rob_specs)
            p[str(i)] = p[str(i)][0:3]
        
        m = {}
        for i in range (0,self.num_robots):
            m[f'{i}'] = gnp.array([self.penalty*(p[str(i)][2,0] - self.z_coord)])
        
        mfolds = gnp.array(list(m.values()))
        return mfolds

class UVMSonRodManifold(Manifold):
    def __init__(self, params, rob_specs):
        self.penalty = params['penalty_factor']
        params['name'] = "UVMSonRod"
        Manifold.__init__(self, params=params, rob_specs=rob_specs)
        

    def y(self, q):
        '''
        Returns the manifold for a configuration q

        Parameters
        ----------
        q = concatenated configuration of n robots holding the structure

        Returns
        -------
        Array -> array of constraints for UVMS end effectors gripping rod manifold
        '''
        
        if self.struct_type=='S': 
            
            dict_Q = {}
            for i in range (0,self.num_robots):
                dict_Q[str(i)] = dict(zip(self.dofs,q[i*self.num_dofs:(i+1)*self.num_dofs]))
   
            v = {}
            for i in range(0,self.num_robots-1):
                v[str(i)] = util.trans_mat_single_bot_ee_position(dict_Q[str(i)],rob_specs=self.rob_specs) - util.trans_mat_single_bot_ee_position(dict_Q[str(i+1)],rob_specs=self.rob_specs)
                v[str(i)] = gnp.transpose(v[str(i)][0:3])

            count = 0
            c_dict = {}
            for i in range (0,len(v)):
                for j in range(i+1,len(v)):
                    c_dict[str(count)] = gnp.transpose(gnp.cross(v[str(i)],v[str(j)])*self.penalty)
                    count =  count + 1
            
            mfolds = gnp.array(c_dict['0'])
            for i in range (1,len(c_dict)):
                mfolds = gnp.concatenate((mfolds,c_dict[f'{i}']),axis=0)
            return mfolds
        elif self.struct_type=='T':
            dict_Q1 = dict(zip(self.dofs,q[0:6]))
            dict_Q2 = dict(zip(self.dofs,q[6:12]))
            dict_Q3 = dict(zip(self.dofs,q[12:18]))
            center = (util.trans_mat_single_bot_ee_position(dict_Q1, rob_specs=self.rob_specs)/2.0) + (util.trans_mat_single_bot_ee_position(dict_Q2, rob_specs=self.rob_specs)/2.0)
            v1 = util.trans_mat_single_bot_ee_position(dict_Q1,rob_specs=self.rob_specs) - util.trans_mat_single_bot_ee_position(dict_Q2,rob_specs=self.rob_specs)
            v2 = center - util.trans_mat_single_bot_ee_position(dict_Q3,rob_specs=self.rob_specs)
            # v1 = gnp.transpose(v1[0:3])
            # v2 = gnp.transpose(v2[0:3])
            # return gnp.transpose(gnp.cross(v1,v2)*1.4)
            return gnp.array([[gnp.dot(gnp.transpose(v1[0:3])[0], gnp.transpose(v2[0:3])[0])]])*self.penalty

        elif self.struct_type=='I':
            dict_Q1 = dict(zip(self.dofs,q[0:6]))
            dict_Q2 = dict(zip(self.dofs,q[6:12]))
            dict_Q3 = dict(zip(self.dofs,q[12:18]))
            dict_Q4 = dict(zip(self.dofs,q[18:24]))
            dict_Q5 = dict(zip(self.dofs,q[24:30]))

            center = (util.trans_mat_single_bot_ee_position(dict_Q1, rob_specs=self.rob_specs)/2.0) + (util.trans_mat_single_bot_ee_position(dict_Q2, rob_specs=self.rob_specs)/2.0)
            v1 = util.trans_mat_single_bot_ee_position(dict_Q1,rob_specs=self.rob_specs) - util.trans_mat_single_bot_ee_position(dict_Q2,rob_specs=self.rob_specs)
            v2 = center - util.trans_mat_single_bot_ee_position(dict_Q3,rob_specs=self.rob_specs)
            v1 = gnp.transpose(v1[0:3])
            v2 = gnp.transpose(v2[0:3])

            center_2 = (util.trans_mat_single_bot_ee_position(dict_Q4, rob_specs=self.rob_specs)/2.0) + (util.trans_mat_single_bot_ee_position(dict_Q5, rob_specs=self.rob_specs)/2.0)
            v3 = util.trans_mat_single_bot_ee_position(dict_Q4,rob_specs=self.rob_specs) - util.trans_mat_single_bot_ee_position(dict_Q5,rob_specs=self.rob_specs)
            v4 = center_2 - util.trans_mat_single_bot_ee_position(dict_Q3,rob_specs=self.rob_specs)
            v3 = gnp.transpose(v3[0:3])
            v4 = gnp.transpose(v4[0:3])
            # return gnp.transpose(gnp.cross(v1,v2)*1.4)

            mfolds = gnp.reshape(self.penalty*gnp.cross(v2[0], v4[0]),(3,1))
            mfolds = gnp.concatenate((mfolds,self.penalty*gnp.array([[gnp.dot(v4[0], v3[0])]])),axis=0)
            mfolds = gnp.concatenate((mfolds,gnp.array([[gnp.dot(v1[0], v2[0])]])*self.penalty),axis=0)
            
            return mfolds



class UVMSEquidistanctOnRodManifold(Manifold):
    def __init__(self, params, rob_specs):
        params['name'] = "UVMSEquidistanctOnRod"
        Manifold.__init__(self, params=params, rob_specs=rob_specs) ### Not sure about dimensions
        

    def y(self, q):
        if self.struct_type=='S':
            ### L = 0.5
            
            dict_Q = {}
            for i in range (0,self.num_robots):
                dict_Q[str(i)] = dict(zip(self.dofs,q[i*self.num_dofs:(i+1)*self.num_dofs]))

            v = {}
            v_transpose = {}
            m = {}
            for i in range (0,self.num_robots):   
                for j in range(i+1, self.num_robots):
                    v[f'{i}_{j}'] = util.trans_mat_single_bot_ee_position(dict_Q[str(i)], rob_specs=self.rob_specs) - util.trans_mat_single_bot_ee_position(dict_Q[str(j)], rob_specs=self.rob_specs)
                    v_transpose[str(f'{i}_{j}')] = gnp.transpose(v[str(f'{i}_{j}')][0:3])
                    m_temp = gnp.matmul(v_transpose[str(f'{i}_{j}')], v[str(f'{i}_{j}')][0:3])
                    m[f'{i}_{j}'] = gnp.array([m_temp[0,0] - ((j-i)*self.min_len)**2])
            
            mfolds = gnp.array(list(m.values()))
            return mfolds
        elif self.struct_type=='T':
            ### L = 0.5
            dict_Q1 = dict(zip(self.dofs,q[0:6]))
            dict_Q2 = dict(zip(self.dofs,q[6:12]))
            dict_Q3 = dict(zip(self.dofs,q[12:18]))
            center = (util.trans_mat_single_bot_ee_position(dict_Q1, rob_specs=self.rob_specs))/2.0 + (util.trans_mat_single_bot_ee_position(dict_Q2, rob_specs=self.rob_specs)/2.0)
            v1 = util.trans_mat_single_bot_ee_position(dict_Q1, rob_specs=self.rob_specs) - util.trans_mat_single_bot_ee_position(dict_Q2, rob_specs=self.rob_specs)
            v2 = util.trans_mat_single_bot_ee_position(dict_Q2, rob_specs=self.rob_specs)- util.trans_mat_single_bot_ee_position(dict_Q3, rob_specs=self.rob_specs)
            v3 = util.trans_mat_single_bot_ee_position(dict_Q1, rob_specs=self.rob_specs)- util.trans_mat_single_bot_ee_position(dict_Q3, rob_specs=self.rob_specs)
            v4 = util.trans_mat_single_bot_ee_position(dict_Q3, rob_specs=self.rob_specs) - center
            
            v1_t = gnp.transpose(v1[0:3])
            v2_t = gnp.transpose(v2[0:3])
            v3_t = gnp.transpose(v3[0:3])
            v4_t = gnp.transpose(v4[0:3])

            m1 = gnp.matmul(v1_t, v1[0:3])
            m1 = m1[0,0] - 1.0**2
            m2_1 = gnp.matmul(v2_t, v2[0:3])
            m2_2 = gnp.matmul(v3_t, v3[0:3])
            m2 = m2_1[0,0] - m2_2[0,0]
            m3 = gnp.matmul(v4_t, v4[0:3])
            m3 = m3[0,0] - 0.5**2

            return gnp.array([[m1],[m2],[m3]])
        
        elif self.struct_type=='I':
            ### L = 0.5
            dict_Q1 = dict(zip(self.dofs,q[0:6]))
            dict_Q2 = dict(zip(self.dofs,q[6:12]))
            dict_Q3 = dict(zip(self.dofs,q[12:18]))
            dict_Q4 = dict(zip(self.dofs,q[18:24]))
            dict_Q5 = dict(zip(self.dofs,q[24:30]))

            center = (util.trans_mat_single_bot_ee_position(dict_Q1, rob_specs=self.rob_specs))/2.0 + (util.trans_mat_single_bot_ee_position(dict_Q2, rob_specs=self.rob_specs)/2.0)
            v1 = util.trans_mat_single_bot_ee_position(dict_Q1, rob_specs=self.rob_specs) - util.trans_mat_single_bot_ee_position(dict_Q2, rob_specs=self.rob_specs)
            v2 = util.trans_mat_single_bot_ee_position(dict_Q2, rob_specs=self.rob_specs)- util.trans_mat_single_bot_ee_position(dict_Q3, rob_specs=self.rob_specs)
            v3 = util.trans_mat_single_bot_ee_position(dict_Q1, rob_specs=self.rob_specs)- util.trans_mat_single_bot_ee_position(dict_Q3, rob_specs=self.rob_specs)
            v4 = util.trans_mat_single_bot_ee_position(dict_Q3, rob_specs=self.rob_specs) - center
            
            v1_t = gnp.transpose(v1[0:3])
            v2_t = gnp.transpose(v2[0:3])
            v3_t = gnp.transpose(v3[0:3])
            v4_t = gnp.transpose(v4[0:3])

            m1 = gnp.matmul(v1_t, v1[0:3])
            m1 = m1[0,0] - 1.0**2
            m2_1 = gnp.matmul(v2_t, v2[0:3])
            m2_2 = gnp.matmul(v3_t, v3[0:3])
            m2 = m2_1[0,0] - m2_2[0,0]
            m3 = gnp.matmul(v4_t, v4[0:3])
            m3 = m3[0,0] - 0.5**2


            center_2 = (util.trans_mat_single_bot_ee_position(dict_Q4, rob_specs=self.rob_specs))/2.0 + (util.trans_mat_single_bot_ee_position(dict_Q5, rob_specs=self.rob_specs)/2.0)
            v5 = util.trans_mat_single_bot_ee_position(dict_Q4, rob_specs=self.rob_specs) - util.trans_mat_single_bot_ee_position(dict_Q5, rob_specs=self.rob_specs)
            v6 = util.trans_mat_single_bot_ee_position(dict_Q5, rob_specs=self.rob_specs)- util.trans_mat_single_bot_ee_position(dict_Q3, rob_specs=self.rob_specs)
            v7 = util.trans_mat_single_bot_ee_position(dict_Q4, rob_specs=self.rob_specs)- util.trans_mat_single_bot_ee_position(dict_Q3, rob_specs=self.rob_specs)
            v8 = util.trans_mat_single_bot_ee_position(dict_Q3, rob_specs=self.rob_specs) - center_2
            
            v5_t = gnp.transpose(v5[0:3])
            v6_t = gnp.transpose(v6[0:3])
            v7_t = gnp.transpose(v7[0:3])
            v8_t = gnp.transpose(v8[0:3])

            m4 = gnp.matmul(v5_t, v5[0:3])
            m4 = m4[0,0] - 1.0**2
            m5_1 = gnp.matmul(v6_t, v6[0:3])
            m5_2 = gnp.matmul(v7_t, v7[0:3])
            m5 = m5_1[0,0] - m5_2[0,0]
            m6 = gnp.matmul(v8_t, v8[0:3])
            m6 = m6[0,0] - 0.5**2

            v9 = util.trans_mat_single_bot_ee_position(dict_Q1, rob_specs=self.rob_specs) - util.trans_mat_single_bot_ee_position(dict_Q4, rob_specs=self.rob_specs)
            v10 = util.trans_mat_single_bot_ee_position(dict_Q2, rob_specs=self.rob_specs)- util.trans_mat_single_bot_ee_position(dict_Q5, rob_specs=self.rob_specs)
            
            v9_t = gnp.transpose(v9[0:3])
            v10_t = gnp.transpose(v10[0:3])

            m7 = gnp.matmul(v9_t, v9[0:3])
            m7 = m7[0,0] - 1.0**2
            m8 = gnp.matmul(v10_t, v10[0:3])
            m8 = m8[0,0] - 1.0**2

            mfolds = gnp.array([[m1],[m2],[m3],[m4],[m5],[m6],[1.5*m7],[1.5*m8]])
            return mfolds
    

class UVMStoRodOrientManifold(Manifold):
    def __init__(self, params, rob_specs):
        params['name'] = "UVMStoRodOrient"
        Manifold.__init__(self, params=params, rob_specs=rob_specs)
        

    def y(self, q):

        dict_Q = {}
        uvms = {}
        for i in range (0,self.num_robots):
            dict_Q[str(i)] = dict(zip(self.dofs,q[i*self.num_dofs:(i+1)*self.num_dofs]))
            uvms[str(i)] = util.trans_mat_single_bot_ee_orient(dict_Q[str(i)], rob_specs=self.rob_specs)

        r = {}
        for i in range(0,self.num_robots-1):
            r[f'{i}'] = util.trans_mat_single_bot_ee_position(dict_Q[f'{i}'],rob_specs=self.rob_specs) - util.trans_mat_single_bot_ee_position(dict_Q[f'{i+1}'],rob_specs=self.rob_specs)
        
        m = {}
        
        for i in range(0,len(r)):
            for j in range(0,len(uvms)):
                m[f'{i}_{j}'] = gnp.array([gnp.dot(gnp.transpose(r[f'{i}'][0:3])[0], gnp.transpose(uvms[f'{j}'][0:3])[0])])

        mfolds = gnp.array(list(m.values()))
        return mfolds
        
# class LinearEqs(Manifold):
#     def __init__(self):
#         Manifold.__init__(self, name="LinearEqs", dim_ambient=18, dim_manifold=2) ### Not sure about dimensions
        

#     def y_equations(self, dict_Q1):
#         ### L = 0.5
#         vars = var(" ".join(dict_Q1.values()))
#         print(vars)
#         # [3*dict_Q1['x'] + 2*dict_Q1['y'] - 5]
#         m = Matrix([[Matrix([dict_Q1['x'], dict_Q1['y'], 1]).dot(Matrix([2,3,1]))], [Matrix([dict_Q1['x'], dict_Q1['y'], 1]).dot(Matrix([3,2,-5]))]])
#         return m