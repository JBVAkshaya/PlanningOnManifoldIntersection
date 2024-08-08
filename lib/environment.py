
import numpy as np
import plotly.graph_objects as go
import logging
import fcl
import sys
from glob import glob
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def R_mat(a, axis: str, inverse=False):
    """
    Returns the rotation matrix corresponding to 
    rotation about given axis by a radians

    Parameters
    ----------
    a: np.ndarray, torch.tensor (n, 1)
        a vector of n angles to rotate by
    
    axis: str
        rotate about "x", "y", "z" axes
    
    inverse: bool
        return the inverse rotation to the given parameters
        (in rotation matrices, this is equivalent to transpose)

    Returns
    -------
    R: np.ndarray or torch.tensor (n, 3, 3)
        The rotation matrix corresponding to the transformation
    """


    if isinstance(a, np.ndarray):
        c = np.cos(a)
        s = np.sin(a)
        ones = np.ones(a.shape, dtype=a.dtype)
        zeros = np.zeros(a.shape, dtype=a.dtype)
        if axis == "x":
            R = np.array([[ones, zeros, zeros],
                          [zeros, c, -s],
                          [zeros, s, c]])
        elif axis == "y":
            R = np.array([[c, zeros, s],
                          [zeros, ones, zeros],
                          [-s, zeros, c]])
        elif axis == "z":
            R = np.array([[c, -s, zeros],
                          [s, c, zeros],
                          [zeros, zeros, ones]])
        # Change to [Batch, 3, 3] format
        R = np.moveaxis(R, [2, 0, 1],[0, 1, 2])
        # Transpose array if inverse
        if inverse:
            R = np.swapaxes(R, 1, 2)
        return R
        
def rpy_to_mat_single(a):
    """
    Computes the rotation matrix (or matrices) 
    that is specified by the roll, pitch, yaw
    in a

    As opposed to rpy_to_mat(), this function accepts (3) arrays 
    of just the orientation.

    Parameters
    ----------
    a: np.ndarray or torch.tensor or List or Tuple (3,) 
        an iterable of 3 values consisting of roll, pitch, and yar

    Returns
    -------
    R:  np.ndarray or torch.tensor (3,3) 
        Rotation matrix representing the three values
    """
    assert len(a) == 3, f"The input to rpy_to_mat_single must be of length 3, got input of length {len(a)}"

    if isinstance(a, (list, tuple)):
        a = np.array(a)
        
    a= a.reshape(1, 3)

    Rx = R_mat(a[:,0], "x", inverse=True)[0]
    Ry = R_mat(a[:,1], "y", inverse=True)[0]
    Rz = R_mat(a[:,2], "z", inverse=True)[0]

    return Rx @ Ry @ Rz

class Environment:
    def __init__(self):
        self.request_continuous = fcl.ContinuousCollisionRequest(ccd_motion_type=fcl.CCDMotionType.CCDM_TRANS)
        self.result_continuous = fcl.ContinuousCollisionResult()
        self.request_collide = fcl.CollisionRequest()
        self.result_collide = fcl.CollisionResult()

    def cuboid_mesh(self, extents = [0.5, 1.0, 0.1], offset = [2.0, 0, -0.5], rotation = [0.0,0.0,0.0], colors = [0, 1., 0, 0.5]):
        return fcl.CollisionObject(fcl.Box(extents[0],extents[1],extents[2]), fcl.Transform(np.array(rpy_to_mat_single(rotation)), np.array(offset)))

    def cylinder_mesh(self, radius = 0.5, height = 1.0, offset = [2.0, 0, -0.5], rotation = [0.0,0.0,0.0], colors = [0, 1., 0, 0.5]):
        return fcl.CollisionObject(fcl.Cylinder(radius=0.5,height=0.5), fcl.Transform(np.array(rpy_to_mat_single(rotation)), np.array(offset)))

    def get_obstacle_objs(self,box_defs, obs_three_d, z_translation = 0.0):
        obstacle_objs = {}
        for i in range(0,len(box_defs)):
            size = ','.join(str(x) for x in box_defs[i]['sides'])

            ## Just for testing collision check
            if obs_three_d == False:
                box_defs[i]['position'][2] = 0.0
            else:
                box_defs[i]['position'][2] = box_defs[i]['position'][2] - z_translation
            obstacle_objs['obs_'+str(i)+'_'+str(size)] = self.cuboid_mesh(extents=box_defs[i]['sides'],
                                                        offset=box_defs[i]['position'])                                              
        return obstacle_objs

    def get_structure_objs(self, state, sides = [1.0,1.0,1.0]):
        
        meshes = {}
        for i in range(0,len(state)):
            meshes[str(i)] = self.cuboid_mesh(extents = sides, offset =state[i] ,rotation = [0,0,0])
        return meshes

    def no_continuous_collision(self, obs_list, struct_source, struct_target):
        request_continuous = fcl.ContinuousCollisionRequest(ccd_motion_type=fcl.CCDMotionType.CCDM_TRANS)
        result_continuous = fcl.ContinuousCollisionResult()
        request_collide = fcl.CollisionRequest()
        result_collide = fcl.CollisionResult()
        in_collision = False
        for i in range (0,len(struct_source)):
            if in_collision == False:
                for _, obs_mesh in obs_list.items():
                    ret = fcl.continuousCollide(obs_mesh, 
                                        fcl.Transform(obs_mesh.getRotation(),obs_mesh.getTranslation()),
                                        struct_source[str(i)],
                                        fcl.Transform(struct_target[str(i)].getRotation(),struct_target[str(i)].getTranslation()),
                                        request_continuous,
                                        result_continuous)
                    # print(f'ret: status: ', ret, result_continuous.is_collide)
                    if result_continuous.is_collide == True:
                        in_collision = True
                        break
                
                for _, target in struct_target.items():
                    ret_1 = fcl.collide(struct_source[str(i)], target, request_collide, result_collide)
                    # print(f'ret_1: status: ', ret_1, result_collide.is_collision)
                    if result_collide.is_collision == True:
                        in_collision = True
                        break
            else:
                break
        

        logger.debug(f'in collision: {in_collision}')
        return not in_collision

    def create_collision_manager(self, mesh_list):
        pass
    
    def is_scene_collision(self, collision_manager, mesh, return_names=True):
        pass
    
    def gen_bb_points(self, sides : np.ndarray = np.array([1.0, 1.0, 1.0]),
                  orient : np.ndarray = None,
                  pos : np.ndarray = None):
        # Create box centered at the origin with given side lengths
        points = 0.5*np.array([[-1, -1, -1],
                            [ -1, -1, 1],
                            [ -1,  1, -1],
                            [-1,  1,  1],
                            [1, -1,  -1],
                            [ 1, -1,  1],
                            [ 1,  1,  -1],
                            [1,  1,  1]])
        points *= sides
        if orient is not None:
            points = (orient @ points.T).T
        if pos is not None:
            points += pos
        return points

    def plot_in_plotly(self, mesh_list, sides = [5.0,5.0,5.0]):
        '''
        Parameters
        ----------
        mesh_list: array of CollsionObjects objects. Generally boxes created using cuboid_mesh() function
        '''
        fig = go.Figure()
        for _,mesh in mesh_list.items():

            #TODO: send sides as parameter

            points = self.gen_bb_points(sides=sides,orient=mesh.getRotation(),pos=mesh.getTranslation())
            x,y,z = points[:,0],points[:,1],points[:,2]
            fig.add_trace(go.Mesh3d(
                # 8 vertices of a cube
                x=x,y=y,z=z,
                # # i, j and k give the vertices of triangles
                i = [7, 1, 0, 1, 4, 5, 6, 6, 2, 4, 3, 1],
                j = [3, 5, 1, 2, 5, 6, 4, 2, 0, 1, 6, 0],
                k = [1, 7, 2, 3, 6, 7, 2, 3, 4, 5, 7, 4],
                opacity=1.0,
                color='#DC143C',
                flatshading = True))

        # fig.update_layout(
        #     scene = dict(
        #         xaxis = dict(nticks=4, range=[0,50],),
        #                     yaxis = dict(nticks=4, range=[-1,50],),
        #                     zaxis = dict(nticks=4, range=[-0.5, 50],),),
        #     width=700,
        #     margin=dict(r=10, l=10, b=10, t=10))
        fig.show()

if __name__=="__main__":
    print(sys.path)
    file_names = glob('/home/akshaya/Research/SeqManifold/v7/PlanningOnManifolds/cellular_automata/random_worlds/*/*.json')
    logger.debug(file_names[0])
    with open(file_names[1],'r') as f:
        data = json.load(f)
    logger.debug(data['obstacles'])

    env = Environment()

    meshs = env.get_obstacle_objs(data['obstacles'])
    print('mesh list:', meshs)
    env.plot_in_plotly(mesh_list=meshs)