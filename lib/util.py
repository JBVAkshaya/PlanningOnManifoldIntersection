import logging
import autograd.numpy as gnp
from autograd import grad, jacobian
import numpy as np
import trimesh
import shapely.geometry as sg
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from robot_configs.uvms import Uvms
from robot_configs.turtlebot_manipulator import TurtlebotManipulator

def trans_mat_single_bot_ee_orient(dict_var, rob_specs):
    # dict_var: variable names for rov configurations for a robot
    if rob_specs['type']=='UVMS':
        return Uvms.ee_orient(dict_var=dict_var, rob_specs=rob_specs)
    elif rob_specs['type']=='TB3OM':
        return TurtlebotManipulator.ee_orient(dict_var=dict_var, rob_specs=rob_specs)
    else:
        raise LookupError

def trans_mat_single_bot_ee_position(dict_var, rob_specs):
    if rob_specs['type']=='UVMS':
        return Uvms.forward_kinematics(dict_var=dict_var, rob_specs=rob_specs)
    elif rob_specs['type']=='TB3OM':
        return TurtlebotManipulator.forward_kinematics(dict_var=dict_var, rob_specs=rob_specs)
    else:
        raise LookupError
  
def rob_vars(conf, num):
    eval_on = {}
    for i in range(1,num+1):
        eval_on[str(i)] = {}
        for key in conf:
            tmp = key +"_" + str(i)
            eval_on[str(i)][key] = tmp
    return eval_on

def robs_config(conf, samps):
    dict_samps = {}
    for i in range(0,len(samps)):
        for idx,key in enumerate(conf):
            tmp = key +"_" + str(i+1)
            dict_samps[tmp] = samps[i][idx]
    return dict_samps

def robs_config_num (conf, samp, num):
    dict_samp = {}
    for idx,key in enumerate(conf):
        tmp = key +"_" + str(num)
        dict_samp[tmp] = samp[tmp]
    return dict_samp

def y(equations, eval_on):
    f = lambdify(list(eval_on.keys()), equations)
    f_val = f(*eval_on.values())
    return f_val

## Make this work
def J(cost_autograd, x):
    jacobian_cost = jacobian(cost_autograd)
    return jacobian_cost(x)

def ee_orient(dofs, q, rob_specs):
    arr = np.empty([0, 4])
    
    for i in range(0,int(q.shape[0]/len(dofs))):
        dict_q = dict(zip(dofs,q[i*len(dofs):(i+1)*len(dofs)])) 
        orient = np.transpose(trans_mat_single_bot_ee_orient(dict_q, rob_specs))
        arr = np.append(arr, orient, axis = 0)
    ## TODO: delete axis 3 (Column). Make this change everywhere.
    return arr[:,:-1]

def ee_position(dofs, q, rob_specs) -> (np.ndarray):
    # logging.info("ee")
    arr = np.empty([0, 4])

    for i in range(0,int(len(q)/len(dofs))):
        dict_q = dict(zip(dofs,q[i*len(dofs):(i+1)*len(dofs)])) 
        arr = np.append(arr, np.transpose(trans_mat_single_bot_ee_position(dict_q, rob_specs=rob_specs)), axis = None)
        arr = np.delete(arr,[arr.shape[0]-1],0)

    return arr

def get_point_distance(point_coord, line_a, line_b):
    line = sg.LineString([line_a, line_b]) 
    return line.distance(sg.Point(point_coord))

def get_line_length(line_a, line_b):
    line = sg.LineString([line_a, line_b])
    return line.length

def get_point_at_distance_on_line(line_a, line_b, dist):
    line = sg.LineString([line_a, line_b])
    return line.interpolate(dist)

def is_intersecting(line1_a, line1_b, line2_a, line2_b) -> (bool):
    l1_a = sg.Point(line1_a[0],line1_a[1])
    l1_b = sg.Point(line1_b[0],line1_b[1])
    l2_a = sg.Point(line2_a[0],line2_a[1])
    l2_b = sg.Point(line2_b[0],line2_b[1])
    line1 = sg.LineString([l1_a,l1_b]).buffer(0.01)
    line2 = sg.LineString([l2_a,l2_b]).buffer(0.01)
    return line2.intersects(line1)

def get_intersection(line1_a, line1_b, line2_a, line2_b):
    l1_a = sg.Point(line1_a[0],line1_a[1])
    l1_b = sg.Point(line1_b[0],line1_b[1])
    l2_a = sg.Point(line2_a[0],line2_a[1])
    l2_b = sg.Point(line2_b[0],line2_b[1])
    line1 = sg.LineString([l1_a,l1_b])
    line2 = sg.LineString([l2_a,l2_b])
    int_pt = line1.intersection(line2)
    return int_pt.x, int_pt.y

def get_polygon_mesh (shell_vertices, mesh_height = -0.1):
    '''
    shell_vertices: should be in anticlockwise direction. 
    holes: is a list of arrays representing holes
    '''
    two_d_poly = sg.Polygon(shell=shell_vertices)

    status = two_d_poly.is_valid
    logging.debug(f"is the polygon valid: {two_d_poly.is_valid}")
    if status == True:
        return status, trimesh.creation.extrude_polygon(two_d_poly, mesh_height)
    else:
        return status, None

def get_triangular_mesh(shell_vertices, mesh_height = -0.1):
    '''
    shell_vertices: should be in anticlockwise direction. 
    '''
    two_d_triangle = sg.Polygon(shell=shell_vertices).buffer(0)
    logging.debug(f"is the polygon valid: {two_d_triangle.is_valid}")
    return trimesh.creation.extrude_polygon(two_d_triangle, mesh_height)

def plotly_mesh(fig, mesh):
    x,y,z = mesh.vertices.T
    i,j,k = mesh.faces.T
    fig.add_trace(go.Mesh3d(
        x=x,y=y,z=z,
        # # i, j and k give the vertices of triangles
        i = i,
        j = j,
        k = k,
        opacity=0.7,
        color='#2ca02c',
        flatshading = True))
    return fig

def check_threshold(task_a, task_goal, num_bots, thresh):
    valid = False
    task_a = np.reshape(task_a,(num_bots,3))
    task_goal = np.reshape(task_goal,(num_bots,3))
    
    for i in range (0,task_a.shape[0]):
        if np.linalg.norm(task_a[i] - task_goal[i]) < thresh:
            valid = True
    return valid