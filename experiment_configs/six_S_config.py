num_robots = 6
struct_type = 'S'
rob_specs = {
    'b_x':0.6,
    'b_y':0.6,
    'b_z':0.6,
    'l_1':0.3,
    'l_2':0.1,
    'base_z':0.0, # Update this
    'type':'UVMS', # can be UVMS, TB3OM
    'dof': ['x', 'y', 'z', 'yaw', 'q1', 'q2']

}

robot_params = {
    'num_bots':num_robots,
    'struct_type': struct_type,
    'base_z':0.0,
    'dof': ['x', 'y', 'z', 'yaw', 'q1', 'q2'],
    'min_len':0.5
}

constraint_params = {
    'm1': {
            'min_len': 0.5,
            'struct_type': struct_type,
            'num_robots': num_robots,
            'penalty_factor': 3.0,
            'eps': 0.3
        },
    'm2': {
            'min_len': 0.5,
            'struct_type': struct_type,
            'num_robots': num_robots,
            'penalty_factor': 1.4,
            'eps': 0.3
        },
    'm3': {
            'min_len': 0.5,
            'struct_type': struct_type,
            'num_robots': num_robots,
            'penalty_factor': 3.0,
            'eps': 0.3
        },
    'm4': {
            'min_len': 0.5,
            'struct_type': struct_type,
            'num_robots': num_robots,
            'penalty_factor': 3.0,
            'eps': 0.3
        }
    }

rrt_params = {
    'n_samples' : 100,
    'gamma': 2.0,
    'alpha': 7.0 ,##### Right now direct radius to find nearest neighbor
    'beta': 0.5, # prb of sampling goal (Generally 0.1)
    'collision_res': 0.1,
    'proj_eps': 0.3, #### This will change as number of robots will increase
    'proj_max_iter': 700,
    'converge_thresh':6.0,
    'step_factor': 2.2
} 

plot_params = {
    'debug_collision':False,
    'obs_3d':True, 
    'struct_type': struct_type
}
debug_flag = False
