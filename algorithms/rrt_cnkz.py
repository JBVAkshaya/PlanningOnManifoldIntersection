import logging
from os import path
import numpy as np
from sympy import re
from lib.configuration_space import Qspace, ConstraintManifolds
import tqdm
from lib.tree import Tree
from lib import util

from lib.environment import Environment
import plotly.graph_objects as go
from math import ceil
import trimesh
import shapely.geometry as sg
import time
from lib.plot import plot, debug_collision_plot
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
try:
    from lib.robot import Robot
except ImportError as e:
    logger.error('import failed in rrtstar manifold module')
    logger.error(e)
except:
    logger.error("Unknown error in rrtstar manifold module.")


class RRTStarManifold:

    def __init__(self, rob_specs, robot_limits, robot_params, constraint_params, rrt_params, env_params, plot_params):
                #  min_robot_limits: np.ndarray,
                #  max_robot_limits: np.ndarray):
        self.rob_specs = rob_specs

        self.robots = Qspace(rob_specs = self.rob_specs,robot_limits=robot_limits, robot_params=robot_params,constraint_params=constraint_params, env_params=env_params)
        
        #TODO: rod_length may not be required
        self.num_robots = robot_params['num_bots']
        self.struct_type = robot_params['struct_type']
        # self.rod_length = robot_params['min_len']*self.num_robots
        
        #TODO: Start and goal state will come as input.
        self.start_q = self.robots.start_state

        eps_dict = {
            'm1':list(np.full((self.robots.mfolds.m1.y(self.start_q).shape[0],),
                      constraint_params['m1']['eps'])),
            'm2':list(np.full((self.robots.mfolds.m2.y(self.start_q).shape[0],),
                      constraint_params['m2']['eps'])),
            'm3':list(np.full((self.robots.mfolds.m3.y(self.start_q).shape[0],),
                      constraint_params['m3']['eps'])),
            'm4':list(np.full((self.robots.mfolds.m4.y(self.start_q).shape[0],),
                      constraint_params['m4']['eps']))
        }
        self.constraint_eps = []
        for k,v in eps_dict.items():
            self.constraint_eps.extend(v)
        self.constraint_eps = np.reshape(np.array(self.constraint_eps),
                                         (len(self.constraint_eps),1))

        self.goal_q = self.robots.goal_state
        
        self.task_goal_q = util.ee_position(self.robots.conf_rob, self.goal_q, rob_specs=self.rob_specs)

        # logging.info(f"ee: position: {self.task_goal_q}")
        self.n_samples = rrt_params['n_samples']
        self.d = self.start_q.shape[0]
        self.G = Tree(self.d, exact_nn=False)
        self.gamma = rrt_params['gamma']
        self.alpha = rrt_params['alpha'] ##### Right now direct radius to find nearest neighbor
        self.beta = rrt_params['beta'] # prb of sampling goal (Generally 0.1)
        self.collision_res = rrt_params['collision_res']
        self.proj_eps = rrt_params['proj_eps']
        self.proj_max_iter = rrt_params['proj_max_iter'] 
        self.conv_tol = rrt_params['converge_thresh'] 
        self.step_factor = rrt_params['step_factor']
        
        ##### Set Plotting params ######
        self.obs_three_d = plot_params['obs_3d']
        self.debug_collision = plot_params['debug_collision']

        ##### Set the Environment ######
        self.env = Environment()
        self.collision_meshs = self.env.get_obstacle_objs(env_params['obstacles'], obs_three_d = self.obs_three_d, z_translation = env_params['obs_z_translation'])
        self.struct_size = env_params['struct_size']
        self.obs_size = env_params['obs_size']
        # self.env_new.plot_in_plotly(mesh_list=self.collision_meshs)

        ##### Check start and goal ######
        
        # ##### Initialize center bots for collision check at various places ######
        # if self.struct_type=='S':
        #     tmp_1 = np.array([i for i in range (1,self.num_robots) if (i+1)%self.num_robots != 0])
        #     tmp_2 = tmp_1 + self.num_robots
        #     self.center_bots = np.append(tmp_1,tmp_2,0)
        # elif self.struct_type=='T':
        #     # tmp_1 = np.array([i for i in range (1,self.num_robots) if (i+1)%self.num_robots != 0])
        #     # tmp_2 = tmp_1 + self.num_robots
        #     self.center_bots = np.array([2,5])
        
        ##### recording results ######
        self.record_results = {
            'proj':{
                'time':[],
                'status':[]   
            },
            'collision':{
                'time':[],
                'status':[]
            }

        }

        logger.info(f"goal:")

##### Debug no collision ######
    def no_collision(self,
                     task_q_a: np.ndarray,
                     task_q_b: np.ndarray, is_rewire=False) -> bool:
        '''
        q_a: From coords
        q_b: to coords
        '''
        time_start = time.time()
        struct_source = self.env.get_structure_objs(state = np.reshape(task_q_a,(self.num_robots,3)), sides=self.struct_size)
        struct_target = self.env.get_structure_objs(state = np.reshape(task_q_b,(self.num_robots,3)), sides=self.struct_size)

        valid = self.env.no_continuous_collision(obs_list=self.collision_meshs,
                                            struct_source=struct_source,
                                            struct_target=struct_target)
        
        if is_rewire==False and self.debug_collision == True:
            fig = go.Figure()
            fig = debug_collision_plot(env = self.env,
                                obs_meshes = self.collision_meshs,
                                obs_size = self.obs_size,
                                start_struct_meshes=struct_source,
                                target_struct_meshes=struct_target,
                                struct_size=self.struct_size,
                                fig=fig)
            fig.show()
        # print(valid)
        # Visualize scene
        time_end = time.time()
        self.record_results['collision']['time'].append(time_end-time_start)
        self.record_results['collision']['status'].append(valid)
        return valid

    # TODO: Shift to environment.
    def debug_plot(self, q_a, q_b):
        fig = go.Figure()
        bot_color = [f'rgb({252},{186},{3})', f'rgb({2},{219},{252})', f'rgb({252},{94},{3})']
        q_a = np.reshape(q_a,(int(q_a.shape[0]/3),3))
        q_b = np.reshape(q_b,(int(q_b.shape[0]/3),3))
        fig.add_trace(go.Scatter3d(x = q_a[:,0], y=q_a[:,1], z = q_a[:,2], mode='markers+lines', marker=dict(color=bot_color,
                        size=7),
                        line = dict(color='firebrick', width=3),
                        name=str('q1')))
        fig.add_trace(go.Scatter3d(x = q_b[:,0], y=q_b[:,1], z = q_b[:,2], mode='markers+lines', marker=dict(color=bot_color,
                        size=7),
                        line = dict(color='firebrick', width=3),
                        name=str('q2')))
        for _,mesh in self.mesh_list.items():
            x,y,z = mesh.vertices.T
            i,j,k =mesh.faces.T
            fig.add_trace(go.Mesh3d(
                # All vertices of the mesh
                x=x,y=y,z=z,
                # # i, j and k give the vertices of triangles
                i = i,
                j = j,
                k = k,
                opacity=1.0,
                color='#DC143C',
                flatshading = True))
        fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1)))
        fig.show()

    def env_plot(self, fig):
        for _,mesh in self.mesh_list.items():
            x,y,z = mesh.vertices.T
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
        return fig
    
    def debug_collision_plot(self, q_a, q_b, mesh, fig):
        # fig = go.Figure()
        bot_color = [f'rgb({252},{186},{3})', f'rgb({2},{219},{252})', f'rgb({252},{94},{3})']
        q_a = np.reshape(q_a,(int(q_a.shape[0]/3),3))
        q_b = np.reshape(q_b,(int(q_b.shape[0]/3),3))
        fig.add_trace(go.Scatter3d(x = q_a[:,0], y=q_a[:,1], z = q_a[:,2], mode='markers+lines', marker=dict(color=bot_color,
                        size=7),
                        line = dict(color='firebrick', width=3),
                        name=str('q1')))
        fig.add_trace(go.Scatter3d(x = q_b[:,0], y=q_b[:,1], z = q_b[:,2], mode='markers+lines', marker=dict(color=bot_color,
                        size=7),
                        line = dict(color='firebrick', width=3),
                        name=str('q2')))
        # for _,mesh in self.mesh_list.items():
        try:
            fig = util.plotly_mesh(fig,mesh)
        except:
            pass
        fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.2)))
        return fig

    def rewire_task(self,
               q_from_id: int):

        for idx in self.Q_near_ids:  # Q_near_ids was previously computed in extend function
            task_q_idx = self.G.V[idx].task_value
            task_c_idx = self.G.V[idx].task_cost
            task_c_new = self.G.V[q_from_id].task_cost + np.linalg.norm(self.G.V[q_from_id].task_value.copy() - task_q_idx)
            conf_q_idx = self.G.V[idx].conf_value
            conf_c_new = self.G.V[q_from_id].conf_cost + np.linalg.norm(self.G.V[q_from_id].conf_value.copy() - conf_q_idx)

            # if not self.is_collision(q_from, q_idx) and c_new < c_idx:
            if self.no_collision(self.G.V[idx].task_value, self.G.V[q_from_id].task_value, is_rewire=True) and task_c_new < task_c_idx:
                idx_parent = self.G.V[idx].parent
                self.G.remove_edge(idx_parent, idx)
                self.G.add_edge(edge_id=self.G.edge_count, node_a=q_from_id, node_b=idx)
                self.G.V[idx].task_cost = task_c_new
                self.G.V[idx].conf_cost = conf_c_new
                self.G.V[idx].parent = q_from_id
                self.G.V[idx].task_inc_cost = np.linalg.norm(self.G.V[q_from_id].task_value.copy()  - task_q_idx)
                self.G.V[idx].conf_inc_cost = np.linalg.norm(self.G.V[q_from_id].conf_value.copy() - conf_q_idx)
                self.G.update_child_costs(node_id=idx)

                # check for convergence
                self.result = util.check_threshold(task_q_idx, self.task_goal_q, self.num_robots, self.conv_tol)


    def task_space_run(self, data):
        # Returns path node ids and #### not sure what all should be returned

        #TODO: Send Orientation
        
        t1 = time.time()
        self.G.add_node(node_id=0, conf_node_value=self.start_q, conf_node_cost=0.0, conf_inc_cost=0.0,
                        task_node_value = util.ee_position(self.robots.conf_rob, self.start_q,rob_specs=self.rob_specs) , 
                        orientation = util.ee_orient(self.robots.conf_rob, self.start_q,rob_specs=self.rob_specs),
                        task_node_cost=0.0, task_inc_cost=0.0)

        
        self.result = False

        pbar = tqdm.tqdm(total=self.n_samples)
        i = 0
        num_success = 0
        task_goal_q = util.ee_position(self.robots.conf_rob, self.goal_q,rob_specs=self.rob_specs)

        # logging.debug(f"goal task: {task_goal_q}")
        success_proj = 0
        ##### Check from here #####
        while (i< self.n_samples):
            
            if np.random.rand() < self.beta:
                conf_q_target = self.robots.get_biased_rand_config()
                logger.debug("moving towards goal")
            else:
                conf_q_target = self.robots.get_rand_config()
                logger.debug("exploring")
            task_q_target = util.ee_position(self.robots.conf_rob, conf_q_target,rob_specs=self.rob_specs)
            ### Get nearest neighbor in task space.
            task_q_near, q_near_id = self.G.get_task_nearest_neighbor(task_q_target)
            conf_q_near = self.G.V[q_near_id].conf_value
            # logging.debug(f"near: {task_q_near, q_near_id}")

            status, conf_q_new, _, time_taken = self.robots.get_projection_cnkz(self.G.V[q_near_id].conf_value, conf_q_target, self.constraint_eps, self.proj_max_iter, factor = self.step_factor)
            self.record_results['proj']['time'].append(time_taken)
            self.record_results['proj']['status'].append(status)
            
            # logging.error(f"status: {status}")
            task_q_new = util.ee_position(self.robots.conf_rob, conf_q_new,rob_specs=self.rob_specs)
            
            # logging.debug(f"lin norm: {np.linalg.norm(task_q_new-task_q_near)}")
            
            # logging.error(f" collision: {self.no_collision(task_q_new, task_q_near)}, Status: {status}")

            ### Delete below line should be in next if block. Just for checking...
            i = i+1
            ''' 
            TODO: Need to fix the collision method (no_collision()). 
            Mostly returning False even when dist > collision_res
            '''
            # if status == True:
            #     num_success = num_success + 1

            if status == True and self.no_collision(task_q_new, task_q_near):
                
                pbar.update()
                # i = i + 1
        #         q_new = np.array(q_new)
        #         print("got proj!!")
        #     #     q_new = self.steer(q_near, q_target)
                q_new_idx = self.G.node_count

                n = float(len(self.G.V))
                # r = min([self.gamma * np.power(np.log(n) / n, 1.0 / self.d), self.alpha])
                r = self.alpha
                # logging.debug(f"r:{r, self.gamma * np.power(np.log(n) / n, 1.0 / self.d), self.alpha}")
                
                self.Q_near_ids = self.G.get_task_nearest_neighbors(node_value=task_q_new, radius=r)
                
                # logging.debug(f"num near ids: {len(self.Q_near_ids)}")

                ### Add node to tree ###
                task_c_min = self.G.V[q_near_id].task_cost + np.linalg.norm(task_q_near - task_q_new)
                task_c_min_inc = np.linalg.norm(task_q_near - task_q_new)
                conf_c_min = self.G.V[q_near_id].conf_cost + np.linalg.norm(conf_q_near - conf_q_new)
                conf_c_min_inc = np.linalg.norm(conf_q_near - conf_q_new)
                q_min_idx = q_near_id

                #### Add New node and potential edge ####
                
                #TODO: Send Orientation
                self.G.add_node(node_id=q_new_idx, conf_node_value=conf_q_new, conf_node_cost=conf_c_min, conf_inc_cost= conf_c_min_inc,
                                task_node_value=task_q_new, 
                                orientation = util.ee_orient(self.robots.conf_rob, conf_q_new,rob_specs=self.rob_specs),
                                task_node_cost=task_c_min, task_inc_cost=task_c_min_inc)
                self.G.add_edge(edge_id=self.G.edge_count, node_a=q_min_idx, node_b=q_new_idx)
                logger.debug(f"cost: {task_c_min}")
                ### Rewiring Task Space ####
                # self.rewire_task(q_from_id=q_new_idx)
                success_proj = success_proj + 1
            else:
                logger.info("Proj not success")
        t2 = time.time()
        pbar.close()

        # try:
        self.G.comp_task_opt_path(task_goal_value= self.task_goal_q,
                                    conf_goal_value= self.goal_q,
                                    env = self.env,
                                    collision_meshes=self.collision_meshs,
                                    struct_size=self.struct_size,
                                    num_robots=self.num_robots,
                                    conv_tol=self.conv_tol)
        # except:
        #     logger.error("Failed in compute path")
        # logger.info(f"Costs when path computed for task space: task_cost->{task_cost}, Conf_cost->{conf_cost}")
        if len(self.G.path)>0:
            opt_path = [{
                        'task_conf':np.reshape(self.G.V[idx].task_value,(self.num_robots,3)),
                        'orient':self.G.V[idx].orient,
                        'conf_space':np.reshape(self.G.V[idx].conf_value,(self.num_robots,len(self.robots.conf_rob)))
                        } for idx in self.G.path]
            opt_path.append({
                        'task_conf':np.reshape(self.task_goal_q,(self.num_robots,3)),
                        'orient':util.ee_orient(self.robots.conf_rob,self.goal_q,rob_specs=self.rob_specs),
                        'conf_space':np.reshape(self.goal_q,(self.num_robots,len(self.robots.conf_rob)))
                        })
            try:
                plot(env= self.env, data = opt_path, mesh_list=self.collision_meshs, obs_size= self.obs_size, path=True, num_robots=self.num_robots, struct_type=self.struct_type)
            except:
                pass
            ## Export json file ##
            data['path'] = opt_path.copy()
            data['n_actual_nodes'] = success_proj
            data['time_taken'] = t2-t1
            data['collision'] = self.record_results['collision']
            data['projection'] = self.record_results['proj']

        # return num_success     
        return data








