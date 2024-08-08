from sre_constants import SUCCESS
import numpy as np
import copy
from math import inf
from sklearn.neighbors import KDTree as cKDTree
from lib import util

class Tree:
    """This class represents a directed tree that uses a KDTree for nearest neighbor queries."""
    def __init__(self,
                 d: int = 2,
                 exact_nn: bool = True,
                 abstract_root: bool = False):
        self.d = d
        self.V = dict()
        self.E = dict()
        self.path = []
        self.edge_count = 0
        self.node_count = 0
        self.exact_nn = exact_nn
        self.abstract_root = abstract_root

        self.kd_buffer_limit = 100
        self.kd_tree = None
        self.kd_off_ids = []

    def task_node_values(self) -> np.ndarray:
        """Returns an array of all node values (removing abstract root nodes)."""
        data = np.stack([v.task_value for v in self.V.values()])
        if self.abstract_root:
            return data[1:]
        else:
            return data
    
    def conf_node_values(self) -> np.ndarray:
        """Returns an array of all node values (removing abstract root nodes)."""
        data = np.stack([v.conf_value for v in self.V.values()])
        if self.abstract_root:
            return data[1:]
        else:
            return data
    
    def add_node(self,
                 node_id: int,
                 conf_node_value: np.ndarray,
                 conf_node_cost: float,
                 conf_inc_cost: float,
                 task_node_value: np.ndarray,
                 orientation: np.ndarray,
                 task_node_cost: float,
                 task_inc_cost: float):
        """Adds a new node to Tree."""
        if node_id in self.V:
            raise Exception('[Tree/add_node] Node id already exists.')

        self.V[node_id] = Node(node_id, conf_node_value, conf_node_cost, conf_inc_cost,
                                task_node_value,orientation, task_node_cost, task_inc_cost)
        self.node_count += 1
        self.kd_off_ids.append(node_id)

    def add_edge(self,
                 edge_id: int,
                 node_a: int,
                 node_b: int):
        """Adds new edge to Tree from node_a to node_b with id edge_id."""
        if edge_id in self.E:
            raise Exception('[Tree/add_edge] Edge id already exists.')

        self.E[edge_id] = Edge(edge_id, node_a, node_b)
        self.V[node_a].children.append(node_b)
        self.V[node_b].parent = node_a
        self.edge_count += 1

    def is_node(self, node_id: int) -> bool:
        """Checks if a node with id node_id exists."""
        if node_id in self.V:
            return True
        else:
            return False
    

    ##### This may have to be made into 2 functions for task and conf as well.
    def is_node_value(self,
                      node_value: np.ndarray,
                      tol: float = 1e-12) -> bool:
        """Check if a node with value node_value exists."""
        ##### Compare in configuration space
        ids = [v.id for v in self.V.values() if np.linalg.norm(v.conf_value - node_value) < tol]
        if len(ids) == 0:
            return False
        else:
            return True

    def is_edge(self,
                node_a: int,
                node_b: int) -> bool:
        """Check if an edge from node_a to node_b exists."""
        ids = [e.id for e in self.E.values() if (e.node_a == node_a and e.node_b == node_b)]

        if len(ids) == 0:
            return False
        else:
            return True

    def update_kd_task_tree(self):
        """Update kd tree for fast nearest neighbor computations."""
        if self.kd_tree is None or len(self.V) < self.kd_buffer_limit or (
                len(self.V) - len(self.kd_tree.data)) > self.kd_buffer_limit:
            self.kd_tree = cKDTree(self.task_node_values())
            self.kd_off_ids.clear()
    
    def update_kd_conf_tree(self):
        """Update kd tree for fast nearest neighbor computations."""
        if self.kd_tree is None or len(self.V) < self.kd_buffer_limit or (
                len(self.V) - len(self.kd_tree.data)) > self.kd_buffer_limit:
            self.kd_tree = cKDTree(self.conf_node_values())
            self.kd_off_ids.clear()

    def get_task_nearest_neighbor(self, node_value: np.ndarray) -> (np.ndarray, int):
        """Returns the node value and id of the nearest neighbor of node_value in the tree."""
        if self.exact_nn:
            near_node = min(self.V.values(), key=lambda v: np.linalg.norm(v.task_value - node_value))
        else:
            self.update_kd_task_tree()

            near_idx = self.kd_tree.query(X=[node_value], return_distance=False)
            if self.abstract_root: near_idx += 1
            near_idx = min(self.kd_off_ids + list(near_idx.flatten()), key=lambda idx: np.linalg.norm(self.V[idx].task_value - node_value))
            near_node = self.V[near_idx]

        return near_node.task_value, near_node.id
    
    def get_conf_nearest_neighbor(self, node_value: np.ndarray) -> (np.ndarray, int):
        """Returns the node value and id of the nearest neighbor of node_value in the tree."""
        if self.exact_nn:
            near_node = min(self.V.values(), key=lambda v: np.linalg.norm(v.conf_value - node_value))
        else:
            self.update_kd_conf_tree()

            near_idx = self.kd_tree.query(X=[node_value], return_distance=False)
            if self.abstract_root: near_idx += 1
            near_idx = min(self.kd_off_ids + list(near_idx.flatten()), key=lambda idx: np.linalg.norm(self.V[idx].conf_value - node_value))
            near_node = self.V[near_idx]

        return near_node.conf_value, near_node.id

    def get_task_nearest_neighbors(self,
                              node_value: np.ndarray,
                              radius: float) -> list:
        """Get the node indices of the nearest neighbors of node_value that are within distance radius."""
        if self.exact_nn:
            near_ids = [v.id for (k, v) in self.V.items() if np.linalg.norm(v.task_value - node_value) <= radius]
        else:
            self.update_kd_task_tree()

            near_ids = self.kd_tree.query_radius(X=[node_value], r=np.max([radius, 1e-12]), return_distance=False)[0]
            if self.abstract_root: near_ids += 1
            if len(near_ids) > 0:
                near_ids = [idx for idx in self.kd_off_ids + list(near_ids.flatten())
                            if np.linalg.norm(self.V[idx].task_value - node_value) <= radius]
            else:
                near_ids = [idx for idx in self.kd_off_ids
                            if np.linalg.norm(self.V[idx].task_value - node_value) <= radius]

        return near_ids
    
    def get_conf_nearest_neighbors(self,
                              node_value: np.ndarray,
                              radius: float) -> list:
        """Get the node indices of the nearest neighbors of node_value that are within distance radius."""
        if self.exact_nn:
            near_ids = [v.id for (k, v) in self.V.items() if np.linalg.norm(v.conf_value - node_value) <= radius]
        else:
            self.update_kd_conf_tree()

            near_ids = self.kd_tree.query_radius(X=[node_value], r=np.max([radius, 1e-12]), return_distance=False)[0]
            if self.abstract_root: near_ids += 1
            if len(near_ids) > 0:
                near_ids = [idx for idx in self.kd_off_ids + list(near_ids.flatten())
                            if np.linalg.norm(self.V[idx].conf_value - node_value) <= radius]
            else:
                near_ids = [idx for idx in self.kd_off_ids
                            if np.linalg.norm(self.V[idx].conf_value - node_value) <= radius]

        return near_ids

    def remove_edge_from_id(self, edge_id: int):
        """Removes the edge with edge_id from the tree."""
        if edge_id not in self.E:
            raise Exception('[tree/remove_edge] Edge is not in tree')
        self.V[self.E[edge_id].node_a].children.remove(self.E[edge_id].node_b)
        del self.E[edge_id]

    def remove_edge(self,
                    node_a_id: int,
                    node_b_id: int):
        """Removes the edge from node node_a_id to node_b_id from the tree."""
        ids = [e.id for e in self.E.values() if (e.node_a == node_a_id and e.node_b == node_b_id) or
               (e.node_a == node_b_id and e.node_b == node_a_id)]
        if len(ids) == 0:
            raise Exception('[tree/remove_edge] Edge is not in traph')
        self.V[node_a_id].children.remove(node_b_id)
        del self.E[ids[0]]

    def comp_path(self, node_idx: int) -> list:
        """Compute the path (list of node indices) from the root node the node with id node_idx."""
        # find nodes that reached the goal
        if not self.is_node(node_idx):
            raise Exception('node_idx is not in the Tree')

        # iterate through parent list until start is reached
        path = [node_idx]
        node_idx = self.V[node_idx].parent

        while node_idx != 0:
            path.append(node_idx)
            node_idx = self.V[node_idx].parent

        # check if root node is virtual
        if not np.isinf(self.V[0].task_value[0]):
            path.append(node_idx)

        path = list(reversed(path))
        return path

    
##### Debug no collision ######
    def is_collision(self,
                     task_q_a: np.ndarray,
                     task_q_b: np.ndarray, 
                     struct_size, 
                     num_robots, 
                     collision_meshs, 
                     env) -> bool:
        '''
        q_a: From coords
        q_b: to coords
        '''

        struct_source = env.get_structure_objs(state = np.reshape(task_q_a,(num_robots,3)), sides=struct_size)
        struct_target = env.get_structure_objs(state = np.reshape(task_q_b,(num_robots,3)), sides=struct_size)

        valid = env.no_continuous_collision(obs_list=collision_meshs,
                                            struct_source=struct_source,
                                            struct_target=struct_target)
        
        return not valid

    # def is_collision(self, task_q_a, task_q_b, center_bots):
    #     x_x1,y_x1,_ = np.transpose(np.reshape(task_q_a,(int(len(task_q_a)/3),3)))
    #     x_x2,y_x2,_ = np.transpose(np.reshape(task_q_b,(int(len(task_q_b)/3),3)))
    #     print("in is_col")
    #     ##### Check Environmental  collisions
    #     two_d_points = list(zip(x_x1, y_x1))
    #     two_d_points.extend(list(zip(x_x2, y_x2)))
    #     two_d_points = np.asarray(two_d_points)
    #     two_d_points = np.delete(two_d_points,center_bots, axis= 0)

    #     #### Check for self Collision ####
    #     return util.is_intersecting(two_d_points[0], two_d_points[1], two_d_points[3], two_d_points[2])

    def comp_task_opt_path(self,
                      task_goal_value,
                      conf_goal_value,
                      env,
                      collision_meshes,
                      struct_size, 
                      num_robots,
                      conv_tol=None) -> float:
        """Computes the shortest path (list of node indices) from root node to goal_value."""
        # find nodes that reached the goal
        V_near = []
        not_V_near = []
        success = False
        while(success==False):
            # print('Finding nearest node task to goal... ')
            if conv_tol:
                # temp = [np.linalg.norm(v.task_value - task_goal_value) for v in self.V.values()]

                V_near = [v for v in self.V.values() if util.check_threshold(v.task_value, task_goal_value, num_robots, conv_tol) and v not in not_V_near]
                if len(V_near) == 0:
                    return inf
                min_node = min(V_near, key=lambda v: v.task_cost)
            else:
                min_node = min([v for (k, v) in self.V.items() if k >= 0], key=lambda v: np.linalg.norm(v.task_value - task_goal_value))
            # print("here in collision")
            if self.is_collision( task_q_a = min_node.task_value,
                                task_q_b = task_goal_value,
                                struct_size= struct_size,
                                num_robots=num_robots,
                                collision_meshs=collision_meshes,
                                env=env)==True:
                # print("here in collision")
                not_V_near.append(min_node)
            else:
                # print("no collision")
                success = True
            # print("num v nodes:", len(V_near))
        # iterate through parent list until start is reached
        self.path = [min_node.id]
        node_id = min_node.parent

        while node_id is not None and node_id >= 0:
            self.path.append(node_id)
            node_id = self.V[node_id].parent

        self.path = list(reversed(self.path))
        return min_node.task_cost+np.linalg.norm(min_node.task_value - task_goal_value), min_node.conf_cost+np.linalg.norm(min_node.conf_value - conf_goal_value)
    
    # def comp_conf_opt_path(self,
    #                   goal_value: np.ndarray,
    #                   goal_task_val: np.ndarray,
    #                   center_bots:list,
    #                   conv_tol:
    #                   float = None) -> float:
    #     """Computes the shortest path (list of node indices) from root node to goal_value."""
    #     # find nodes that reached the goal
    #     V_near = []
    #     success = False
    #     while(success==False):
    #         print('Finding nearest node conf to goal... ')
    #         if conv_tol:
    #             V_near = [v for v in self.V.values() if np.linalg.norm(v.conf_value - goal_value) <= conv_tol]
    #             if len(V_near) == 0:
    #                 return inf
    #             min_node = min(V_near, key=lambda v: v.conf_cost)
    #         else:
    #             min_node = min([v for (k, v) in self.V.items() if k >= 0], key=lambda v: np.linalg.norm(v.conf_value - goal_value))

    #         if self.is_collision(min_node.task_value, goal_task_val, center_bots=center_bots)==True:
    #             V_near.remove(min_node)
    #         else:
    #             success = True
    #         print("num v nodes:", len(V_near))
    #     # iterate through parent list until start is reached
    #     self.path = [min_node.id]
    #     node_id = min_node.parent

    #     while node_id is not None and node_id >= 0:
    #         self.path.append(node_id)
    #         node_id = self.V[node_id].parent

    #     self.path = list(reversed(self.path))
    #     return min_node.task_cost, min_node.conf_cost

    def update_child_costs(self, node_id: int):
        """Update costs of child nodes of node_id (used in RRT* rewiring)."""
        curr_s = copy.copy(self.V[node_id].children)

        while len(curr_s) > 0:
            child_id = curr_s.pop()
            parent_id = self.V[child_id].parent
            self.V[child_id].task_cost = self.V[parent_id].task_cost + \
                                    np.linalg.norm(self.V[child_id].task_value - self.V[parent_id].task_value)
            self.V[child_id].conf_cost = self.V[parent_id].conf_cost + \
                                    np.linalg.norm(self.V[child_id].conf_value - self.V[parent_id].conf_value)
            curr_s.extend(self.V[child_id].children)

    def sample_node(self) -> int:
        """Randomly samples a node in the tree and keeps track how often it has been sampled. Returns the index of the
        sampled node in the tree.
        """
        prob = [v.n_sampled for v in self.V.values()]
        prob = 1. / (np.array(prob) + 1.) ** 2
        prob = prob / np.sum(prob)
        idx = np.random.choice(a=prob.shape[0], p=prob)
        self.V[idx].n_sampled += 1
        return self.V[idx].id


class Node:
    def __init__(self,
                 id: int,
                 conf_value: np.ndarray,
                 conf_cost: float,
                 conf_inc_cost: float,
                 task_value: np.ndarray,
                 orientation: np.ndarray,
                 task_cost: float,
                 task_inc_cost: float):
        self.id = id
        self.conf_value = conf_value # Configuration space robot state
        self.task_value = task_value # Task space robot state
        self.orient = orientation # Orientation of end effector
        self.parent = None
        self.children = []
        self.conf_cost = conf_cost           # costs from root to node in configuration space
        self.conf_inc_cost = conf_inc_cost   # costs from parent to node in configuration space
        self.task_cost = task_cost           # costs from root to node in task space
        self.task_inc_cost = task_inc_cost   # costs from parent to node in task space
        self.con_extend = False    # property used by PSM to determine if node was already extended
        self.aux = None            # variable used to store additional information
        self.n_sampled = 0         # how often the node has been sampled already
        self.path = None

    def __str__(self):
        return "id: " + str(self.id) + ", conf value: " + str(self.conf_value) \
                + ", orientation: " + str(self.orient) + ", task value: " + str(self.task_value) + ", parent: " + \
               str(self.parent) + ", conf cost: " + str(self.conf_cost) + ", task cost: " + str(self.task_cost) 


class Edge:
    def __init__(self,
                 id: int,
                 node_a: int,
                 node_b: int):
        self.id = id
        self.node_a = node_a
        self.node_b = node_b

    def __str__(self):
        return "id: " + str(self.id) + ", node_a: " + str(self.node_a) + ", node_b: " + str(self.node_b)
