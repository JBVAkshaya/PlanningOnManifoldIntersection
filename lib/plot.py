import plotly.graph_objects as go
from lib import util
import numpy as np
import plotly.express as px
import time


def plot_obstacles(mesh_list, fig):
    for _,mesh in mesh_list.items():
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

def plot_cuboid_plotly(fig, points, color, opacity = 1.0):
    x,y,z = points[:,0],points[:,1],points[:,2]
    fig.add_trace(go.Mesh3d(
        # 8 vertices of a cube
        x=x,y=y,z=z,
        # # i, j and k give the vertices of triangles
        i = [7, 1, 0, 1, 4, 5, 6, 6, 2, 4, 3, 1],
        j = [3, 5, 1, 2, 5, 6, 4, 2, 0, 1, 6, 0],
        k = [1, 7, 2, 3, 6, 7, 2, 3, 4, 5, 7, 4],
        opacity = opacity,
        color = color,
        flatshading = True))
    return fig

def plot_fcl_obstacles_plotly(env, obs_meshes, obs_size, fig, color='#AB9278'):
    for _,mesh in obs_meshes.items():
        points = env.gen_bb_points(sides=obs_size,orient=mesh.getRotation(),pos=mesh.getTranslation())
        fig = plot_cuboid_plotly(fig=fig,points=points,color=color)
    return fig

def plot_structure_plotly(env, struct_meshes, struct_size, fig, color='#2CA02C', opacity = 0.6):
    for _,mesh in struct_meshes.items():
        points = env.gen_bb_points(sides=struct_size,orient=mesh.getRotation(),pos=mesh.getTranslation())
        fig = plot_cuboid_plotly(fig=fig,points=points,color=color, opacity = 0.6)
        ## plot points of structure
    return fig

def debug_collision_plot(env, obs_meshes, obs_size, fig, start_struct_meshes, target_struct_meshes, struct_size):
    
    ## Plot obstacles:

    fig = plot_fcl_obstacles_plotly(env, obs_meshes, obs_size, fig)
    
    ## Call plot Structure. Reduce the opacity and change the obstacle color
    fig = plot_structure_plotly(env, start_struct_meshes, struct_size, fig)
    fig = plot_structure_plotly(env, target_struct_meshes, struct_size, fig)
    
    fig.update_layout(scene = dict(
            # zaxis = dict(showticklabels=False),
            # yaxis = dict(nticks = 4, range = [0,5]),
            aspectratio=dict(x=1, y=2, z=1),
            zaxis_title='Z',
            xaxis_title='X',
            yaxis_title='Y'), 
            showlegend=False)

    return fig


def plot(env, data, mesh_list, obs_size, path=True, num_robots = 3, struct_type ='S'):

    t1 = time.time()    
    fig = go.Figure()

    r = []
    for i in range (0, num_robots):
        r.append([])
    
    bot_color = px.colors.diverging.Portland


    for i in range(0, len(data)):
        # Input should be end effector state.

        
        data[i]['task_conf'] = np.floor(100.0*np.array(data[i]['task_conf']))
        data[i]['task_conf'] = data[i]['task_conf']/100.0

        data[i]['orient'] = np.floor(100.0*np.array(data[i]['orient']))
        data[i]['orient'] = data[i]['orient']/100.0
        

        x_1 = data[i]['task_conf'].copy()

        if struct_type=='S':
            fig.add_trace(go.Scatter3d(x = x_1[:,0], y=x_1[:,1], z = x_1[:,2], mode='markers+lines', marker=dict(color=bot_color,
                    size=3),
                    line = dict(color='firebrick', width=6),
                    name=str(i)))
        elif struct_type=='T':
            fig.add_trace(go.Scatter3d(x = x_1[:,0], y=x_1[:,1], z = x_1[:,2], mode='markers', marker=dict(color=bot_color,
                    size=3),
                    name=str(i)))
            fig.add_trace(go.Scatter3d(x = x_1[:2,0], y=x_1[:2,1], z = x_1[:2,2], mode='lines',
                    line = dict(color='firebrick', width=6)))
            fig.add_trace(go.Scatter3d(x = [(x_1[0,0] + x_1[1,0])/2.0,x_1[-1,0]], y=[(x_1[0,1]+ x_1[1,1])/2.0,x_1[-1,1]], z = [(x_1[0,2] + x_1[1,2])/2.0,x_1[-1,2]], mode='lines',
                        line = dict(color='firebrick', width=6)))
        
        elif struct_type=='I':
            fig.add_trace(go.Scatter3d(x = x_1[:,0], y=x_1[:,1], z = x_1[:,2], mode='markers', marker=dict(color=bot_color,
                    size=3),
                    name=str(i)))
            fig.add_trace(go.Scatter3d(x = x_1[:2,0], y=x_1[:2,1], z = x_1[:2,2], mode='lines',
                    line = dict(color='firebrick', width=6)))
            fig.add_trace(go.Scatter3d(x = [(x_1[0,0] + x_1[1,0])/2.0,x_1[2,0]], y=[(x_1[0,1]+ x_1[1,1])/2.0,x_1[2,1]], z = [(x_1[0,2] + x_1[1,2])/2.0,x_1[2,2]], mode='lines',
                        line = dict(color='firebrick', width=6)))
            
            fig.add_trace(go.Scatter3d(x = x_1[3:,0], y=x_1[3:,1], z = x_1[3:,2], mode='lines',
                    line = dict(color='firebrick', width=6)))
            fig.add_trace(go.Scatter3d(x = [(x_1[3,0] + x_1[4,0])/2.0,x_1[2,0]], y=[(x_1[3,1]+ x_1[4,1])/2.0,x_1[2,1]], z = [(x_1[3,2] + x_1[4,2])/2.0,x_1[2,2]], mode='lines',
                        line = dict(color='firebrick', width=6)))

        for j in range(0,num_robots):
            fig.add_trace(go.Cone(
            x=[data[i]['task_conf'][j,0]],
            y=[data[i]['task_conf'][j,1]],
            z=[data[i]['task_conf'][j,2]],
            u=[data[i]['orient'][j,0]],
            v=[data[i]['orient'][j,1]],
            w=[data[i]['orient'][j,2]],
            colorscale='viridis',
            sizemode="absolute",
            sizeref=0.1,
            anchor="tip", showscale=False))

            r[j].append(x_1[j,:])
    
    for i in range (0, num_robots):
        t = np.asarray(r[i])
        r[i] = t.copy()   

    color = f'rgb({140},{150},{150})'
    
    if path==True:
        for i in range (0, num_robots):
            fig.add_trace(go.Scatter3d(x = r[i][:,0], y=r[i][:,1], z = r[i][:,2], mode='lines', 
            line = dict(color=color, width=4, dash='dash'), name='r' + str(i)))
    
    # i = int((num_robots-1)/2)
    # fig.add_trace(go.Scatter3d(x = r[i][:,0], y=r[i][:,1], z = r[i][:,2], mode='lines', 
    #         line = dict(color=color, width=4, dash='dash'), name='r' + str(i)))

    # fig = plot_obstacles(mesh_list=mesh_list,fig=fig)

    fig = plot_fcl_obstacles_plotly(env = env, obs_meshes = mesh_list, obs_size = obs_size , fig=fig)

    fig.update_layout(scene = dict(
            # zaxis = dict(showticklabels=False),
            # yaxis = dict(nticks = 4, range = [0,5]),
            aspectratio=dict(x=1, y=1, z=1),
            zaxis_title='Z',
            xaxis_title='X',
            yaxis_title='Y'), 
            showlegend=False)
    fig.update_xaxes(ticklabelposition="inside top")
    fig.update_yaxes(ticklabelposition="inside top")
    # fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.2)))
    t2 = time.time()
    print(t2-t1)
    fig.show()
    return fig
