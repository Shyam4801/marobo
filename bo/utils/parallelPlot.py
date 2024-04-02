

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import yaml

with open('config.yml', 'r') as file:
    configs = yaml.safe_load(file)

def plot_parallel_coord(pts, dims, y, name):
    print( y ,name)
    fig = px.parallel_coordinates(pts, color=y.reshape(pts.shape[0]),
                                dimensions=dims,
                                color_continuous_scale=px.colors.diverging.Portland,
                                color_continuous_midpoint=2)
    
    fig.write_html(f"results/pcoord/{name}.html")

def plot_agents_parallel_coord(agents, dims, name):
    fig = go.Figure()
    pts = np.empty((0,agents[0].x_train.shape[1]))
    # ptsy = np.empty((0,agents[0].x_train.shape[1]+1))
    y = np.empty((0))
    id = []
    for i, agent in enumerate(agents):
        pts = np.vstack((pts, agent.x_train[-1]))
        
        y = np.hstack((y, [agent.y_train[-1]]))

        # print(pts, y)
        id.extend([agent.id]*len([agent.y_train[-1]]))

    id.extend([-1])
    ptsy = np.column_stack((pts, y))
    ptsy = np.vstack((ptsy, configs['globmin']))

    

    fig = px.parallel_coordinates(ptsy, color= id, #y.reshape(pts.shape[0]),
                                        dimensions=dims,
                                        color_continuous_scale=px.colors.diverging.Portland,
                                        color_continuous_midpoint=2)
    
    fig.write_html(f"results/pcoord/{name}.html")

def plot_all_parallel_coord(agents, dims, name):
    fig = go.Figure()
    pts = np.empty((0,agents[0].x_train.shape[1]))
    # ptsy = np.empty((0,agents[0].x_train.shape[1]+1))
    y = np.empty((0))
    id = []
    for i, agent in enumerate(agents):
        pts = np.vstack((pts, agent.x_train))
        
        y = np.hstack((y, agent.y_train))

        # print(pts, y)
        id.extend([agent.id]*len(agent.y_train))

    id.extend([-1])
    ptsy = np.column_stack((pts, y))
    ptsy = np.vstack((ptsy, configs['globmin']))

    

    fig = px.parallel_coordinates(ptsy, color= id, #y.reshape(pts.shape[0]),
                                        dimensions=dims,
                                        color_continuous_scale=px.colors.diverging.Portland,
                                        color_continuous_midpoint=2)
    
    fig.write_html(f"results/pcoord/{name}.html")

    

def plot_smp_parallel_coord(agents, dims, name):
    fig = go.Figure()
    pts = np.empty((0,agents[0].region_support.smpXtr.shape[1]))
    # ptsy = np.empty((0,agents[0].x_train.shape[1]+1))
    y = np.empty((0))
    id = []
    for i, agent in enumerate(agents):
        pts = np.vstack((pts, agent.region_support.smpXtr))
        
        y = np.hstack((y, agent.region_support.smpYtr))

        # print(pts, y)
        id.extend([agent.id]*len(agent.region_support.smpYtr))
    ptsy = np.column_stack((pts, y))
    

    fig = px.parallel_coordinates(ptsy, color= id, #y.reshape(pts.shape[0]),
                                        dimensions=dims,
                                        color_continuous_scale=px.colors.diverging.Portland,
                                        color_continuous_midpoint=2)
    
    fig.write_html(f"results/pcoord/smpXtr{name}.html")