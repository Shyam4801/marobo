import matplotlib.pyplot as plt
import numpy as np
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
# from .constants import NAME, H
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"


def vis_ei(x,ei_vals):
    (fig, ax) = plt.subplots(1, 2, figsize=(5, 5))
    # ei = ExpectedImprovement(gp, opt_domain)
    # ei_vals = ei.evaluate_at_point_list(x)

    # # We estimate EI2 using 100 MC iterations, using grid search of size 50 to maximize the inner EI
    # ei2 = RolloutEI(gp, opt_domain, horizon=2, opt_mode='grid', mc_iters=20, grid_size=100)
    # ei2_vals = ei2.evaluate_at_point_list(x)

    _ = ax[0].plot(x, ei_vals, '--g')
    # _ = ax[1].plot(x, ei2_vals, '--g')
    _ = ax[0].set_title('EI Acquisition')
    # _ = ax[1].set_title('EI2 Acquisition')
    plt.show()


    
def plot_obj(X,func,opt,xcord,ycord,init,nopred):
    xaxis = np.arange(xcord[0], xcord[1], 0.01)
    yaxis = np.arange(ycord[0], ycord[1], 0.01)
    # create a mesh from the axis
    x, y = meshgrid(xaxis, yaxis)
    # compute targets
    results = func(np.array([x,y]))
    # simulate a sample made by an optimization algorithm
    # seed(1)
    sample_x = X[:init,0] #r_min + rand(10) * (r_max - r_min)
    sample_y = X[:init,1] #r_min + rand(10) * (r_max - r_min)
    # create a filled contour plot with 50 levels and jet color scheme
    pyplot.figure(figsize=(5,5))
#     axis = figure.subplot(projection='3d')
    pyplot.contourf(x, y, results, levels=50, cmap='jet')
#     figure = pyplot.figure(figsize=(10,10))
#     axis = figure.add_subplot(projection='3d')
#     axis.plot_surface(x, y, results, cmap='jet')
    
    # define the known function optima
    optima_x = opt
    # draw the function optima as a white star
    pyplot.plot([optima_x[0]], [optima_x[1]], '*', color='white')
    # plot the sample as black circles
    pyplot.plot(sample_x, sample_y, 'o', color='black')
    rollx = X[init:,0] #xyrec[:,0] #
    rolly = X[init:,1] #xyrec[:,1] #
    pyplot.plot(rollx, rolly, 'x', color='white')
    pyplot.colorbar()
    # show the plot
    plt.savefig(str(H)+'_'+NAME+'_boplots1d_'+str(init)+'_'+str(nopred)+'.png')
    pyplot.show()


def plot_1d(X,func,opt,xcord,ycord,init,nopred):
    r_min, r_max = xcord, ycord
    xtrain = X[:,0]
    sc = lambda x: func(x) 
    ytrain = X[:,1]
    x = np.linspace(xcord, ycord, 40)[:, None]
    # sample input range uniformly at 0.01 increments
    inputs = np.arange(r_min, r_max, 0.01)
    # y_gp = gp.mean(x)
    # y_var = np.sqrt(gp.variance(x))
    # compute targets
    results = func([inputs]) #func(inputs)
    # print(results,inputs,xtrain,ytrain)
    # create a line plot of input vs result
    pyplot.figure(figsize=(8,5))
    pyplot.plot(inputs, results)
    # pyplot.plot(x, y_gp, color='r')
    print("xtrain: ",xtrain[init:])
    pyplot.plot(xtrain[:init], ytrain[:init], 'k.', markersize=15)
    pyplot.plot(xtrain[init:], ytrain[init:], '+', markersize=15)

    # define optimal input value
    x_optima = opt
    # draw a vertical line at the optimal input
    # pyplot.axvline(x=x_optima, ls='--', color='red')
    # pyplot.fill_between(x[:, 0], y_gp - y_var, y_gp + y_var, color='m', alpha=0.25)
    pyplot.legend(['True Function','Observations','Pred_obs'])
    # show the plot
    pyplot.savefig(str(H)+'_'+NAME+'_boplots1d_'+str(init)+'_'+str(nopred)+'.png')
    pyplot.show()


def sc():
    

    app = Dash(__name__)


    app.layout = html.Div([
        html.H4('Interactive scatter plot with Iris dataset'),
        dcc.Graph(id="scatter-plot"),
        html.P("Filter by petal width:"),
        dcc.RangeSlider(
            id='range-slider',
            min=0, max=2.5, step=0.1,
            marks={i: str(i) for i in range(0, 10, 1)},
            value=[0.5, 2]
        ),
        
    ])


    @app.callback(
        Output("scatter-plot", "figure"), 
        Input("range-slider", "value"))
    def update_bar_chart(slider_range):
        df = px.data.iris() # replace with your own data source
        low, high = slider_range
        mask = (df['petal_width'] > low) & (df['petal_width'] < high)
        fig = px.scatter(
            df[mask], x="sepal_width", y="sepal_length", 
            color="species", size='petal_length', 
            hover_data=['petal_width'])
        
        
        return fig


    app.run_server(debug=True)


def contour(agents, assignments, region, test_function, inactive_region_samples, sample, mins,minobs, fig = go.Figure()):
    app = Dash(__name__)

    print('sample ',sample)
    app.layout = html.Div([
        # html.H4('Interactive scatter plot with Iris dataset'),
        dcc.Graph(id="scatter-plot"),
        # html.P("Filter by petal width:"),
        dcc.RangeSlider(
            id='range-slider',
            min=1, max=sample, step=1,
            marks={i: str(i) for i in range(1, sample, 1)},
            value=[1, sample]
        ),
        # html.Div(id='scatter-plot')
    ])
    
    agent_hist = agents #test_function.agent_point_history
    print('b4 reshape :',agent_hist)
    # agent_hist = np.array(agent_hist)
    agent_hist = np.array(agent_hist).reshape((4*sample,2)) #[np.array(i).reshape((4,2)) for i in agent_hist ]
    # agent_hist = [i[-1] for i in agent_hist]
    print('after reshape ',agent_hist)
    print('inactive_region_samples:', inactive_region_samples)
    print('from test func agent hist: ',test_function.agent_point_history)
    # Generate data
    print(region)
    x = np.linspace(region[0,0], region[0,1], 100)
    y = np.linspace(region[1,0], region[1,1], 100)
    X, Y = np.meshgrid(x, y)
    # def internal_function(X):
    #         return X[0] ** 2 + X[1] ** 2 -1
    # x1 = X
    # x2 = Y
    # t = 1 / (8 * np.pi)
    # s = 10
    # r = 6
    # c = 5 / np.pi
    # b = 5.1 / (4 * np.pi ** 2)
    # a = 1
    # term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
    # term2 = s * (1 - t) * np.cos(x1)
    # l1 = 5 * np.exp(-5 * ((x1+3.14)**2 + (x2-12.27)**2))
    # l2 = 5 * np.exp(-5 * ((x1+3.14)**2 + (x2-2.275)**2))
    # Z =  term1 + term2 + s + l1 + l2 #
    # Z = X**4 + Y**4 - 4*X*Y + 1 #(X - 2)**2 + (Y - 2)**2
    # Z = (X - 2)**2 + (Y - 2)**2 
    # Z = X ** 2 + Y ** 2 -1
    # Z = (X[0]**2+X[1]-11)**2 + (X[0]+X[1]**2-7)**2
    Z = test_function(np.array([X,Y]), from_agent=None)
    # Create the figure
    # data = np.array(test_function.point_history)
    print('glob mins : ', mins, minobs)
    minobsx = np.array(minobs[:2])
    minobsy = np.array(minobs[2])

    
    @app.callback(
        Output("scatter-plot", "figure"), 
        Input("range-slider", "value"))
    def update_plot(slider_range):
        fig = go.Figure()

        fig.add_trace(go.Contour(
        x=x,
        y=y,
        z=Z,
        colorscale='Viridis'
        ))

        fig.add_trace(go.Scatter(
        x = [minobsx[0]],
        y = [minobsx[1]],
        mode='markers',
        marker=dict(
            color='green',
            symbol='star-open',
            size=10
        )
        ))
        
        fig.add_trace(go.Scatter(
        x = mins[0][:,0],
        y = mins[0][:,1],
        mode='markers',
        marker=dict(
            color='white',
            symbol='star-open',
            size=10
        )
        ))
        
        

        low, high = slider_range
        print('l - h - :',low,high)
        # mask = (df['petal_width'] > low) & (df['petal_width'] < high)
        # fig = px.scatter(
        #     df[mask], x="sepal_width", y="sepal_length", 
        #     color="species", size='petal_length', 
        #     hover_data=['petal_width'])
        if agent_hist != []:
        # for i in agent_hist[-1]:
            # print('agent y',agent_hist[high-1])
            fig.add_trace(go.Scatter(
                x=agent_hist[(low-1)*4:(high)*4,0],
                y=agent_hist[(low-1)*4:(high)*4,1],
                # Z = test_function,
                mode='markers',
                marker=dict(
                    color=['red', 'orange', 'blue', 'yellow', 'green'],
                    size=10
                )
            ))
        # print('inactive_region_samples[high-1]: high',high, inactive_region_samples[high-1])
        if (inactive_region_samples != []) and (inactive_region_samples[high-1].any() != None):
            fig.add_trace(go.Scatter(
                x=inactive_region_samples[high-1][:,0],
                y=inactive_region_samples[high-1][:,1],
                # Z = test_function,
                mode='markers',
                marker=dict(
                    color='white',
                    symbol= 'x',#['circle', 'square', 'diamond', 'cross', 'x'],
                    size=10
                )
            ))

        for i,t in enumerate(assignments[high-1]):
            min_x, max_x = t.input_space[0,0],t.input_space[0,1]
            min_y, max_y = t.input_space[1,0],t.input_space[1,1]

            
            if assignments[high-1][t] >= 1:
                fig.add_shape(
                type="rect",
                x0=min_x,
                y0=min_y,
                x1=max_x,
                y1=max_y,
                fillcolor="rgba(255, 0, 0, 0)",
                line_color="rgba(0, 0, 0, 0.5)"
                )
            else:
                fig.add_shape(
                type="rect",
                x0=min_x,
                y0=min_y,
                x1=max_x,
                y1=max_y,
                fillcolor="rgba(255, 0, 0, 0.4)",
                line_color="rgba(255, 0, 0, 0.5)"
                )

        fig.update_layout(
        title='Contour Plot',
        xaxis_title='X',
        yaxis_title='Y',
        hovermode='closest',
        # shapes=splits_lst
        height = 700,
        )
        fig.update_traces(hovertemplate='x: %{x}<br>y: %{y}<br>z: %{z:.2f}')
        # fig.update_shapes(selector=dict(type="rect"))
        fig.write_html('/Users/shyamsundar/ASU/sem2/RA/psytaliro_bo/results/x2y2_plot.html')

        return fig
    
    

    # splits = {}
    
        # splits['split_'+str(i)] = fig
        

    # splits_lst = list(splits.values())

    # @app.callback(
    #     Output("scatter-plot", "figure"), 
    #     Input("range-slider", "value"))
    # Add interactivity with a hover label
    
    # Display the figure
    # if sample == 4:
    # fig.show()
    

        # return fig
    # pio.orca.shutdown_server()

    app.run_server(debug=True)
    

# contour()

# sc()