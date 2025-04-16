#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 10:44:26 2023

@author: noahscheider
"""

import numpy as np
import scipy.stats as st
import scipy.sparse as spa
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns

from HJB_Functions import *



# Intiate true value function for any time and space grid
def TrueValueFunction(t_grid, x_grid, vers):
    """

    Parameters
    ----------
    t_grid : Array of float64
        discretized time grid.
    x_grid : Array of float64
        discretized space grid.
    vers : string
        type of forcing term.

    Returns
    -------
    Array of float64, shape(len(t_grid), len(x_grid))
        true value function evaluated on time and space grid.

    """
    
    v_true = spa.lil_matrix((len(t_grid), len(x_grid)))
    for n, t in enumerate(t_grid):
        if vers=="linear": 
            v_true[n,:] = t*x_grid
            
        if vers=="quadratic":
            v_true[n,:] = (t_grid[-1]-t)*x_grid**2
            
        if vers=="exp":
            v_true[n,:] = (t+1)*x_grid**3-x_grid**2/5+np.exp(t)
            
        if vers=="sqrt":
            v_true[n,:] = np.sqrt(t+.5)*np.cos(x_grid)
            
        if vers=="zero":
            v_true[n,:] = (t_grid[-1]-t)*np.cos(x_grid)
    
    print("-- True value function done --")
    return v_true.toarray()



# Compute forcing_term for the different value functions for verifiction purpuses
def ForcingTerm(t, b_t, c_t, p_t, x, sig, T, vers, d_grid, dxv=None, lam=None):
    """

    Parameters
    ----------
    t : float64
        time t.
    b_t : float64
        forecast for the residual demand at time t.
    c_t : float64
        price process for the grid power at time t.
    p_t : float64
        price process for diesel at time t.
    x : float64
        state of SDE.
    sig : float64
        diffusion constant of SDE.
    T : float64
        terminal time.
    vers : string
        type of forcing term.
    dxv : float64, optional
        derivative for "intermediate-hjb". The default is None.
    lam : float64 >= 0, optional
        penalization term for "intermediate-hjb". The default is None.

    Returns
    -------
    float64
        returns forcing term driving PDE, Regression Monte Carlo.

    """
    
    if vers=="linear":
        return -x-(b_t-x)*t
    
    elif vers=="quadratic":
        return x**2-(b_t-x)*(T-t)*2*x-(T-t)*sig**2
    
    elif vers=="exp":
        return -x**3-np.exp(t)-(b_t-x)*(3*(t+1)*x**2-2*x/5)-sig**2*(3*(t+1)*x-1/5)
    
    elif vers=="sqrt":
        return -np.cos(x)/(2*np.sqrt(t+.5))+(b_t-x)*np.sqrt(t+.5)*np.sin(x)+sig**2*np.sqrt(t+.5)*np.cos(x)/2
    
    elif vers=="zero":
        return np.cos(x)+(b_t-x)*(T-t)*np.sin(x)+sig**2*(T-t)*np.cos(x)/2
    
    elif vers[:4]=="hjb-":
        d_optimal = diesel(x, c_t, p_t, vers, d_grid[-1])
        return gain_function(x, c_t, p_t, d_optimal, vers)
    
    elif vers=="intermediate-hjb":
        d_grid_evaluated = np.array([(-np.log((1+np.exp(lam*(x-1-d)))/2))*dxv+gain_function(x, c_t, p_t, d, "hjb-quad") for d in d_grid])
        magic = np.argmax(d_grid_evaluated, axis=0)
        d_optimal = d_grid[magic]
        return d_grid_evaluated[magic, np.arange(len(x))], d_optimal
        

        
    else:
        print("-- No function defined --")



# Plotting the value function in 3d
def PlotValueFunction3d(t_grid, x_grid, v, title):
    """

    Parameters
    ----------
    t_grid : Array of float64
        discretized time grid.
    x_grid : Array of float64
        discretized space grid.
    v : Array of float64, shape(len(t_grid), len(x_grid))
        (Approximated) Value function.

    Returns
    -------
    3D-plot of (approximated) value function.

    """
    

    
    """ Create 3D surface plot """
    Y, X = np.meshgrid(t_grid, x_grid)
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=v.T)])
    
    # Customize plot
    fig.update_traces(colorscale='Viridis')
    fig.update_layout(scene=dict(xaxis_title='x', yaxis_title='t', zaxis_title='v(t,x)'), title=title)
    
    fig.show(renderer="browser")
    #fig.show()





# Plotting the value function as heatmap
def PlotValueFunctionHm(t_grid, x_grid, v, title, control=False, dmax=None, save=False):
    """

    Parameters
    ----------
    t_grid : Array of float64
        discretized time grid.
    x_grid : Array of float64
        discretized space grid.
    v : Array of float64, shape(len(t_grid), len(x_grid))
        (Approximated) Value function.
    title : string
        title of the plot
    control : boolean (optional)
        whether control or value function is plotted. The default is False.
    dmax : float64 (optional)
        maximum power from diesel generator. The default is None.
    save : boolean (optional)
        saves plot (True). The default is False.

    Returns
    -------
    heatmap of (approximated) value function.

    """
    
    
    """ Create a heatmap """
    sns.set()  # Set Seaborn's default style
    if control:
        hm = sns.heatmap(v, cmap="winter", fmt="d", vmin=0, vmax=dmax)
    else:
        hm = sns.heatmap(v, cmap="viridis", fmt="d", vmax=0, vmin=-41)
    
    # Setting axis
    x_grid_ticks = np.arange(0, len(x_grid)+1, len(x_grid)//9)
    hm.set_xticks(x_grid_ticks) # x-axis is space
    x_grid_tick_labels = [f"{x:.1f}" for x in x_grid[x_grid_ticks]]
    hm.set_xticklabels(x_grid_tick_labels, rotation=0)
    hm.set_xlabel("Space")
    
    t_grid_ticks = np.arange(0, len(t_grid), len(t_grid)//5)
    hm.set_yticks(t_grid_ticks) # y-axis is time
    t_grid_tick_labels = [f"{t:.1f}" for t in t_grid[t_grid_ticks]]
    hm.set_yticklabels(t_grid_tick_labels, rotation=0)
    hm.set_ylabel("Time")
    
    plt.title(title)
    
    if save: plt.savefig(f"Intermediate_Microgrid/ValueFunction_Plots/{title}.png")
    
    plt.show()
    
    
    
    