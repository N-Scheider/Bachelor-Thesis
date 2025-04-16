#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 10:35:45 2023

@author: noahscheider
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt



# Defining multiple cost function as in thesis
def gamma(x, d, vers):
    if vers=="hjb-lin": return x-d    
    if vers=="hjb-quad": return (x-d)**2
    
    # # Old Stuff
    # if vers=="hjb-abs": return np.abs(x-d)
    # if vers=="hjb-max": return np.maximum(0, x-d)

    

# Defining function that returns liter depending on amount of diesel power we want to generate
def rho(d):
    return ((d-6)**3+6**3+d)/10


# Gain function or cost function to the corresponding HJB formulation
def gain_function(x, c_t, p_t, d, vers):
    return -p_t*rho(d)-c_t*gamma(x, d, vers)





# Define optimal control policy for the diesel generator based on maximizing HJB
def diesel(x, c_t, p_t, vers, dmax, dxv=None, lam=None):
    """

    Parameters
    ----------
    x : Array of float64
        states of SDE.
    c_t : float64
        price process for the grid power at time t.
    p_t : float64
        price process for diesel at time t.
    vers : string
        type of forcing term.
    dxv : float64, optional
        derivative for "intermediate-hjb". The default is None.
    lam : float64 >= 0, optional
        penalization term for "intermediate-hjb". The default is None.
    dmax : float64, optional
        maximum diesel power generatable from generator. The default is 10.

    Returns
    -------
    float64
        returns optimal control policy for current state.

    """
    
    
    # Control policy with linear cost function
    if vers=="hjb-lin":
        if p_t==0:
            return dmax
        elif 10*c_t<p_t or c_t==0:
            return 0
        else:
            #return np.sqrt((10*c_t/p_t-1)/3)+6
            d_arr = np.array([0, np.minimum(np.sqrt((10*c_t/p_t-1)/3)+6, dmax)])
            d_arr_evaluated = np.array([gain_function(x, c_t, p_t, d, "hjb-lin") for d in d_arr]) # for x array returns x_d1 over x_d2 over x_d3 ...
            max_index = np.argmax(d_arr_evaluated, axis=0)
            return d_arr[max_index]
    
    
    
    # Control policy with quadratic cost function
    elif vers=="hjb-quad":
        
        if p_t == 0:
            return np.minimum(np.maximum(0,x), dmax)

        # vectorized conditions
        ordering = (36*p_t-20*c_t)**2 >= 12*p_t*(109*p_t-20*c_t*x)
        try:
            d_optimal = np.zeros(len(ordering))
            
            # Extract residual demands that match condition
            x_pos_ordering = x[ordering]
            
            # Only perform maximization, where the variable ordering is true
            d_1 = np.minimum(np.maximum(0, ((36*p_t-20*c_t)+np.sqrt((36*p_t-20*c_t)**2-12*p_t*(109*p_t-20*c_t*x_pos_ordering)))/(6*p_t)), dmax)
            d_arr = np.array([np.repeat(0, len(d_1)), d_1, np.repeat(dmax, len(d_1))])
            d_arr_evaluated = np.array([gain_function(x_pos_ordering, c_t, p_t, d, "hjb-quad") for d in d_arr])
            max_index = np.argmax(d_arr_evaluated, axis=0)
            
            d_optimal[ordering] = d_arr[max_index, np.arange(len(max_index))] # other cases are already covered by the initialization of d_optimal=0, d_optimal[~ordering] = 0
            
            return d_optimal
        
        # When x is only an integer, for instance in Initiate processes function
        except:
            print("Except")
            if (36*p_t-20*c_t)**2 >= 12*p_t*(109*p_t-20*c_t*x):
                d_1 = np.minimum(np.maximum(0, ((36*p_t-20*c_t)+np.sqrt((36*p_t-20*c_t)**2-12*p_t*(109*p_t-20*c_t*x)))/(6*p_t)), dmax)
                d_arr = np.array([0, d_1, dmax])
                d_arr_evaluated = np.array([gain_function(x, c_t, p_t, d, "hjb-quad") for d in d_arr])
                max_index = np.argmax(d_arr_evaluated, axis=0)
                return d_arr[max_index]
            else:
                return 0
                
            


# Initiate forecasting process b, and price processes c and p
def InitiateProcesses(t_grid, sig, vers=None, dmax=None, seed=False, display=False, trajectory=None, diesel_policy=None, save=False):
    """

    Parameters
    ----------
    t_grid : Array of float64
        discretized time grid.
    sig : float64
        diffusion constant for SDE.
    vers : string, optional
        type of forcing term. The default is None.
    dmax : boolean, optional
        maximum diesel power. The default is None.
    seed : boolean, optional
        random seed. The default is False.
    display : boolean, optional
        display the process for intuition. The default is False.
    trajectory : Array of float64
        previously simulated trajectory (in intermediate case). The default is None.
    diesel_policy: Array of float64
        previously simulated diesel policy to trajectory (in intermediate case). The default is None.
    save : boolean
        saves fig or not

    Returns
    -------
    three/four Arrays of float64, shape(len(t_grid), 1)
        forecast of residual demand, price process for the grid, price process for diesel.

    """
        
    # Initiate whether random seed is turned on or off
    if seed: np.random.seed(seed=90)
    
    # Initiate processes
    b, c, p, x = np.zeros(len(t_grid)), np.zeros(len(t_grid)), np.zeros(len(t_grid)), np.zeros(len(t_grid))
    dt = t_grid[1]-t_grid[0]

    # Initial condition
    c[0] = 1
    p[0] = 1.25
    x[0] = 0
    
    for n, t in enumerate(t_grid[:-1]):
        b[n+1] = 6*np.sin(np.pi*t)
        c[n+1] = c[n]+0.08*c[n]*dt+0.3*c[n]*np.sqrt(dt)*st.norm.rvs(0,1)
        p[n+1] = p[n]+0.08*p[n]*dt+0.8*p[n]*np.sqrt(dt)*st.norm.rvs(0,1)
        x[n+1] = x[n]+(b[n]-x[n])*dt+sig*np.sqrt(dt)*st.norm.rvs(0,1)
    
    
    if display:
        
        if trajectory is None:
            # Compute optimal control policy
            d = np.array([diesel(x[k], c[k], p[k], vers, dmax) for k in range(len(t_grid))])
            g = x - d
        
        if trajectory is not None:
            # Trajectory and optimal control policy is already given
            g = trajectory - diesel_policy
            x = trajectory
        
        fig, ax = plt.subplots()
        
        # Plotting Process overview
        ax.plot(t_grid, b, label="mean process")
        ax.plot(t_grid, x, label="residual demand")
        ax.plot(t_grid, c, label="grid cost")
        ax.plot(t_grid, p, label="diesel cost")
        ax.set(title=f"Process overview | sig={sig}", xlabel="Time", ylabel="Energy/Value")
        plt.legend(loc="lower left")
        if save: plt.savefig(f"Easy_Microgrid/ValueFunction_Results/Process_overview_sig={sig}.png")
        plt.show()
        
        
        fig, ax = plt.subplots()
                
        # Plotting the power regulation in a specific scenario (trajectory of x)
        ax.plot(t_grid, b, label="mean process")
        ax.plot(t_grid, x, label="residual demand")
        ax.plot(t_grid, g, label="grid power")
        ax.plot(t_grid, d, label="diesel power")
        ax.set(title=f"Power regulation | {vers} | sig={sig}", xlabel="Time", ylabel="Energy")
        plt.legend(loc="lower left")
        if save: plt.savefig("Easy_Microgrid/ValueFunction_Results/Power_regulation_type_"+vers+f"_sig={sig}.png")
        plt.show()
        
    

    return b, c, p
    
    
    
    