#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:52:46 2023

@author: noahscheider
"""

import numpy as np
import scipy.stats as st
import scipy.sparse as spa
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import sys

from Verification_Functions import *
from HJB_Functions import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.interpolate import CubicSpline



def IntermediateSDEfix(N, Mp, x0, d_controlled, b, dt, lam, sig, shift=0):
    """
    
    Parameters
    ----------
    N : integer
        length of SDE.
    Mp : integer
        amount of SDE samples in total (M*amount of diferent initial points)
    x0 : float64
        initial Value of SDE.
    d_controlled : Array of float64, shape(Mp, len(t_grid))
        exerted control
    b : Array of float64, shape(len(t_grid), 1)
        mean-reverting process.
    dt : float64
        time discretization.
    lam : float64 >=0
        penalization term for excessive energy consumption on consumer.
    shift : integer, optional
        time-shift for correct SDE.

    Returns
    -------
    SDE, Array of float shape(Mp, N) - (price and cost process for on whole t_grid in case GBM_model=True).

    """
    
    # Initialize SDE 
    SDE = np.zeros((Mp, N))
    SDE[:, 0] = x0
    
    # Construct SDE as in Intermediate Example
    for k in range(N-1):
        SDE[:, k+1] = SDE[:, k] + (b[k+shift]-SDE[:, k]-np.log((1+np.exp(lam*(SDE[:, k]-1-d_controlled[:, k+shift])))/2))*dt + sig*np.sqrt(dt)*st.norm.rvs(loc=0, scale=1, size=Mp)
    
    return SDE

    



# Monte Carlo for the Value function
def RegressionMC_improved(t_grid, x_grid, b, c, p, dt, M, lam, Q, L, d_grid, sig, vers=""):
    """

    Parameters
    ----------
    t_grid : Array of float64, shape(len(t_grid), 1)
        time grid.
    x_grid : Array of float64, shape(len(x_grid), 1)
        space grid.
    b : Array of float64, shape(len(t_grid), 1)
        deterministic energy forecast for consumer.
    c : Array of float64, shape(len(t_grid), 1)
        cost process of public grid.
    p : Array of float64, shape(len(t_grid), 1)
        price process of diesel.
    dt : float64
        time discretization size.
    M : integer
        Amount of simulated trajectories.
    lam : float64 >=0
        penalization term for excessive energy consumption on consumer.
    Q : integer
        amount of basis function (Q+1 in case of degree Q for polynomial).
    L : integer
        Levels of resampling optimal trajectories.
    d_grid : Array of float64, shape(levels, 1)
        discretized control grid.
    sig : float64
        Volatility of underlying SDE.
    zero : boolean
        whether known "zero" function is approximated

    Returns
    -------
    v_final_lin : Array of float64, shape(len(t_grid), len(x_grid))
        value function interpolated linearly in x_grid.
    d_final_lin : Array of float64, shape(len(t_grid), len(x_grid))
        optimal control policy interpolated linearly in x_grid.
    v_final_cub : Array of float64, shape(len(t_grid), len(x_grid))
        value function interpolated with cubic splines in x_grid.
    d_final_cub : Array of float64, shape(len(t_grid), len(x_grid))
        optimal control policy interpolated with cubic splines in x_grid.

    """
        
    # Initialize polynomial regression model with degree Q and accordingly Q+1 basis fct for later optimised sampling
    poly = PolynomialFeatures(degree=Q, include_bias=True)
    
    # Initiate empty value function, control for values for all the trajectories starting in different points of x1_grid
    x1_grid = x_grid#[::10] #np.array([0]) # whole grid, or sampling according to a specific distribution    
    Mp = M*len(x1_grid)
    v = np.zeros((Mp, len(t_grid))) # Terminal Condition v(T,x)=0
    d = np.zeros((Mp, len(t_grid))) # Last column untouched as, no control in Terminal time needed

    # Start with naive control strategy 0 that improves with higher l
    d_controlled_l = np.zeros((Mp, len(t_grid)))


    # Initiate storage for regression parameters that contains all storage levels at each time step
    regress_models = [np.copy(None) for _ in range(len(t_grid))] # No regression model at final time T
    
    
    
    # Code to get error for quadratic approximation with various parameters
    # v_final_list = [np.copy(None) for _ in range(L)]
    # d_final_list = [np.copy(None) for _ in range(L)]
    
    
    """ Initial sampling """
        
    # Sample M paths with fixed control starting from initial point (3rd argument), shape(Mp, len(t_grid))
    x_traj = IntermediateSDEfix(len(t_grid), Mp, np.repeat(x1_grid, M), d_controlled_l, b, dt, lam, sig)


    """ Regression Monte Carlo """

    # Looping until optimisation level D of resampling SDE with improved control policy is met
    l0=0
    while l0<L:
        
        for n, t in enumerate(t_grid[:-1][::-1]):
                        
            # Initialize storage for continuation values
            cont_value_n = np.zeros((len(d_grid), Mp))
            
            # Initialize storage for later regression information, i.e. parameters for later reconstruction of optimal paths
            regress_models[-n-2] = [np.copy(None) for _ in range(len(d_grid))]
                
            
            """ Polynomial Regression on grid discretisation """
        
            # Needed for Grid-Optimisation, needs to be changed for more sophisticated optimization approach
            for d_index, d_value in enumerate(d_grid):
                        
                # New version with softplus
                hood_ratio_n = (st.norm.pdf(x_traj[:, -n-1], loc=x_traj[:, -n-2] + (b[-n-2]-x_traj[:, -n-2] - np.log((1+np.exp(lam*(x_traj[:, -n-2]-1-d_value)))/2))*dt, scale=sig*np.sqrt(dt))
                                     /st.norm.pdf(x_traj[:, -n-1], loc=x_traj[:, -n-2] + (b[-n-2]-x_traj[:, -n-2] - np.log((1+np.exp(lam*(x_traj[:, -n-2]-1-d_controlled_l[:, -n-2])))/2))*dt, scale=sig*np.sqrt(dt)))
                
                # Initialize polynomial regression model with degree Q and accordingly Q+1 basis fct
                poly = PolynomialFeatures(degree=Q, include_bias=True) # fits without intercept
                poly_reg_model = LinearRegression()
                    
                # Generate polynomial features of x_traj[:, -n-2] up to the Qth degree (stored in a new feature matrix)
                poly_features = poly.fit_transform(x_traj[:, -n-2].reshape(-1, 1)) # Reshape for correct dimension
                
                # Intertwining likelihood-ratio into dependent and independent variable, componentwise array multiplication
                v_tilde = np.sqrt(hood_ratio_n)*v[:, -n-1]
                phi_tilde = np.sqrt(hood_ratio_n)*poly_features.T
                
                # Perform regression
                poly_reg_model.fit(phi_tilde.T, v_tilde)
                
                # Compute continuation value based on prediction of the independent variables
                cont_value_n[d_index, :] = poly_reg_model.predict(poly_features)
            
                # Store model, i.e. parameters for later reconstruction of optimal paths starting
                regress_models[-n-2][d_index] = poly_reg_model
            
            
            """ Optimal control grid discretization """

            # Simple grid optimisation by evaluating on different potential controls and storing argsup
            if vers != "":
                d_grid_evaluated = ForcingTerm(t, b[-n-2], c[-n-2], p[-n-2], x_traj[:, -n-2], sig, t_grid[-1], vers, d_grid)*dt + cont_value_n
            else:
                d_grid_evaluated = np.array([gain_function(x_traj[:, -n-2], c[-n-2], p[-n-2], d_value, "hjb-quad")*dt for d_value in d_grid]) + cont_value_n # extracting matrix results in faster computations
            
            magic = np.argmax(d_grid_evaluated, axis=0) # evaluating index of supremum, only first index considered
            
            # Storing the value function and optimal control at any trajectory point m in time n
            v[:, -n-2] = d_grid_evaluated[magic, np.arange(Mp)]
            d[:, -n-2] = d_grid[magic] # optimal control
            
            
            print(t)
            #PlotContinuationValues(d_grid, regress_models[-n-2], x_grid, c[-n-2], p[-n-2], x_traj[:, -n-1])
        
        PlotTrajectories(Mp, t_grid, x_traj, l0, sig)
        #PlotValueFunctionAlongTrajectories(t_grid, x_traj, v)
        
        
        
        # Code to get error for quadratic approximation with various parameters
        
        # # Initialize storage for the final value function and control policy in t_grid, x_grid for linear and cubic interpolation
        # v_final_old = np.zeros((len(t_grid), len(x_grid))) # terminal condition is 0
        # d_final_old = np.zeros((len(t_grid), len(x_grid))) # control on terminal time 0

        
        # # Old Approach - Only compute v with conditional value and not along resampled d_L trajectories
        # for n, t in enumerate(t_grid[:-1]): # Terminal value given by terminal condition
            
            
        #     # Predicting continuation value for newly gotten point x_traj[i][m, n]
        #     cont_value_n = np.array([regress_models[n][d_index].predict(
        #         poly.fit_transform(x_grid.reshape(-1, 1))) for d_index in range(len(d_grid))])
            
        #     # Simple grid discretization by evaluating on different potential controls and storing argsup  
        #     if vers != "":
        #         d_grid_evaluated = ForcingTerm(t, b[n], c[n], p[n], x_grid, sig, t_grid[-1], vers, d_grid)*dt + cont_value_n # shape(x,) and shape(y,x) leads to row-wise addition
        #     else: 
        #         d_grid_evaluated = np.array([gain_function(x_grid, c[n], p[n], d_value, "hjb-quad")*dt for d_value in d_grid]) + cont_value_n
            
        #     magic = np.argmax(d_grid_evaluated, axis=0) # evaluating supremum in every column, only first maximum considered
            
        #     # Storing optimal control and use it for calculating next point of SDE
        #     d_final_old[n, :] = d_grid[magic]
        #     v_final_old[n, :] = d_grid_evaluated[magic, np.arange(len(x_grid))] # .T for matching dimensions


        # v_final_list[l0] = v_final_old
        # d_final_list[l0] = d_final_old
        
        
        
        
        # Increasing l0 here as if increased in the end, samples of new trajectories don't coincide with evalutation points of value function
        l0 += 1
        if l0==L: break
    
    
        """ Resampling trajectories with improved control """
        
        # resampling M paths with improved control, control at every intial value x_i of trajectory is the same
        d_controlled_l[:, 0] = d[:, 0]
        
        # Compute first two evaluations of SDE
        x_traj[:, 0:2] = IntermediateSDEfix(2, Mp, np.repeat(x1_grid, M), d_controlled_l, b, dt, lam, sig)
        
        # for rest of the SDE compute optimal storage, by using regression parameters stored earlier
        for n, t in enumerate(t_grid[2:]):

            n1 = n+2
            n = n+1
                        
            # Predicting continuation value for newly gotten point x_traj[i*M+m, n]
            cont_value_n = np.array([regress_models[n][d_index].predict(
                            poly.fit_transform(x_traj[:, n].reshape(-1, 1))) for d_index in range(len(d_grid))])
            # every d corresponds to a row with columns i*M:(i+1)*M, for all i in 0:len(x1_grid)
            
            # Simple grid discretization by evaluating on different potential controls and storing argsup  
            if vers != "":
                d_grid_evaluated = ForcingTerm(t, b[-n-2], c[-n-2], p[-n-2], x_traj[:, -n-2], sig, t_grid[-1], vers, d_grid)*dt + cont_value_n
            else:
                d_grid_evaluated = np.array([gain_function(x_traj[:, n], c[n], p[n], d, "hjb-quad")*dt for d in d_grid]) + cont_value_n
            
            max_index = np.argmax(d_grid_evaluated, axis=0) #evaluating supremum, only first maximum considered, columnwise max
            
            # Storing optimal control and use it for calculating next point of SDE
            d_controlled_l[:, n] = d_grid[max_index]
            x_traj[:, n1] =  IntermediateSDEfix(2, Mp, x_traj[:, n], d_controlled_l, b, dt, lam, sig, shift=n)[:, 1] # [:, 1], in order to get next evaluation of SDE
                

        print(f"-- RegMC {l0/L*100:.0f}% done --")
        
        
        
        
    """ Evaluating value function on DoI - Old Approach """
    
    # Initialize storage for the final value function and control policy in t_grid, x_grid for linear and cubic interpolation
    v_final_old = np.zeros((len(t_grid), len(x_grid))) # terminal condition is 0
    d_final_old = np.zeros((len(t_grid), len(x_grid))) # control on terminal time 0

    
    # Old Approach - Only compute v with conditional value and not along resampled d_L trajectories
    for n, t in enumerate(t_grid[:-1]): # Terminal value given by terminal condition
        
        
        # Predicting continuation value for newly gotten point x_traj[i][m, n]
        cont_value_n = np.array([regress_models[n][d_index].predict(
            poly.fit_transform(x_grid.reshape(-1, 1))) for d_index in range(len(d_grid))])
        
        # Simple grid discretization by evaluating on different potential controls and storing argsup  
        if vers != "":
            d_grid_evaluated = ForcingTerm(t, b[n], c[n], p[n], x_grid, sig, t_grid[-1], vers, d_grid)*dt + cont_value_n # shape(x,) and shape(y,x) leads to row-wise addition
        else: 
            d_grid_evaluated = np.array([gain_function(x_grid, c[n], p[n], d_value, "hjb-quad")*dt for d_value in d_grid]) + cont_value_n
        
        magic = np.argmax(d_grid_evaluated, axis=0) # evaluating supremum in every column, only first maximum considered
        
        # Storing optimal control and use it for calculating next point of SDE
        d_final_old[n, :] = d_grid[magic]
        v_final_old[n, :] = d_grid_evaluated[magic, np.arange(len(x_grid))] # .T for matching dimensions

    
    # Code to get error for quadratic approximation with various parameters
    # v_final_list[-1] = v_final_old
    # d_final_list[-1] = d_final_old
    # return v_final_list, d_final_list


    return v_final_old, d_final_old





def RegressionMC_easy(t_grid, x_grid, b, c, p, dt, M, Q, lam, d_max, sig, vers, levels=30):
    """

    Parameters
    ----------
    t_grid : Array of float64, shape(len(t_grid), 1)
        time grid.
    x_grid : Array of float64, shape(len(x_grid), 1)
        space grid.
    b : Array of float64, shape(len(t_grid), 1)
        deterministic energy forecast for consumer.
    c : Array of float64, shape(len(t_grid), 1)
        cost process of public grid.
    p : Array of float64, shape(len(t_grid), 1)
        price process of diesel.
    dt : float64
        time discretization size.
    M : integer
        Amount of simulated trajectories.
    Q : integer
        amount of basis function (Q+1 in case of degree Q for polynomial).
    lam :
        
        
    d_max : float64
        maximum power from diesel generator.
    sig : float64
        Volatility of underlying SDE.
    vers : 
    
    levels : integer, optional
        discretisation level of control space. The default is 20.

    Returns
    -------
    v_final_lin : Array of float64, shape(len(t_grid), len(x_grid))
        value function interpolated linearly in x_grid.
    d_final_lin : Array of float64, shape(len(t_grid), len(x_grid))
        optimal control policy interpolated linearly in x_grid.
    v_final_cub : Array of float64, shape(len(t_grid), len(x_grid))
        value function interpolated with cubic splines in x_grid.
    d_final_cub : Array of float64, shape(len(t_grid), len(x_grid))
        optimal control policy interpolated with cubic splines in x_grid.

    """
            
    # Discretize compact control space for later grid optimisation
    d_grid = np.linspace(0, d_max, num=levels)
    
    # Initialize polynomial regression model with degree Q and accordingly Q+1 basis fct for later optimised sampling
    poly = PolynomialFeatures(degree=Q, include_bias=True)
    
    # Initiate empty value function, control for values for all the trajectories starting in different points of x1_grid
    x1_grid = x_grid#[::10] #np.array([0]) # whole grid, or sampling according to a specific distribution    
    Mp = M*len(x1_grid)
    v = np.zeros((Mp, len(t_grid))) # Terminal Condition v(T,x)=0
    d = np.zeros((Mp, len(t_grid))) # Last column untouched as, no control in Terminal time needed
    x_traj = np.zeros((Mp, len(t_grid)))
    d_controlled_l = np.zeros((Mp, len(t_grid)))

    # Initiate storage for regression parameters that contains all storage levels at each time step
    regress_models = [np.copy(None) for _ in range(len(t_grid))] # Last list item untouched, as no regression model in Terminal time
    # Pen-ultimate regression parameters are zero in accordance to terminal condition
    # Validated by looking at regress_models[-2].coef_
    
    

    """ Initial sampling """
    
    # Sample M paths with fixed control d=0        
    x_traj = IntermediateSDEfix(len(t_grid), Mp, np.repeat(x1_grid, M), d_controlled_l, b, dt, lam, sig)


    """ Regression Monte Carlo """
        
    
    for n, t in enumerate(t_grid[:-1][::-1]):
        
        # Initialize polynomial regression model with degree Q and accordingly Q+1 basis fct
        poly = PolynomialFeatures(degree=Q, include_bias=True) # fits without intercept
        poly_reg_model = LinearRegression()

        # Generate polynomial features of x_n up to the Qth degree (stored in a new feature matrix)
        poly_features = poly.fit_transform(x_traj[:, -n-2].reshape(-1, 1)) # Reshape for later .fit() function
                    
        # Performing regression and computing continuation value based on prediction of the independent variables
        poly_reg_model.fit(poly_features, v[:, -n-1])
        cont_value_n = poly_reg_model.predict(poly_features)
    
        # Store model, i.e. parameters for later reconstruction of optimal paths starting
        regress_models[-n-2] = poly_reg_model
        # +1 in regress_models index to keep dimensionality as last entry corresponds to t_{N-1}

        # Update value function
        v[:, -n-2] = ForcingTerm(t, b[-n-2], c[-n-2], p[-n-2], x_traj[:, -n-2], sig, 2, vers, d_grid)*dt + cont_value_n
                
        
    PlotTrajectories(Mp, t_grid, x_traj, 0, sig)    
    #PlotValueFunctionAlongTrajectories(t_grid, x_traj, v)    
    
    
    """ Evaluate correct value function stage """
    
    # Initiate storage for value function evaluated on the grid
    v_final = np.zeros((len(t_grid), len(x_grid)))
    
    for n, t in enumerate(t_grid[:-1]): # Terminal value given by terminal condition
        
        cont_value_n = regress_models[n].predict(poly.fit_transform(x_grid.reshape(-1, 1)))
        v_final[n, :] = ForcingTerm(t, b[n], c[n], p[n], x_grid, sig, t_grid[-1], vers, d_grid)*dt + cont_value_n
                                
    
    return v_final



    # """ Interpolating stage """
    
    # # Initialize storage for the final value function and control policy in t_grid, x_grid for linear and cubic interpolation
    # v_final_lin = np.zeros((len(t_grid), len(x_grid)))
    # v_final_cub = np.zeros((len(t_grid), len(x_grid)))
    
    # for n, t in enumerate(t_grid):
    #     # Reordering list of v, d, x_traj to use them for interpolation step
    #     # Ordering w.r.t x_traj, hence keep column as t_grid but enlarge rows for all the information given at the node points of trajectories
    #     ind = x_traj[:, n].argsort()
    #     x_traj[:, n] = x_traj[ind, n]
    #     v[:, n] = v[ind, n]
        
    #     # Linear interpolation with constant outside domain
    #     v_final_lin[n,:] = np.interp(x_grid, x_traj[:, n], v[:, n])
        
    #     # Cubic spline interpolation but linear in t=0 as strictly increasing sequence of x can't be guaranteed
    #     # as all trajetories start in the same grid point
    #     if n==0:
    #         v_final_cub[n, :] = v_final_lin[n, :]
    #     else:
    #         v_spl = CubicSpline(x_traj[:, n], v[:, n])
    #         v_final_cub[n, :] = v_spl(x_grid)
        
    # return v_final, v_final_lin, v_final_cub, regress_models



    
# Plots of all continuation values to see the impact of delta
def PlotContinuationValues(d_grid, regress_models_n, x_grid, c_n, p_n, x_traj_n1):
    """
    Parameters
    ----------
    d_grid : Array of float64
        discretized control space.
    regress_models_n : List of Regression variables
        contains Regression parameters of all values in discretized control space.
    x_grid : Array of float64
        discretized space grid.
    c_n : float64
        gird price at time n.
    p_n : float64
        diesel price at time n.
    x_traj_n1 : Array of float64
        residual demands at previous time n-1 that were necessary to compute Regression information.

    Returns
    -------
    Plot of all continuation values.

    """
    # Examine continuation value/value function in every time step and for different delta
    for d_index, d_value in enumerate(d_grid):

        cont_value_n = regress_models_n[d_index].predict(poly.fit_transform(x_grid.reshape(-1, 1)))
            
        value_function_d = gain_function(x_grid, c_n, p_n, d_value, "hjb-quad") * dt + cont_value_n
        plt.plot(x_grid, cont_value_n, label=f"cont d={d_value:.2f}", color="C{}".format(d_index), linestyle="dotted")
        plt.plot(x_grid, value_function_d, color="C{}".format(d_index), label=f"val d={d_value:.2f}")
        # domain on which continuation value at time -n-2 is computed
        plt.axvline(x=max(x_traj_n1), color="black", linewidth=0.5)
        plt.axvline(x=min(x_traj_n1), color="black", linewidth=0.5)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    plt.title(f"time {t}, level {l0}")
    plt.show()



# Plot value function along trajectories for intuition on how domain of interest is covered
def PlotTrajectories(Mp, t_grid, x_traj, l0, sig):
    """
    Parameters
    ----------
    Mp : integer
        overall amount of trajectories.
    t_grid : Array of float64
        discretized time grid.
    x_traj : Array of float64
        samples trajectories.
    l0 : integer
        level of optimised trajectories.
    sig : float64
        diffusion constant.

    Returns
    -------
    Plot of value function along trajectories.

    """
    
    for w in range(Mp):
        plt.plot(t_grid, x_traj[w,:])
    plt.axhline(y=-6, color='black', linestyle='-')
    plt.axhline(y=6, color='black', linestyle='-')
    plt.title(f"level {l0}, sig={sig}")
    plt.show()




def PlotValueFunctionAlongTrajectories(t_grid, x_traj, v):
    """

    Parameters
    ----------
    t_grid : Array of float64, shape(len(t_grid), 1)
        time grid.
    x_traj : Array of float64, shape(len(t_grid), 1)
        trajectories of SDE starting in the same point.
    v : Array of float64, shaoe(M, len(t_grid))
        value function evaluated in x_traj.

    Returns
    -------
    Scatter plot with v plotted along SDEs.

    """
    data = []  # List to store the scatter plot traces
    Mp = x_traj.shape[0]
    
    # Generate traces iteratively
    for m in range(Mp):
        z_values = v[m,:]
    
        trace = go.Scatter3d(
            x=x_traj[m, :],
            y=t_grid,
            z=z_values,
            mode='markers',
            marker=dict(
                size=4,
                color=z_values,
                colorscale='Viridis',
                opacity=0.8
            )
        )
    
        data.append(trace)
    
    layout = go.Layout(
        title='3D Scatter Plot',
        scene=dict(
            xaxis=dict(title='X-axis'),
            yaxis=dict(title='T-axis'),
            zaxis=dict(title='Z-axis')
        )
    )

    fig = go.Figure(data=data, layout=layout)

    fig.show(renderer="browser")



