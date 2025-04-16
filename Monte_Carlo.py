#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 10:41:10 2023

@author: noahscheider
"""


import numpy as np
import scipy.stats as st
import scipy.stats as spa
from Verification_Functions import *
from HJB_Functions import *




def SDE(N_tilde, shift, x, b, dt, sig):
    """
    
    Parameters
    ----------
    N : integer
        Length of SDE.
    shift : integer
        Time-shift for correct SDE.
    x : float64
        Initial Value of SDE.
    b : Array of float64 shape(len(t_grid), 1)
        mean-reverting process with length of the time grid t_grid.
    dt : float64
        time discretization.
    GBM_model : Boolean, optional
        Whether price and cost process of diesel and public energy are modelled according to a GBM.

    Returns
    -------
    SDE of length N_tilde (price and cost process in case GBM_model=True).

    """
    
    x_traj = np.zeros((N_tilde,len(x)))
    x_traj[0,:] = x
    
    for k in range(N_tilde-1):
        x_traj[k+1,:] = x_traj[k,:] + (b[k+shift]-x_traj[k,:])*dt + sig*np.sqrt(dt)*st.norm.rvs(0, 1, size=len(x))
            
    return x_traj


# Monte Carlo for the Value function
def MonteCarloEasy(t_grid, x_grid, b, c, p, dt, M, vers, sig, d_grid, seed=False):
    """

    Parameters
    ----------
    t_grid : Array of float64
        discretized time grid.
    x_grid : Array of float64
        discretized space grid.
    b : Array of float64, shape(len(t_grid), 1)
        forecast for the residual demand.
    c : Array of float64, shape(len(t_grid), 1)
        price process for the grid power.
    p : Array of float64, shape(len(t_grid), 1)
        price process for diesel.
    dt : float64
        time discretization.
    M : integer
        number of Monte Carlo simulations for value function, e.g. number of samples trajectories in each step.
    vers : string
        type of forcing term.
    sig : float64
        diffusion constant for SDE.
    d_grid : Array of float64
        discretized control grid
    seed : Boolean, optional
        random seed. The default is False.


    Returns
    -------
    Array of float64, shape(len(t_grid), len(x_grid))
        value function on time and space grid via Monte Carlo Method.

    """
    
    if seed: np.random.seed(seed=200823)

    # Initiate empty value function for values and standard deviation of different MC approximations
    v = spa.lil_matrix((len(t_grid), len(x_grid))) # Terminal Condition implies v(T,x)=0
        
    # Initiate status in order to monitor progress of algorithm
    status = np.zeros(2)
    
    # Iterate over time (n, t) and space (i, x)
    for n, t in enumerate(t_grid[:-1]): # LRAM
            
        # Initiating Monte Carlo sum
        mc_sum = 0
                    
        # Generate M samples of state trajectory
        for m in range(M):
            
            x_traj = SDE(len(t_grid)-n, n, x_grid, b, dt, sig)
                            
            # Monte Carlo for gain function
            mc_sum += np.sum(np.array([ForcingTerm(t, b[k], c[k-n], p[k-n], x_traj[k-n, :], sig, t_grid[-1], vers, d_grid)*dt for k in range(n, len(t_grid[:-1]))]), axis=0) # LRAM
        
        # Updating value function 
        v[n, :] = mc_sum/M
        
        print(t)
        
        # Status of the current Monte Carlo Algorithm
        progress = n/len(t_grid)
        if progress>0.33 and status[0]==0: 
            status[0]=1
            print(f"-- MC {progress*100:.0f}% done --")
        if progress>0.66 and status[1]==0: 
            status[1]=1
            print(f"-- MC {progress*100:.0f}% done --")
        
    # Converting sparse matrix back into array
    return v.toarray()




# Function to quantify convergence of MC with respect to Monte Carlo trajectories
def ConvergenceMC_Sim(dt_grid, x_grid, M_grid, T, vers, intra, K, sig, d_grid, seed=False):
    """

    Parameters
    ----------
    dt_grid : Array of float64
        containing different time discretization sizes.
    x_grid : Array of float64
        discretized space grid.
    M_grid : Array of integer
        number of Monte Carlo simulations for value function, e.g. number of samples trajectories in each step.
    T : float64
        Terminal time.
    vers : string
        type of forcing term.
    intra : boolean
        comparision to the true solution (False) if available or to the best approximation available (True).
    K : integer
        number of error approximation in order to monitor impact of sig
    sig : float64
        diffusion constant for SDE.
    d_grid : Array of float64
        discretized control grid

    Returns
    -------
    error_list: list of Arrays of float64, shape(len(dt_grid), len(h_grid))
        Error of the approximation with dt fix and varying amount of Monte Carlo simulations.

    """
    
    if seed: np.random.seed(9071)
    
    error_list = [np.copy(None)]*K
    
    # Parameters for the most accurate approximation, indifferent w.r.t h
    dt = dt_grid[0]
    t0_grid = np.arange(0, T+dt, dt)
    b0, c0, p0 = InitiateProcesses(t0_grid, sig)
    
    # Get most accurate approximation
    if intra:
        v_best = MonteCarloEasy(t0_grid, x_grid, b0, c0, p0, dt, M_grid[-1], vers, sig, d_grid, seed=False)
    else:
        v_best = TrueValueFunction(t0_grid, x_grid, vers)

    
    for k in range(K):
        
        # Initiate empty array to store error
        error = np.zeros((len(dt_grid), len(M_grid)))    
        
        for dt_index, dt in enumerate(dt_grid):
            
            # Refine t_grid and x_grid for every discretization size
            t_grid = np.arange(0, T+dt, dt)
            slice_time = int(dt/dt_grid[0])
            b, c, p = b0[::slice_time], c0[::slice_time], p0[::slice_time]
            
            for m_index, M in enumerate(M_grid[:-1]):
                
                # Compute numerical approximation by Monte Carlo for comparision
                v_MC = MonteCarloEasy(t_grid, x_grid, b, c, p, dt, M, vers, sig, d_grid, seed=False)
                
                # Compare with previously computed numerical approximation when increasing Monte Carlo simulations            
                error[dt_index, m_index] = np.linalg.norm(v_MC-v_best[::slice_time,], ord="fro")/np.sqrt(len(t_grid)*len(x_grid))
                
        error_list[k] = error
        print("K", k)
        
    # Mean and Std for a confidence intervall of the error for different runs
    # mean_error = np.array(sum(error_list))/K
    # std_error = np.array(np.sqrt(sum([(M-mean_error)**2 for M in error_list])/(K-1)))

    return error_list




def PlotConvergenceMC_Sim(dt_grid, M_grid, error_list, vers, intra, K, sig, h, save=False):
    """
    
    Parameters
    ----------
    dt_grid : Array of float64
        containing different time discretization sizes.
    M_grid : Array of integer
        number of Monte Carlo simulations for value function, e.g. number of samples trajectories in each step.
    error_list: list of Arrays of float64, shape(len(dt_grid), len(h_grid))
        Error of the approximation on variety of different Monte Carlo trajectories.
    vers : string
        type of forcing term.
    intra : boolean
        comparision to the true solution (False) if available or to the best approximation available (True).
    K : integer
        number of error approximation in order to monitor impact of sig
    sig : float64
        diffusion constant for SDE.
    h : float64, optional
        fixed space discretization. The default is 0.2.
    save : boolean, optional
        saves the plot (True) or not (False). The default is False.

    Returns
    -------
    error plots against different amounts of Monte Carlo simulations.

    """
    
    fig, ax = plt.subplots()
    
    # Mean error 
    error = np.array(sum(error_list))/K
    
    # Plot reference rate to compare convergence with higher number of simulations
    ax.plot(M_grid[:-1], error[0, 0]*3.5/np.sqrt(M_grid[:-1]), label=f"Sqrt reference", color="black", lw=1)
    
    # Loop over different time discretization sizes
    for dt_index, dt in enumerate(dt_grid):
        ax.plot(M_grid[:-1], error[dt_index, :-1], color="C{}".format(dt_index), label=f"h = {h}, dt = {dt}", marker = 'o', markersize='8', markeredgewidth='2', markeredgecolor='white')
        for k in range(K):
            ax.scatter(M_grid[:-1], error_list[k][dt_index, :-1], color="C{}".format(dt_index), marker="_")
        
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.legend(loc="lower left")

    ax.set(title=f"Monte Carlo | {vers} | intra={intra} | sig={sig}", xlabel=f"# MC Simulations", ylabel="Error")
    fig.savefig(f"Easy_Microgrid/Convergence_Plots/Conv_MC_{vers}_intra={intra}_sig={sig}.png")
   
    plt.show()
    
    
    
    
# Function to quantify convergence of MC with respect to time
def ConvergenceMC_Time(dt_grid, x_grid, M_grid, T, vers, intra, K, sig, d_grid, seed=False):
    """

    Parameters
    ----------
    dt_grid : Array of float64
        containing different time discretization sizes.
    x_grid : Array of float64
        discretized space grid.
    M_grid : Array of integer
        number of Monte Carlo simulations for value function, e.g. number of samples trajectories in each step.
    T : float64
        Terminal time.
    vers : string
        type of forcing term.
    intra : boolean
        comparision to the true solution (False) if available or to the best approximation available (True).
    K : integer
        number of error approximation in order to monitor impact of sig
    sig : float64
        diffusion constant for SDE.
    d_grid : Array of float64
        discretized control grid

    Returns
    -------
    error: Array of float64, shape(len(dt_grid), len(h_grid))
        Error of the approximation with amount of Monte Carlo simulations fix and varying dt.

    """
    
    if seed: np.random.seed(9071)
        
    # Parameters for the most accurate approximation, indifferent w.r.t h
    dt = dt_grid[0]
    t0_grid = np.arange(0, T+dt, dt)
    b0, c0, p0 = InitiateProcesses(t0_grid, sig)
    
    # Get most accurate approximation
    if intra:
        v_best = MonteCarloEasy(t0_grid, x_grid, b0, c0, p0, dt, M_grid[-1], vers, sig, d_grid)
    else:
        v_best = TrueValueFunction(t0_grid, x_grid, vers)

            
    # Initiate empty array to store error
    error = np.zeros((len(dt_grid), len(M_grid)))
    
    for M_index, M in enumerate(M_grid):

        for dt_index, dt in enumerate(dt_grid[1:]):
            
            # Refine t_grid and x_grid for every discretization size
            t_grid = np.arange(0, T+dt, dt)
            slice_time = int(dt/dt_grid[0])
            b, c, p = b0[::slice_time], c0[::slice_time], p0[::slice_time]
            
            # Compute numerical approximation by Monte Carlo for comparision
            v_MC = MonteCarloEasy(t_grid, x_grid, b, c, p, dt, M, vers, sig, d_grid)
            
            # Compare with previously computed numerical approximation when increasing Monte Carlo simulations            
            error[dt_index+1, M_index] = np.linalg.norm(v_MC-v_best[::slice_time,], ord="fro")/np.sqrt(len(t_grid)*len(x_grid))
            # +1 for correct storing
                    
    return error



def PlotConvergenceMC_Time(dt_grid, M_grid, error, vers, intra, K, sig, h=0.1, save=False):
    """
    
    Parameters
    ----------
    dt_grid : Array of float64
        containing different time discretization sizes.
    M_grid : Array of integer
        number of Monte Carlo simulations for value function, e.g. number of samples trajectories in each step.
    error: Array of float64, shape(len(dt_grid), len(h_grid))
        error of the approximation on variety of different Monte Carlo trajectories.
    vers : string
        type of forcing term.
    intra : boolean
        comparision to the true solution (False) if available or to the best approximation available (True).
    K : integer
        number of error approximation in order to monitor impact of sig
    sig : float64
        diffusion constant for SDE.
    h : float64, optional
        fixed space discretization. The default is 0.2.
    save : boolean, optional
        saves the plot (True) or not (False). The default is False.

    Returns
    -------
    error plots against different amounts of Monte Carlo simulations.

    """
    
    fig, ax = plt.subplots()
    
    # Plot reference rate to compare convergence with higher number of simulations
    #ax.plot(dt_grid[1:], 6*dt_grid[1:]**(1.4), label=f"^1.4 reference", color="black", lw=1)
    ax.plot(dt_grid[1:], 3*dt_grid[1:], label=f"Linear reference", color="black", lw=1)


    
    # Loop over different amount of Monte Carlo simulations
    for M_index, M in enumerate(M_grid):
        ax.plot(dt_grid[1:], error[1:, M_index], color="C{}".format(M_index), label=f"h = {h}, M = {M}", marker = 'o', markersize='8', markeredgewidth='2', markeredgecolor='white')
        
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.legend(loc="lower right")

    ax.set(title=f"Monte Carlo | {vers} | intra={intra} | sig={sig}", xlabel=f"Time discretization dt", ylabel="Error")
    fig.savefig(f"Easy_Microgrid/Convergence_Plots/Conv_MC_{vers}_intra={intra}_sig={sig}_Time.png")
   
    plt.show()

