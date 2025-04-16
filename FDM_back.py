#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 12:12:17 2023

@author: noahscheider
"""

import numpy as np
import scipy.sparse as spa
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator
from Verification_Functions import *
from HJB_Functions import *


# Finite Difference method solving a function implicitly backwards in time
def FDMImplicitBackwards(t_grid, x_grid, xa, xb, Ter, b, c, p, dt, h, vers, sig, d_grid, upwind):
    """

    Parameters
    ----------
    t_grid : Array of float64
        discretized time grid.
    x_grid : Array of float64
        discretized space grid.
    xa : Array of float64
        "leftest" boundary condition in space.
    xb : Array of float64
        "rightest" boundary condition in space.
    Ter : Array of float64
        Terminal boundary condition in time.
    b : Array of float64, shape(len(t_grid), 1)
        forecast for the residual demand.
    c : Array of float64, shape(len(t_grid), 1)
        price process for the grid power.
    p : Array of float64, shape(len(t_grid), 1)
        price process for diesel.
    dt : float64
        time discretization.
    h : float64
        space discretization.
    vers : string
        type of forcing term.
    sig : float64
        diffusion constant for SDE.
    d_grid : Array of float64
        discretized control grid
    upwind : boolean, optional
        Turns upwind scheme on (True) of off (False). The default is False.

    Returns
    -------
    Array of float64, shape(len(t_grid), len(x_grid))
        value function evaluated on time and space grid via implicit FDM.

    """
    
    # Set boundary conditions
    v_fdm_HJB = spa.lil_matrix((len(t_grid), len(x_grid)))
    v_fdm_HJB[:, 0] = xa
    v_fdm_HJB[:, -1] = xb
    v_fdm_HJB[-1, :] = Ter
    
    # Initiate Matrices for Finite Difference Method
    up_diag = np.ones(len(x_grid[1:-2]))
    diag = -2*np.ones(len(x_grid[1:-1]))
    low_diag = np.ones(len(x_grid[2:-1]))
    A = spa.diags(up_diag, offsets=1) + spa.diags(diag, offsets=0) + spa.diags(low_diag, offsets=-1)
    C = spa.diags(up_diag, offsets=1) - spa.diags(low_diag, offsets=-1)
    
    # Initiate status in order to monitor the progress of the algorithm
    status = np.zeros(2)

    # Iteration backwards in time
    for n, t in enumerate(t_grid[1:]):    
        
        # Compute forcing term     
        f = ForcingTerm(t_grid[-n-1], b[-n-1], c[-n-1], p[-n-1], x_grid[1:-1], sig, t_grid[-1], vers, d_grid)
        
        # Continue with backwards approximation of convection term
        if upwind:
            # Create Finite Difference Matrix
            b_arr = b[-n-2]-x_grid[1:-1]
            b_abs = np.abs(b_arr)
            M = spa.eye(len(x_grid[1:-1])) + dt/h**2*(-sig**2/2*spa.eye(len(x_grid[1:-1])) - spa.diags(b_abs)*h/2)@A - dt/(2*h)*spa.diags(b_arr)@C
        
            # Include boundary conditions
            f[0] = f[0] - (-sig**2/2/h**2 - b_abs[0]/(2*h) + b_arr[0]/(2*h)) * v_fdm_HJB[-n-2, 0]
            f[-1] = f[-1] - (-sig**2/2/h**2 - b_abs[-1]/(2*h) - b_arr[-1]/(2*h)) * v_fdm_HJB[-n-2, -1]
        
        
        # Continue with central approximation of convection term
        else:
            # Create Finite Difference Matrix
            eu = spa.diags(b[-n-2]-x_grid[1:-1])
            M = spa.eye(len(x_grid[1:-1])) - dt*sig**2/2/h**2*A - dt/(2*h)*eu@C
                
            # Include boundary conditions
            f[0] = f[0] + (sig**2/2/h**2 - (b[-n-2]-x_grid[1])/(2*h)) * v_fdm_HJB[-n-2, 0]
            f[-1] = f[-1] + (sig**2/2/h**2+ (b[-n-2]-x_grid[-2])/(2*h)) * v_fdm_HJB[-n-2, -1]
            
            
        # Update value function 
        v_fdm_HJB[-n-2, 1:-1] = spa.linalg.spsolve(M.tocsr(), v_fdm_HJB[-n-1, 1:-1].T+dt*f.reshape(len(f), 1))
    
        # Status of the current Monte Carlo Algorithm
        progress = n/len(t_grid)
        if progress>0.33 and status[0] == 0: 
            status[0] = 1
            print(f"-- FDM {progress*100:.0f}% done --")
        if progress>0.66 and status[1] == 0: 
            status[1] = 1
            print(f"-- FDM {progress*100:.0f}% done --")

    return v_fdm_HJB.toarray()




# Coded only for intermediate-hjb case
def FDMExplicitBackwards(t_grid, x_grid, xa, xb, Ter, b, c, p, dt, h, vers, lam, sig, d_grid, diesel_off=False):
    """

    Parameters
    ----------
    t_grid : Array of float64
        discretized time grid.
    x_grid : Array of float64
        discretized space grid.
    xa : Array of float64
        "leftest" boundary condition in space.
    xb : Array of float64
        "rightest" boundary condition in space.
    Ter : Array of float64
        Terminal boundary condition in time.
    b : Array of float64, shape(len(t_grid), 1)
        forecast for the residual demand.
    c : Array of float64, shape(len(t_grid), 1)
        price process for the grid power.
    p : Array of float64, shape(len(t_grid), 1)
        price process for diesel.
    dt : float64
        time discretization.
    h : float64
        space discretization.
    lam : float64
        penalization term for the intermediate hjb example.
    sig : float64
        diffusion constant for SDE.
    d_grid: Array of float64, shape(len(levels), 1)
        discretized control grid 
    diesel_off : boolean, optional
        whether diesel generator is always turned off. Sets d_grid = 0. The default is False.

    Returns
    -------
    Array of float64, shape(len(t_grid), len(x_grid))
        value function evaluated on time and space grid via explicit FDM.

    """    
 

    if diesel_off: d_grid = np.zeros(len(d_grid))

    # Set boundary conditions
    v_fdm_HJB = spa.lil_matrix((len(t_grid), len(x_grid)))
    v_fdm_HJB[:, 0] = xa
    v_fdm_HJB[:, -1] = xb
    v_fdm_HJB[-1, :] = Ter
    
    # Set array for control policy
    d_fdm_HJB = spa.lil_matrix((len(t_grid), len(x_grid)))
    
    # Initiate Components of Finite Difference Matrix
    up_diag = np.ones(len(x_grid[1:-2]))
    diag = -2*np.ones(len(x_grid[1:-1]))
    low_diag = np.ones(len(x_grid[2:-1]))
    A = spa.diags(up_diag, offsets=1) + spa.diags(diag, offsets=0) + spa.diags(low_diag, offsets=-1)
    C = spa.diags(up_diag, offsets=1) - spa.diags(low_diag, offsets=-1)
    
    # Extension for intermediate example
    diagonals = np.ones(len(x_grid)-1) # for offset diagonals
    # as boundary values are given, we don't need correct derivative approximation on those
    Dx = (-spa.diags(diagonals, offsets=1) + spa.diags(diagonals, offsets=-1))/(2*h)
    
    # Initiate status in order to see how much of the algorithm is already done
    status = np.zeros(2)
    
    # Iteration backwards in time
    for n, t in enumerate(t_grid[1:]):
        
        # Create Finite Difference Matrix, central finite difference approximation
        eu = spa.diags(b[-n-1]-x_grid[1:-1])
        M = spa.eye(len(x_grid[1:-1])) + dt*sig**2/(2*h**2)*A + dt/(2*h)*eu@C
        
        # Extension for intermediate example
        dxv = np.zeros(len(x_grid[1:-1]))
        dxv = Dx.dot(v_fdm_HJB[-n-1,:].T)[1:-1] # has len(x_grid)-2
        dxv = dxv.toarray()

        # dxv[:,0] because slicing in order to have matching dimensions in Forcing term
        f, d_fdm_HJB[-n-1, 1:-1] = ForcingTerm(t_grid[-n-1], b[-n-1], c[-n-1], p[-n-1], x_grid[1:-1], sig, t_grid[-1], "intermediate-hjb", d_grid, dxv[:,0], lam)
        
        # Include boundary conditions
        f[0] += (sig**2/(2*h**2)-(b[-n-1]-x_grid[1])/(2*h))*v_fdm_HJB[-n-1, 0]
        f[-1] += (sig**2/(2*h**2)+(b[-n-1]-x_grid[-2])/(2*h))*v_fdm_HJB[-n-1, -1]
        
        # Update value function
        v_fdm_HJB[-n-2, 1:-1] = M.dot(v_fdm_HJB[-n-1, 1:-1].T + dt*f.reshape(len(f), 1))
    
        # Status of the current Monte Carlo Algorithm
        progress = n/len(t_grid)
        if progress>0.33 and status[0]==0: 
            status[0]=1
            print(f"-- FDM {progress*100:.0f}% done --")
        if progress>0.66 and status[1]==0: 
            status[1]=1
            print(f"-- FDM {progress*100:.0f}% done --")

    
    return v_fdm_HJB.toarray(), d_fdm_HJB.toarray()




# Function to quantify convergence of implicit FDM in space to the true function
def ConvergenceFDMImplicitSpace(dt_grid, h_grid, T, prodmax, consmax, vers, sig, d_grid, intra, upwind):
    """

    Parameters
    ----------
    dt_grid : Array of float64
        containing different time discretization sizes.
    h_grid : Array of float64 
        containing different space discretization sizes.
    T : float64
        final time.
    prodmax : float64
        maximum production of the microgrid.
    consmax : float64
        maximum consumption of the microgrid.
    vers : string
        type of forcing term.
    sig : float64
        diffusion constant for SDE.
    d_grid : Array of float64
        discretized control grid
    intra : boolean
        comparision to the true solution (False) if available or to the best approximation available (True).
    upwind : boolean
        Turns upwind scheme on (True) of off (False).

    Returns
    -------
    error : Array of float64, shape(len(dt_grid), len(h_grid))
        Error of the approximation on variety of discretization sizes.

    """
        
    # Check if we are in HJB function
    hjb_Bool = vers[:4]=="hjb-"
    
    # Compute dt_grid according to h_grid in order to guarantee stability
    error = np.zeros((len(dt_grid), len(h_grid)))
    
    # finest space and time discretization grid
    h = h_grid[0]
    dt = dt_grid[0]
    x0_grid = np.arange(-prodmax, consmax+h, h)
    t0_grid = np.arange(0, T+dt, dt)
    
    # Initiate finest (deterministic) process
    b0, c0, p0 = InitiateProcesses(t0_grid, sig, seed=True)
    
    # In case of a version of HJB, we set the boundary values of a bigger domain to zero
    if hjb_Bool:
        v_best = FDMImplicitBackwards(t0_grid, x0_grid, np.zeros(len(t0_grid)),  np.zeros(len(t0_grid)), np.zeros(len(x0_grid)), b0, c0, p0, dt, h, vers, sig, d_grid, upwind=upwind)
    
    # Otherwise we compute True value function (especially because we need the boundary values)
    else:
        v_best = TrueValueFunction(t0_grid, x0_grid, vers)
    
    # In case we want to do an intra comparision we can overwrite v_best for later simplicity as now the
    # boundary values are fully passed on, which would be the only information we need from the True Value function
    if intra and not hjb_Bool:
        v_best = FDMImplicitBackwards(t0_grid, x0_grid, v_best[:, 0], v_best[:, -1], v_best[-1, :], b0, c0, p0, dt, h, vers, sig, d_grid, upwind)

    
    for dt_index, dt in enumerate(dt_grid):
        
        # Impair t_grid for every discretization size and compute slice_time, time discretization ratio for slicing comparision function to match dimensions
        t_grid = np.arange(0, T+dt, dt)
        slice_time = int(dt/dt_grid[0])
        b, c, p = b0[::slice_time], c0[::slice_time], p0[::slice_time]
    
        for h_index, h in enumerate(h_grid[1:]):
            
            # For correct storing
            h_index = h_index+1
            
            # Impair x_grid for every discretization size and compute slice_space, space discretization ratio for slicing comparision function to match dimensuons
            x_grid = np.arange(-prodmax, consmax+h, h)
            slice_space = int(h/h_grid[0])
            
            # In case of a version of HJB, we half the domain, e.g. a quarter on left and quarter on right
            if hjb_Bool:                   
                # Compute FDM for dt and h grid
                v_FDM_im = FDMImplicitBackwards(t_grid, x_grid, np.zeros(len(t_grid)),  np.zeros(len(t_grid)), np.zeros(len(x_grid)), b, c, p, dt, h, vers, sig, d_grid, upwind)
                trunc_left = int(prodmax/2/h)
                trunc_right = int((consmax/2+prodmax)/h)+1
                    
                # L2-norm
                error[dt_index, h_index] = np.linalg.norm(v_FDM_im[:, trunc_left:trunc_right]-v_best[::slice_time, slice_space*trunc_left:slice_space*trunc_right:slice_space], "fro")/np.sqrt(len(t_grid)*len(x_grid))
            
            # Only need to differ once as comparision function is anyway v_best 
            else:
                # In case of intra true and non hjb-function we can slice to get exact boundary value that we passed on earlier 
                v_FDM_im = FDMImplicitBackwards(t_grid, x_grid, v_best[::slice_time, 0], v_best[::slice_time, -1], v_best[-1, ::slice_space], b, c, p, dt, h, vers, sig, d_grid, upwind)
                
                # L2-norm
                error[dt_index, h_index] = np.linalg.norm(v_FDM_im-v_best[::slice_time, ::slice_space], "fro")/np.sqrt(len(t_grid)*len(x_grid))

    return error



# Plot Convergence error of Implicit scheme above
def PlotConvergenceFDMImplicitSpace(dt_grid, h_grid, error, sig, vers, intra, upwind, save=False):
    """

    Parameters
    ----------
    dt_grid : Array of float64
        containing different time discretization sizes.
    h_grid : Array of float64
        containing different space discretization sizes.
    error : Array of float64, shape(len(dt_grid), len(h_grid))
        Error of the approximation on variety of discretization sizes.
    sig : float64
        diffusion constant for SDE.
    vers : string
        type of forcing term.
    intra : boolean
            comparision to the true solution (False) if available or to the best approximation available (True).
    upwind : boolean
        Turns upwind scheme on (True) of off (False).
    save : boolean, optional
        saves the plot (True) or not. The default is False.

    Returns
    -------
    error plots against different space discretization sizes.

    """    
    
    fig, ax = plt.subplots()
        
    if upwind: 
        ax.plot(h_grid[1:], 0.8*error[0, -1]*h_grid[1:]/h_grid[-1], label=f"Linear reference", color="black", lw=1)
    else:
        ax.plot(h_grid[1:], 0.8*error[0, -1]*(h_grid[1:]/h_grid[-1])**2, label=f"Quadratic reference", color="black", lw=1)
    
    # Loop over different time discretization sizes
    for dt_index, dt in enumerate(dt_grid):
        ax.plot(h_grid[1:], error[dt_index, 1:], color="C{}".format(dt_index), label=f"dt = {dt}", marker = 'o', markersize='8', markeredgewidth='2', markeredgecolor='white')        
        
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)

    ax.set(title=f"Implicit FDM | {vers} | upwind={upwind} | intra={intra} | sig={sig}", xlabel=f"Space discretization h", ylabel="Error")
    ax.legend(loc="lower right")
    
    if save: fig.savefig(f"Easy_Microgrid/Convergence_Plots/Conv_FDM_{vers}_intra={intra}_upwind={upwind}_sig={sig}.png")
    
    plt.show()
    
    
    
    
def ConvergenceFDMImplicitTime(dt_grid, h_grid, T, prodmax, consmax, vers, sig, d_grid, intra, upwind):
    """

    Parameters
    ----------
    dt_grid : Array of float64
        containing different time discretization sizes.
    h_grid : Array of float64 
        containing different space discretization sizes.
    T : float64
        final time.
    prodmax : float64
        maximum production of the microgrid.
    consmax : float64
        maximum consumption of the microgrid.
    vers : string
        type of forcing term.
    sig : float64
        diffusion constant for SDE.
    d_grid : Array of float64
        discretized control grid
    intra : boolean
        comparision to the true solution (False) if available or to the best approximation available (True).
    upwind : boolean
        Turns upwind scheme on (True) of off (False).

    Returns
    -------
    error : Array of float64, shape(len(dt_grid), len(h_grid))
        Error of the approximation on variety of discretization sizes.

    """    
    # Check if we are in HJB function
    hjb_Bool = vers[:4]=="hjb-"
    
    # Compute dt_grid according to h_grid in order to guarantee stability
    error = np.zeros((len(dt_grid), len(h_grid)))
    
    # finest space and time discretization grid
    h = h_grid[0]
    dt = dt_grid[0]
    x0_grid = np.arange(-prodmax, consmax+h, h)
    t0_grid = np.arange(0, T+dt, dt)
    
    # Initiate finest (deterministic) process
    b0, c0, p0 = InitiateProcesses(t0_grid, sig, seed=True)
    
    # In case of a version of HJB, we set the boundary values of a bigger domain to zero
    if hjb_Bool:
        v_best = FDMImplicitBackwards(t0_grid, x0_grid, np.zeros(len(t0_grid)),  np.zeros(len(t0_grid)), np.zeros(len(x0_grid)), b0, c0, p0, dt, h, vers, sig, d_grid, upwind)
    
    # Otherwise we compute True value function (especially because we need the boundary values)
    else:
        v_best = TrueValueFunction(t0_grid, x0_grid, vers)
    
    # In case we want to do an intra comparision we can overwrite v_best for later simplicity as now the
    # boundary values are fully passed on, which would be the only information we need from the True Value function
    if intra and not hjb_Bool:
        v_best = FDMImplicitBackwards(t0_grid, x0_grid, v_best[:, 0], v_best[:, -1], v_best[-1, :], b0, c0, p0, dt, h, vers, sig, d_grid, upwind)


    for h_index, h in enumerate(h_grid):
                
        # Impair x_grid for every discretization size and compute slice_space, space discretization ratio for slicing comparision function to match dimesnions
        x_grid = np.arange(-prodmax, consmax+h, h)
        slice_space = int(h/h_grid[0])

    
        for dt_index, dt in enumerate(dt_grid[1:]):
                        
            # Impair t_grid for every discretization size and compute slice_time, time discretization ratio for slicing comparision function to match dimesnions
            t_grid = np.arange(0, T+dt, dt)
            slice_time = int(dt/dt_grid[0])
            b, c, p = b0[::slice_time], c0[::slice_time], p0[::slice_time]
                
            # In case of a version of HJB, we half the domain, e.g. a quarter on left and quarter on right
            if hjb_Bool:                   
                # Compute FDM for dt and h grid
                v_FDM_im = FDMImplicitBackwards(t_grid, x_grid, np.zeros(len(t_grid)),  np.zeros(len(t_grid)), np.zeros(len(x_grid)), b, c, p, dt, h, vers, sig, d_grid, upwind)
                trunc_left = int(prodmax/2/h)
                trunc_right = int((consmax/2+prodmax)/h)+1
                    
                # L2-norm
                error[dt_index+1, h_index] = np.linalg.norm(v_FDM_im[:, trunc_left:trunc_right]-v_best[::slice_time, slice_space*trunc_left:slice_space*trunc_right:slice_space], "fro")/np.sqrt(len(t_grid)*len(x_grid))
                # +1 for correct storing
            
            # Only need to differ once as comparision function is anyway v_best 
            else:
                # In case of intra true and non hjb-function we can slice of exact boundary value that we passed on earlier 
                v_FDM_im = FDMImplicitBackwards(t_grid, x_grid, v_best[::slice_time, 0], v_best[::slice_time, -1], v_best[-1, ::slice_space], b, c, p, dt, h, vers, sig, d_grid, upwind)
                
                # L2-norm
                error[dt_index+1, h_index] = np.linalg.norm(v_FDM_im-v_best[::slice_time, ::slice_space], "fro")/np.sqrt(len(t_grid)*len(x_grid))
                # +1 for correct storing
                
    return error



# Plot Convergence error of Implicit scheme above
def PlotConvergenceFDMImplicitTime(dt_grid, h_grid, error, sig, vers, intra, upwind, save=False):
    """

    Parameters
    ----------
    dt_grid : Array of float64
        containing different time discretization sizes.
    h_grid : Array of float64
        containing different space discretization sizes.
    error : Array of float64, shape(len(dt_grid), len(h_grid))
        error of the approximation on variety of discretization sizes.
    sig : float64
        diffusion constant for SDE.
    vers : string
        type of forcing term.
    intra : boolean
            comparision to the true solution (False) if available or to the best approximation available (True).
    upwind : boolean
        Turns upwind scheme on (True) of off (False).
    save : boolean, optional
        saves the plot (True) or not. The default is False.

    Returns
    -------
    error plots against different time discretization sizes.

    """    
    
    fig, ax = plt.subplots()
        
    ax.plot(dt_grid[1:], 0.6*error[-1, 0]*dt_grid[1:]/h_grid[-1], label=f"Linear reference", color="black", lw=1)
    
    # Loop over different time discretization sizes
    for h_index, h in enumerate(h_grid):
        ax.plot(dt_grid[1:], error[1:, h_index], color="C{}".format(h_index), label=f"h = {h}", marker = 'o', markersize='8', markeredgewidth='2', markeredgecolor='white')

        
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)

    ax.set(title=f"Implicit FDM | {vers} | upwind={upwind} | intra={intra} | sig={sig}", xlabel=f"Time discretization dt", ylabel="Error")
    ax.legend(loc="lower right")
    if save: fig.savefig(f"Easy_Microgrid/Convergence_Plots/Conv_FDM_{vers}_intra={intra}_upwind={upwind}_sig={sig}_Time.png")
    plt.show()

