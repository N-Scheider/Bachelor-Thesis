#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:14:40 2023

@author: noahscheider
"""



import numpy as np
import time
from HJB_Functions import *
from Monte_Carlo import *
from Verification_Functions import *
from FDM_back import *



vers = "quadratic"
FDM_im = False
MC = False




""" General parameters """
T = 2 # final Time
prodmax = 6 # maximum production
consmax = 6 # maximum consumption
h = 0.2 # space discretization
x_grid = np.arange(-prodmax, consmax+h, h)
sig = 4 # deviation strength
d_grid = np.linspace(0, 7, num=20) # discretized control grid



""" Parameters for FDM """
upwind = True # turning upwind scheme on (True) or off (False)
dt_FDM = 0.001 # time discretization size
t_grid_FDM = np.arange(0, T+dt_FDM, dt_FDM)  # time grid
b_FDM, c_FDM, p_FDM = InitiateProcesses(t_grid_FDM, sig, vers="hjb-quad", dmax=d_grid[-1], seed=True, display=True, save=True) # forecasting process b, and price processes c and p



""" Parameters for hjb """
M = 500 # Trajectories of SDE
dt_MC = 0.001 # time discretization size
t_grid_MC = np.arange(0, T+dt_MC, dt_MC) # time grid
b_MC, c_MC, p_MC =  b_FDM[::int(dt_MC/dt_FDM)], c_FDM[::int(dt_MC/dt_FDM)], p_FDM[::int(dt_MC/dt_FDM)] # forecasting process b, and price processes c and p



v = TrueValueFunction(t_grid_FDM, x_grid, vers)
PlotValueFunction3d(t_grid_FDM, x_grid, v, title=vers)



""" Easy example, verification functions """

if __name__ == "__main__":
    
    
    if FDM_im and vers[:4] != "hjb-":
        
        v_true = TrueValueFunction(t_grid_FDM, x_grid, vers)
        v_FDM_im = FDMImplicitBackwards(t_grid_FDM, x_grid, v_true[:, 0], v_true[:, -1], v_true[-1, :], b_FDM, c_FDM, p_FDM, dt_FDM, h, vers, sig, d_grid, upwind=upwind)
        PlotValueFunction3d(t_grid_FDM, x_grid, v_FDM_im, title=f"implicit FDM, dt={dt_FDM}")
        PlotValueFunctionHm(t_grid_FDM, x_grid, v_FDM_im, title=f"implicit FDM, dt={dt_FDM}")
    
    
    if FDM_im and vers[:4] == "hjb-":
        
        prodmax = 2*prodmax # maximum production
        consmax = 2*consmax # maximum consumption
        x1_grid = np.arange(-prodmax, consmax+h, h)
        trunc_left= int(prodmax/2/h)
        trunc_right = int((consmax/2+prodmax)/h)+1

        v_true = TrueValueFunction(t_grid_FDM, x1_grid, vers)
        v_FDM_im = FDMImplicitBackwards(t_grid_FDM, x1_grid, v_true[:, 0], v_true[:, -1], v_true[-1, :], b_FDM, c_FDM, p_FDM, dt_FDM, h, vers, sig, d_grid, upwind=upwind)
        np.save(f"Easy_Microgrid/ValueFunction_Results/v_FDM_im_{vers}_dt={dt_FDM}_sig={sig}.npy", [v_FDM_im[:, trunc_left:trunc_right], t_grid_FDM, x1_grid[trunc_left:trunc_right], sig, vers])
        v_FDM_im, t_grid_FDM, x1_grid, sig, vers = np.load(f"Easy_Microgrid/ValueFunction_Results/v_FDM_im_{vers}_dt={dt_FDM}_sig={sig}.npy", allow_pickle=True)
        PlotValueFunction3d(t_grid_FDM, x1_grid, v_FDM_im, title=f"Implicit FDM | sig={sig} | dt={dt_FDM}")
        PlotValueFunctionHm(t_grid_FDM, x1_grid, v_FDM_im, title=f"Implicit FDM | sig={sig} | dt={dt_FDM}", save=True)


    
    if MC:
        
        v_MC = MonteCarloEasy(t_grid_MC, x_grid, b_MC, c_MC, p_MC, dt_MC, M, vers, sig, d_grid, seed=True)
        np.save(f"Easy_Microgrid/ValueFunction_Results/MC_{vers}_M_{M}_dt={dt_MC}_sig={sig}.npy", [v_MC, t_grid_MC, x_grid, sig, vers, M])
        v_MC, t_grid_MC, x_grid, sig, vers, M = np.load(f"Easy_Microgrid/ValueFunction_Results/MC_{vers}_M_{M}_dt={dt_MC}_sig={sig}.npy", allow_pickle=True)
        PlotValueFunction3d(t_grid_MC, x_grid, v_MC, title=f"Monte Carlo | sig={sig} | dt={dt_MC} | M={M}")
        PlotValueFunctionHm(t_grid_MC, x_grid, v_MC, title=f"Monte Carlo | sig={sig} | dt={dt_MC} | M={M}", save=True)
        
        #d_policy = np.array([diesel(x_grid, c_MC[t_index], p_MC[t_index], vers, d_grid[-1]) for t_index, t_value in enumerate(t_grid_MC)])
        #PlotValueFunctionHm(t_grid_MC, x_grid, d_policy, f"Analytical optimal control | sig={sig} | dt={dt_MC}", control=True, dmax=d_grid[-1], save=True)
        
            
        









