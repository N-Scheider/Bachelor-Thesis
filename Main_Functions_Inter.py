#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:21:52 2023

@author: noahscheider
"""


import numpy as np
import time
from HJB_Functions import *
from Monte_Carlo import *
from Regression_Monte_Carlo import RegressionMC_improved, RegressionMC_easy
from Verification_Functions import *
from FDM_back import *



""" General parameters """
T = 2 # final Time
prodmax = 6 # maximum production
consmax = 6 # maximum consumption
h = 0.2 # space discretization
x_grid = np.arange(-prodmax, consmax+h, h)
sig = 4 # deviation strength
d_grid = np.linspace(0, 7, num=20)
#d_grid = np.zeros(20)# diesel off


""" Parameters for FDM """
dt_FDM = 0.001 # time discretization size
t_grid_FDM = np.arange(0, T+dt_FDM, dt_FDM)  # time grid
b_FDM, c_FDM, p_FDM = InitiateProcesses(t_grid_FDM, sig, seed=True) # forecasting process b, and price processes c and p



""" Parameters for (intermediate-)hjb """
lam = 0 # Penalization term for
M = 1000 # Trajectories of SDE
Q = 6 # Polynomial degree of basis functions
L = 4 # Level of optimal control optimisation
dt_MC = 0.001 # time discretization size
t_grid_MC = np.arange(0, T+dt_MC, dt_MC) # time grid
# b_MC, c_MC, p_MC = InitiateProcesses(t_grid_MC, sig, seed=True) # forecasting process b, and price processes c and p
b_MC, c_MC, p_MC = b_FDM[::int(dt_MC/dt_FDM)], c_FDM[::int(dt_MC/dt_FDM)], p_FDM[::int(dt_MC/dt_FDM)]



vers = "" # empty to run intermediate-hjb
FDM_ex = True
RMC = False





if __name__ == "__main__":

    if FDM_ex:
        
        prodmax = 2*prodmax # maximum production
        consmax = 2*consmax # maximum consumption
        x_grid_FDM = np.arange(-prodmax, consmax+h, h)
        trunc_left= int(prodmax/2/h)
        trunc_right = int((consmax/2+prodmax)/h)+1
        
        v_true = TrueValueFunction(t_grid_FDM, x_grid_FDM, vers) # gives back zeros everywhere
    
        v_FDM_ex, d_FDM_ex = FDMExplicitBackwards(t_grid_FDM, x_grid_FDM, v_true[:, 0], v_true[:, -1], v_true[-1, :], b_FDM, c_FDM, p_FDM, dt_FDM, h, "intermediate-hjb", lam, sig, d_grid)
        np.save(f"Intermediate_Microgrid/ValueFunction_Results/v_FDM_ex_lam={lam}_sig={sig}_dt={dt_FDM}_{vers}.npy", [v_FDM_ex[:, trunc_left:trunc_right], d_FDM_ex[:, trunc_left:trunc_right], t_grid_FDM, x_grid_FDM[trunc_left:trunc_right]])
        v_FDM_ex, d_FDM_ex, t_grid_FDM, x_grid_FDM = np.load(f"Intermediate_Microgrid/ValueFunction_Results/v_FDM_ex_lam={lam}_sig={sig}_dt={dt_FDM}_{vers}.npy", allow_pickle=True)
        PlotValueFunction3d(t_grid_FDM, x_grid_FDM, v_FDM_ex, title=f"Explicit FDM | lam={lam} | sig={sig} | dt={dt_FDM}")
        PlotValueFunctionHm(t_grid_FDM, x_grid_FDM, v_FDM_ex, title=f"Explicit FDM | lam={lam} | sig={sig} | dt={dt_FDM}", save=True)
        PlotValueFunctionHm(t_grid_FDM, x_grid_FDM, d_FDM_ex, title=f"Control Explicit FDM | lam={lam} | sig={sig} | dt={dt_FDM}", control=True, dmax=d_grid[-1], save=True)


    
    if RMC:
        
        # Compute Regression Monte Carlo

        print("Start RMC ", time.ctime(time.time()))
        
        v_eval_old, d_eval_old = RegressionMC_improved(t_grid_MC, x_grid, b_MC, c_MC, p_MC, dt_MC, M, lam, Q, L, d_grid, sig, vers=vers)
        np.save(f"Intermediate_Microgrid/ValueFunction_Results/RMC_lam={lam}_sig={sig}_dt={dt_MC}_M={M}_Q={Q}_L={L}_vers={vers}.npy", [v_eval_old, d_eval_old, t_grid_MC, x_grid, lam, sig, dt_MC, M, Q, L, vers])
        v_eval_old, d_eval_old, t_grid_MC, x_grid, lam, sig, dt_MC, M, Q, L, vers = np.load(f"Intermediate_Microgrid/ValueFunction_Results/RMC_lam={lam}_sig={sig}_dt={dt_MC}_M={M}_Q={Q}_L={L}_vers={vers}.npy", allow_pickle=True)
        PlotValueFunction3d(t_grid_MC, x_grid, v_eval_old, title=f"RMC | {vers} | lam={lam} | sig={sig} | dt={dt_MC} | M={M} | Q={Q} | L={L}")
        PlotValueFunctionHm(t_grid_MC, x_grid, v_eval_old, title=f"RMC | lam={lam} | sig={sig} | dt={dt_MC} | M={M} | Q={Q} | L={L}", save=True)
        PlotValueFunctionHm(t_grid_MC, x_grid, d_eval_old, title=f"Control RMC | lam={lam} | sig={sig} | dt={dt_MC} | M={M} | Q={Q} | L={L}", control=True, dmax=d_grid[-1], save=True)

        print("End RMC ", time.ctime(time.time()))







# Code to get error for quadratic approximation with various parameters
# v_true = TrueValueFunction(t_grid_FDM, x_grid, "quadratic")

# # Regression Monte Carlo refines x_grid by 10
# for lam in [0, 1, 2]:
#     for M in [100, 500, 1000]:
#         for dt_MC in [0.01, 0.05, 0.1]:
#             t_grid_MC = np.arange(0, T+dt_MC, dt_MC) # time grid
#             b_MC, c_MC, p_MC = b_FDM[::int(dt_MC/dt_FDM)], c_FDM[::int(dt_MC/dt_FDM)], p_FDM[::int(dt_MC/dt_FDM)]
#             #v_eval_list, d_eval_list = RegressionMC_improved(t_grid_MC, x_grid, b_MC, c_MC, p_MC, dt_MC, M, lam, Q, L, d_grid, sig, vers=vers)
#             #np.save(f"Intermediate_Microgrid/ValueFunction_Results/RMC_liste_lam={lam}_sig={sig}_dt={dt_MC}_M={M}_Q={Q}_L={L}_vers={vers}.npy", [v_eval_list, d_eval_list, t_grid_MC, x_grid, lam, sig, dt_MC, M, Q, L, vers])
#             v_eval_list, d_eval_list, t_grid_MC, x_grid, lam, sig, dt_MC, M, Q, L, vers = np.load(f"Intermediate_Microgrid/ValueFunction_Results/RMC_liste_lam={lam}_sig={sig}_dt={dt_MC}_M={M}_Q={Q}_L={L}_vers={vers}.npy", allow_pickle=True)
            
#             for L_index in [0,2,4]:
#                 print(f"MC_liste_lam={lam}_sig={sig}_dt={dt_MC}_M={M}_Q={Q}_vers={vers}, L={L_index} ", round(np.linalg.norm(v_eval_list[L_index]-v_true[::int(dt_MC/dt_FDM), ], "fro")/np.sqrt(len(t_grid_MC)*len(x_grid)), 4))







# Code to get error in ||.||_{gl} norm for different solutions of hjb-quad
# v_FDM_im, t_grid_FDM, x1_grid, sig, vers = np.load(f"Easy_Microgrid/ValueFunction_Results/v_FDM_im_hjb-quad_dt=0.001_sig=4.npy", allow_pickle=True)

# v_MC, t_grid_MC, x_grid, sig, vers, M = np.load(f"Easy_Microgrid/ValueFunction_Results/MC_hjb-quad_M_500_dt=0.001_sig=4.npy", allow_pickle=True)
# print("alternativ", round(np.linalg.norm(v_FDM_im-v_MC, "fro")/np.sqrt(len(t_grid_MC)*len(x_grid)), 4))

d_policy = np.array([diesel(x_grid, c_FDM[t_index], p_FDM[t_index], "hjb-quad", d_grid[-1]) for t_index, t_value in enumerate(t_grid_FDM)])

v_FDM_ex, d_FDM_ex, t_grid_FDM, x_grid_FDM = np.load(f"Intermediate_Microgrid/ValueFunction_Results/v_FDM_ex_lam=0_sig=4_dt=0.001_.npy", allow_pickle=True)
# print("left", round(np.linalg.norm(v_FDM_im-v_FDM_ex, "fro")/np.sqrt(len(t_grid_FDM)*len(x_grid)), 4))
print("left policy", round(np.linalg.norm(d_policy-d_FDM_ex, "fro")/np.sqrt(len(t_grid_FDM)*len(x_grid_FDM)), 4))

# v_eval_old, d_eval_old, t_grid_MC, x_grid, lam, sig, dt_MC, M, Q, L = np.load(f"Intermediate_Microgrid/ValueFunction_Results/RMC_lam=0_sig=4_dt=0.001_M=1000_Q=6_L=4.npy", allow_pickle=True)
# print("right", round(np.linalg.norm(v_FDM_im-v_eval_old, "fro")/np.sqrt(len(t_grid_MC)*len(x_grid)), 4))
# print("right policy", round(np.linalg.norm(d_policy-d_eval_old, "fro")/np.sqrt(len(t_grid_MC)*len(x_grid)), 4))


# v_eval_old, d_eval_old, t_grid_MC, x_grid, lam, sig, dt_MC, M, Q, L, vers = np.load(f"Intermediate_Microgrid/ValueFunction_Results/RMC_lam=2_sig=4_dt=0.001_M=1000_Q=6_L=4_vers=.npy", allow_pickle=True)
# v_FDM_ex, d_FDM_ex, t_grid_FDM, x_grid_FDM = np.load(f"Intermediate_Microgrid/ValueFunction_Results/v_FDM_ex_lam=2_sig=4_dt=0.001_.npy", allow_pickle=True)


# v_eval_old_nd, d_eval_old_nd, t_grid_MC, x_grid, lam, sig, dt_MC, M, Q, L, vers = np.load(f"Intermediate_Microgrid/ValueFunction_Results/RMC_lam=2_sig=4_dt=0.001_M=1000_Q=6_L=4_vers=_nodiesel.npy", allow_pickle=True)
# v_FDM_ex_nd, d_FDM_ex, t_grid_FDM, x_grid_FDM = np.load(f"Intermediate_Microgrid/ValueFunction_Results/v_FDM_ex_lam=2_sig=4_dt=0.001__nodiesel.npy", allow_pickle=True)


# print("RMC:", (v_eval_old>=v_eval_old_nd).all())
# print("RMC %", np.mean(1-(v_eval_old>=v_eval_old_nd)))
# print("RMC max", np.max(np.abs(v_eval_old-v_eval_old_nd)))
# print("RMC gl", round(np.linalg.norm(v_eval_old-v_eval_old_nd, "fro")/np.sqrt(len(t_grid_MC)*len(x_grid)), 4))
# print("Ex:", (v_FDM_ex>=v_FDM_ex_nd).all())




    