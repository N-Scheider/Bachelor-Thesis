#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:33:15 2023

@author: noahscheider
"""

import numpy as np
import sys
import time
from HJB_Functions import *
from Monte_Carlo import *
from Verification_Functions import *
from FDM_back import *



FDM_Convergence_Space = False
FDM_Convergence_Time = False
MC_Convergence_Sim = True
MC_Convergence_Time = False

vers = "zero" # Selection of verification or hjb-function
intra = False # Comparision to true value function
# intra comparision for hjb convergence mandatory
if vers[:4]=="hjb-" and not intra: sys.exit()



""" Parameters """

T = 2 # final time
prodmax = 6 # maximum production
consmax = 6 # maximum cnosumption
sig = 1 # diffusion constant for residual demand
d_grid = np.linspace(0, 7, num=20) # discretized control grid


""" Parameters FDM """

upwind = False

h_grid_space = 1/2**np.arange(3, 10)[::-1]
dt_grid_space = 1/10**np.arange(1, 4)[::-1]

h_grid_time = 1/10**np.arange(1, 4)[::-1]
dt_grid_time = 1/2**np.arange(3, 10)[::-1]



""" Parameters MC """

K = 1 # amount of error runs
h = 0.1 # space discretization size
x_grid_MC = np.arange(-prodmax, consmax+h, h) # discretized space grid

dt_grid_MC_sim = np.array([0.01, 0.05, 0.1]) #1/10**np.arange(1, 4)[::-1]
M_grid_sim = 2**np.arange(4, 11)

dt_grid_MC_time = 1/2**np.arange(2, 9)[::-1]
M_grid_time = 10**np.arange(1, 4)






""" FDM Convergence Space"""

if __name__ == "__main__" and FDM_Convergence_Space:

    # Compute and store convergence with given prerequisites
    # for vers="hjb-" and intra=True choose bigger domain that is then truncated half of its domain
    # error = ConvergenceFDMImplicitSpace(dt_grid_space, h_grid_space, T, prodmax+prodmax*(vers[:4]=="hjb-"), consmax+consmax*(vers[:4]=="hjb-"), vers, sig, d_grid, intra, upwind)
    # np.save("Easy_Microgrid/Convergence_Results/Conv_FDM_"+vers+f"_intra={intra}_upwind={upwind}.npy", [error, h_grid_space, dt_grid_space, vers, sig, intra, upwind])
    # error, h_grid_space, dt_grid_space, vers, sig, intra, upwind = np.load("Easy_Microgrid/Convergence_Results/Conv_FDM_"+vers+f"_intra={intra}_upwind={upwind}.npy", allow_pickle=True)
    # PlotConvergenceFDMImplicitSpace(dt_grid_space, h_grid_space, error, sig, vers, intra, upwind, save=True)

    print("Start FDM Space ", time.ctime(time.time()))


    for intra in [True, False]:
        for upwind in [True, False]:
            for vers in ["zero", "hjb-quad"]:
                if vers[:4]=="hjb-" and not intra:
                    break
                else:
                    #error = ConvergenceFDMImplicitSpace(dt_grid_space, h_grid_space, T, prodmax+prodmax*(vers[:4]=="hjb-"), consmax+consmax*(vers[:4]=="hjb-"), vers, sig, d_grid, intra, upwind)
                    #np.save("Easy_Microgrid/Convergence_Results/Conv_FDM_"+vers+f"_intra={intra}_upwind={upwind}_sig={sig}.npy", [error, h_grid_space, dt_grid_space, vers, sig, intra, upwind])
                    error, h_grid_space, dt_grid_space, vers, sig, intra, upwind = np.load("Easy_Microgrid/Convergence_Results/Conv_FDM_"+vers+f"_intra={intra}_upwind={upwind}_sig={sig}.npy", allow_pickle=True)
                    PlotConvergenceFDMImplicitSpace(dt_grid_space, h_grid_space, error, sig, vers, intra, upwind, save=True)

    print("End FDM Space ", time.ctime(time.time()))



""" FDM Convergence Time"""

if __name__ == "__main__" and FDM_Convergence_Time:
    
    # Compute and store convergence with given prerequisites
    # for vers="hjb-" and intra=True choose bigger domain which is then truncated
    # error = ConvergenceFDMImplicitTime(dt_grid_time, h_grid_time, T, prodmax+prodmax*(vers[:4]=="hjb-"), consmax+consmax*(vers[:4]=="hjb-"), vers, sig, d_grid, intra, upwind)
    # np.save("Easy_Microgrid/Convergence_Results/Conv_FDM_"+vers+f"_intra={intra}_upwind={upwind}_Time.npy", [error, h_grid_time, dt_grid_time, vers, sig, intra, upwind])
    # error, h_grid_time, dt_grid_time, vers, sig, intra, upwind = np.load("Easy_Microgrid/Convergence_Results/Conv_FDM_"+vers+f"_intra={intra}_upwind={upwind}_Time.npy", allow_pickle=True)
    # PlotConvergenceFDMImplicitTime(dt_grid_time, h_grid_time, error, sig, vers, intra, upwind, save=True)

    print("Start FDM Time ", time.ctime(time.time()))
          
    for intra in [True, False]:
        for upwind in [True, False]:
            for vers in ["zero", "hjb-quad"]:
                if vers[:4]=="hjb-" and not intra:
                    break
                else:
                    #error = ConvergenceFDMImplicitTime(dt_grid_time, h_grid_time, T, prodmax+prodmax*(vers[:4]=="hjb-"), consmax+consmax*(vers[:4]=="hjb-"), vers, sig, d_grid, intra, upwind)
                    #np.save("Easy_Microgrid/Convergence_Results/Conv_FDM_"+vers+f"_intra={intra}_upwind={upwind}_sig={sig}_Time.npy", [error, h_grid_time, dt_grid_time, vers, sig, intra, upwind])
                    error, h_grid_time, dt_grid_time, vers, sig, intra, upwind = np.load("Easy_Microgrid/Convergence_Results/Conv_FDM_"+vers+f"_intra={intra}_upwind={upwind}_sig={sig}_Time.npy", allow_pickle=True)
                    PlotConvergenceFDMImplicitTime(dt_grid_time, h_grid_time, error, sig, vers, intra, upwind, save=True)

    print("End FDM Time ", time.ctime(time.time()))



""" MC Convergence Simulations """

if __name__ == "__main__" and MC_Convergence_Sim:

    #error_list = ConvergenceMC_Sim(dt_grid_MC_sim, x_grid_MC, M_grid_sim, T, vers, intra, K, sig, d_grid, seed=True)
    #np.save("Easy_Microgrid/Convergence_Results/Conv_MC_"+vers+f"_intra={intra}_sig={sig}.npy", [error_list, dt_grid_MC_sim, M_grid_sim, vers, intra, K, sig])
    #error_list, dt_grid_MC_sim, M_grid_sim, vers, intra, K, sig = np.load("Easy_Microgrid/Convergence_Results/Conv_MC_"+vers+f"_intra={intra}_sig={sig}.npy", allow_pickle=True)
    #PlotConvergenceMC_Sim(dt_grid_MC_sim, M_grid_sim, error_list, vers, intra, K, sig, save=True) # h is positional argument

    print("Start MC Simulation ", time.ctime(time.time()))
    
    for intra in [False]:
        for sig in [3]:
            for vers in ["zero"]:
                if vers[:4]=="hjb-" and not intra:
                    break
                else:
                    #error = ConvergenceMC_Sim(dt_grid_MC_sim, x_grid_MC, M_grid_sim, T, vers, intra, K, sig, d_grid, seed=True)
                    #np.save(f"Easy_Microgrid/Convergence_Results/Conv_MC_{vers}_intra={intra}_sig={sig}.npy", [error, dt_grid_MC_sim, M_grid_sim, vers, intra, K, sig])
                    error, dt_grid_MC_sim, M_grid_sim, vers, intra, K, sig = np.load(f"Easy_Microgrid/Convergence_Results/Conv_MC_{vers}_intra={intra}_sig={sig}.npy", allow_pickle=True)
                    PlotConvergenceMC_Sim(dt_grid_MC_sim, M_grid_sim, error, vers, intra, K, sig, h, save=True) # h is positional argument

    print("End MC Simulation ", time.ctime(time.time()))



""" MC Convergence Time """

if __name__ == "__main__" and MC_Convergence_Time:

    #error_list = ConvergenceMC_Time(dt_grid_MC_time, x_grid_MC, M_grid_time, T, vers, intra, K, sig, d_grid, seed=True)
    #np.save("Easy_Microgrid/Convergence_Results/Conv_MC_"+vers+f"_intra={intra}_sig={sig}_Time.npy", [error_list, dt_grid_MC_time, M_grid_time, vers, intra, K, sig])
    #error_list, dt_grid_MC_time, M_grid_time, vers, intra, K, sig = np.load("Easy_Microgrid/Convergence_Results/Conv_MC_"+vers+f"_intra={intra}_sig={sig}_Time.npy", allow_pickle=True)
    #PlotConvergenceMC_Time(dt_grid_MC_time, M_grid_time, error_list, vers, intra, K, sig, save=True) # h is positional argument
    
    print("Start MC Time ", time.ctime(time.time()))
    
    for intra in [False]:
        for sig in [3]:
            for vers in ["zero"]:
                if vers[:4]=="hjb-" and not intra:
                    break
                else:
                    #error = ConvergenceMC_Time(dt_grid_MC_time, x_grid_MC, M_grid_time, T, vers, intra, K, sig, d_grid, seed=True)
                    #np.save("Easy_Microgrid/Convergence_Results/Conv_MC_"+vers+f"_intra={intra}_sig={sig}_Time.npy", [error, dt_grid_MC_time, M_grid_time, vers, intra, K, sig])
                    error, dt_grid_MC_time, M_grid_time, vers, intra, K, sig = np.load(f"Easy_Microgrid/Convergence_Results/Conv_MC_{vers}_intra={intra}_sig={sig}_Time.npy", allow_pickle=True)
                    PlotConvergenceMC_Time(dt_grid_MC_time, M_grid_time, error, vers, intra, K, sig, save=True) # h is positional argument

    print("End MC Time ", time.ctime(time.time()))

