#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 11:09:41 2023

@author: noahscheider
"""

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



# """ Evaluating value function on DoI """

# # Initialize storage for the final value function and control policy in t_grid, x_grid for linear and cubic interpolation
# v_final = np.zeros((len(t_grid), len(x_grid))) # terminal condition is 0
# d_final = np.zeros((len(t_grid), len(x_grid))) # control on terminal time 0

# d_controlled_l = np.zeros((M*len(x_grid), len(t_grid))) 

# x_traj_final = np.zeros((M*len(x_grid), len(t_grid))) # gets overwritten along the loop, earlier times will stay in array but not accessed


# for n, t in enumerate(t_grid[:-1]): # Terminal value given by terminal condition
    
#     # Predicting continuation value for newly gotten point x_traj_final[i][m, n]
#     cont_value_n = np.array([regress_models[n][d_index].predict(
#     poly.fit_transform(x_grid.reshape(-1, 1))) for d_index in range(len(d_grid))])
    
#     # Simple grid discretization by evaluating on different potential controls and storing argsup  
#     d_grid_evaluated = np.array([gain_function(x_grid, c[n], p[n], d_value, "hjb-quad")*dt for d_value in d_grid]) + cont_value_n
#     magic = np.argmax(d_grid_evaluated, axis=0) # evaluating supremum, only first maximum considered, works fine for array of shape (n,1)
    
#     # Storing optimal control and use it for calculating next point of SDE
#     v_final[n, :] = gain_function(x_grid, c[n], p[n], d_grid[magic], "hjb-quad")*dt
#     d_final[n, :] = d_grid[magic]
    
#     d_controlled_l[:, n] = np.repeat(d_grid[magic], M)


#     # resampling M paths with control in grid points, control at every intial value x_i of trajectory is the same
#     x_traj_final[:, n:n+2] = IntermediateSDEfix(2, M*len(x_grid), np.repeat(x_grid, M), d_controlled_l, b, dt, lam, sig, shift=n)


#     v_along_traj = np.zeros(M*len(x_grid)) 
    
#     # Last resampled trajectory
    
#     for n_tilde, t_tilde in enumerate(t_grid[n:-2]): # if n is pen-ultimate, then second-loop isn't needed

#         n_tilde1 = n_tilde+n+2
#         n_tilde = n_tilde+n+1
                    
#         # Predicting continuation value for newly gotten point x_traj_final[i*M+m, n]
#         cont_value_n = np.array([regress_models[n_tilde][d_index].predict(
#             poly.fit_transform(x_traj_final[:, n_tilde].reshape(-1, 1))) for d_index in range(len(d_grid))])
#         # every d corresponds to a row with columns i*M:(i+1)*M, for all i in 0:len(x1_grid)
        
#         # Simple grid discretization by evaluating on different potential controls and storing argsup  
#         d_grid_evaluated = np.array([gain_function(x_traj_final[:, n_tilde], c[n_tilde], p[n_tilde], d_value, "hjb-quad")*dt for d_value in d_grid]) + cont_value_n
#         max_index = np.argmax(d_grid_evaluated, axis=0) #evaluating supremum, only first maximum considered, columnwise max
        
#         # Storing optimal control and use it for calculating next point of SDE
#         d_controlled_l[:, n_tilde] = d_grid[max_index] # gets overwritten but okay as only used once
#         x_traj_final[:, n_tilde1] =  IntermediateSDEfix(2, M*len(x_grid), x_traj_final[:, n_tilde], d_controlled_l, b, dt, lam, sig, shift=n_tilde)[:, 1] # [:, 1], in order to get next evaluation of SDE
        
#         v_along_traj += gain_function(x_traj_final[:, n_tilde], c[n_tilde], p[n_tilde], d_controlled_l[:, n_tilde], "hjb-quad")*dt
    
#     v_final[n, :] = (v_final[n, :] + np.array([sum(v_along_traj[i*M:(i+1)*M]) for i in range(len(x_grid))]))/M # Terminal condition is 0
