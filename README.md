## Abstract

The bachelor thesis focuses on the challenge of efficiently using a diesel generator within a grid-connected microgrid. The research looks at how a microgrid, consisting of a consumer, a diesel generator, a public grid and solar panels can manage itself. A stochastic model is established and aims to minimize the energy cost to the consumer through an optimal control policy. This leads to a Hamilton-Jacobi-Bellman equation, which is solved using a Regression Monte Carlo method in a discrete time setting.


## Folder structure

0_Convection_Diffusion_Equation and 0_Heat_Equation are Jupyter Notebooks created in order to get familiar with the previously unknown Finite Different Method. They are self explanatory but don't contribute to thesis.

The folders Easy_Microgrid and Intermediate_Microgrid contain saved arrays, convergence plots and approximations to the value function stemming from the three Main files.

The main files (Main_blablabla) are connected to all the other .py files.
Main_Functions_Easy.py contains a frame work the approximates the value function of the easier microgrid via an implicit Finite Difference Method or via Monte Carlo simulation.
Main_Functions_Intermediate.py contains a frame work the approximates the value function of the intermediate microgrid via an explicit Finite Difference Method or via a Regression Monte Carlo simulation.
Main_Convergence.py contains a frame work that explores the convergence behavior of the two methods used for the easier microgrid.
Regression_Monte_Carlo.py contains the main function which implements the Pseudocode at page 49 in the thesis.


## License
This repository is for everyone free to use.
