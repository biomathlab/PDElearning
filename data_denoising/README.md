This folder contains the code that denoise data and approximate their partial derivatives with various methods.

DEMO.py - contains the code to train a new surface fitting artificial neural network (ANN) or make predictions using the ANN, finite difference, spline, local bi-spline, local NCV bi-spline, or global NCV bi-spline methods. All ANN parameters are stored in the checkpoints folder. All predictions are automatically stored in the data folder. 

prediction_functions.py - contains the code to forward evaluate the ANN, finite difference, spline, local bi-spline, local NCV bi-spline, and global NCV bi-spline methods

custom_functions.py - contains functions used for training new ANN models. 

surface_fitter.py - contains the SurfNN class which handles loading data, training new ANNs, and making predictions. This code can also save figures in the plots folder during training to illustrate ANN convergence.

spline_custom_functions.py - contains the code used for implementing the global NCV bi-spline method

make_rmse_tables.py - To compare the different methods, this code computes relative mean square errors (RMSEs) between the true function/derivative values and the approximations. 

make_animations.py - To make animations, this code saves a number of plots in the animations folder and combines them into a .gif file. The plots are deleted after the .gif file is complete.

generate_advection_diffusion_data.py, generate_fisher_data.py, generate_fisher_nonlin_data.py  - code to add proportional error to precomputed noiseless datasets and save the noisy versions in the data folder.

Note: The directory PDElearning/surface_fitting/data/ contains the same data as PDElearning/PDEFIND/Data/
