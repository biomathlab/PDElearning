import time
from surface_fitter import SurfNN
from prediction_functions import *

'''
This function runs the surface fitting and partial derivative
approximation segment of the pipeline. Data are loaded into 
memory via the SurfNN class. By selecting options below, the 
user can either: (1) train a new artificial neural network (ann)
or (2) make predictions using the ANN, finite differences (fd),
univariate splines (sp), or bivariate splines (bsp). Predictions
are saved as dictionaries in the data folder. Each dictionary 
contains lists of input/output variables as well as their names.

In order to make animations or compute RMSE tables, see 
"make_animations.py" or "make_rmse_tables.py", respectively.
'''

# demo options
train_ann     = 0 # train surface fitting ANN?
make_ann_data = 1 # make ANN predictions?
make_fd_data  = 1 # make fin. diff. predictions?
make_sp_data  = 1 # make spline predictions?
make_bsp_data = 1 # make bispline predictions?

# pick datasets
datasets = ['advection_diffusion','fisher','fisher_nonlin'] 
inds = ['00','01','05','10','25','50']

# pick ANN model name
model_name = 'nn'

# training parameters
num_epochs = 100000
batch_size = 10
early_stop = 50
new_model  = True

# loop over data sets
for dataset in datasets:

    # loop over noise levels
    for ind in inds:

        data_name = dataset+'_'+ind

        print ''
        print data_name, model_name
        print ''

        # run options
        if train_ann == 1:
            
            t0 = time.time()
            model = SurfNN(data_name, model_name)
            model.train(epochs=num_epochs,
                        batch_size=batch_size,
                        early_stopper=early_stop,
                        new_model=new_model)
            print 'Elapsed time =', time.time() - t0, 'seconds.'
            
        if make_ann_data == 1:
            
            predict_neural_network(data_name, model_name)
            
        if make_fd_data == 1:
            
            predict_finite_differences(data_name, None)
            
        if make_sp_data == 1:
            
            predict_splines(data_name, None)
            
        if make_bsp_data == 1:
            
            predict_bisplines(data_name, None)