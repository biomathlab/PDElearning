import numpy as np
import matplotlib.pyplot as plt
import imageio, glob, os, time, surface_fitter
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from IPython.display import Image, display
from IPython.core.debugger import Tracer

import pdb

data_names = ['advection_diffusion']
inds = ['25']
model_names = ['nn_gamma_0','nn_gamma_0.5','nn_gamma_1.0']

font = {'family' : 'normal',
        'size'   :35}

plt.rc('font', **font)

# loop over data sets
for data_name in data_names:

    # loop over noise levels
    for ind in inds:
        
        # loop over prediction methods
        for i,model_name in enumerate(model_names):
            
            try:
                
                dataset = data_name+'_'+ind+'_'+model_name
                print dataset

                # load data using surface fitter class
                reload(surface_fitter)
                from surface_fitter import SurfNN
                nn = SurfNN(data_name+'_'+ind, None)
                x_flat, y_flat, U_min, U_max, U_shape = nn.load_data()

                # rank-1 arrays
                x = x_flat[:,0]
                t = x_flat[:,1]
                u_noise = y_flat[:,0]

                if i == 0:
                    gamma = 0.0 #gamma value for residual computation
                    plot_max = 0.1 #to aid plotting
                elif i == 1:
                    gamma = 0.5 #gamma value for residual computation
                    plot_max = 0.5 #to aid plotting
                elif i == 2:
                    gamma = 1.0 #gamma value for residual computation
                    plot_max = 1.1 #to aid plotting

                # load predictions
                surface_data = np.load('data/'+dataset+'.npy').item()
                U_pred = surface_data['inputs'][0].reshape(-1)    # un-normalized prediction
                U_pred = (U_pred - U_min)/U_max # normalized prediction

                #to avoid inf's in residuals
                plot_range = np.abs(U_pred) > 1e-4                
                #compute_residual
                residual = (U_pred[plot_range] - u_noise[plot_range])/(np.abs(U_pred[plot_range])**gamma)

                fig = plt.figure(figsize=(16,11.3))

                plt.scatter(U_pred[plot_range],residual,s=2,c='k')
                plt.ylim((-plot_max,plot_max))#(-.1,.1))
                if i == 2:
                    plt.xlim((0,0.25))
                plt.xlabel('Model (u)')
                plt.ylabel('Modified Residuals')
                plt.title('Modified Residuals for $\gamma$ = ' + str(gamma))

                if i == 0:
                    plt.yticks((-.1,0,.1))

                name = 'Error_model_selection_'+str(i)+'.png'
                plt.savefig(name,dvips=500)
                plt.cla()
                plt.clf()

                
            except:
                pass

            