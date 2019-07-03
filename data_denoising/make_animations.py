import numpy as np
import matplotlib.pyplot as plt
import imageio, glob, os, time, surface_fitter
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from IPython.display import Image, display
from IPython.core.debugger import Tracer

import pdb

data_names = ['advection_diffusion','fisher','fisher_nonlin']
inds = ['00','01','05','10','25','50']
model_names = ['finite_differences','splines','NCV_bisplines','global_NCV_bisplines_3','nn']
num_frames = 10 # make sure does not exceed total number of time points

# loop over data sets
for data_name in data_names:

    # loop over noise levels
    for ind in inds:
        
        # loop over prediction methods
        for model_name in model_names:
            
            try:
                # first delete plots from other animations
                filenames = glob.glob('animations/plot_*.png')
                for f in filenames:
                    os.remove(f)
                
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

                # rank-2 arrays
                X = x.reshape(U_shape)             # space mesh
                T = t.reshape(U_shape)             # time mesh
                U_noise = u_noise.reshape(U_shape) # normalized noisy data
                U_true = nn.U_true                 # normalized clean data
                U_x_true = nn.U_x_true             # normalized clean data
                U_xx_true = nn.U_xx_true           # normalized clean data
                U_t_true = nn.U_t_true             # normalized clean data

                # load predictions
                surface_data = np.load('data/'+dataset+'.npy').item()
                U_pred = surface_data['inputs'][0]    # un-normalized prediction
                U_x_pred = surface_data['inputs'][1]  # un-normalized prediction
                U_xx_pred = surface_data['inputs'][2] # un-normalized prediction
                U_t_pred = surface_data['outputs'][0] # un-normalized prediction
                
                # scale predictions to (0, 1)
                U_pred = (U_pred - U_min)/U_max # normalized prediction
                U_x_pred = U_x_pred/U_max       # normalized prediction
                U_xx_pred = U_xx_pred/U_max     # normalized prediction
                U_t_pred = U_t_pred/U_max       # normalized prediction
                
                # compute max residual
                max_res = max(0.01, np.max(np.abs(U_pred-U_noise)))

                for i in tqdm(range(0, len(X.T), int(np.round(len(X.T)/float(num_frames)))),
                              desc='Plotting and saving'):

                    # plot the surface
                    fig = plt.figure(figsize=(16,9))

                    ax = fig.add_subplot(231)
                    plt.plot(X[:,i],U_noise[:,i],'b.')
                    plt.plot(X[:,i],U_true[:,i],'k--')
                    plt.plot(X[:,i],U_pred[:,i],'r')
                    plt.ylim(0,1)
                    plt.legend(['Data','True','Pred'])
                    plt.legend(['Data','True','Pred'])
                    plt.title('Learned vs Noisy Surface')
                    ax.text(0.01, 0.9, 't = {0:1.4f}'.format(T[0,i]), 
                            bbox=dict(boxstyle='square', alpha=0.0))

                    ax = fig.add_subplot(232)
                    plt.plot(X[:,i],U_pred[:,i]-U_noise[:,i],'k.')
                    plt.ylim(-max_res,max_res)
                    plt.title('Residuals')

                    ax = fig.add_subplot(233)
                    plt.plot(X[:,i],U_true[:,i],'k--')
                    plt.plot(X[:,i],U_pred[:,i],'r')
                    plt.ylim(1.2*np.min(nn.U_true[:,i]),1.2*np.max(nn.U_true[:,i]))
                    plt.legend(['True','Pred'])
                    plt.title('u')

                    ax = fig.add_subplot(234)
                    plt.plot(X[:,i],U_t_true[:,i],'k--')
                    plt.plot(X[:,i],U_t_pred[:,i],'r')
                    plt.ylim(1.2*np.min(nn.U_t_true[:,i]),1.2*np.max(nn.U_t_true[:,i]))
                    plt.legend(['True','Pred'])
                    plt.title('u_t')

                    ax = fig.add_subplot(235)
                    plt.plot(X[:,i],U_x_true[:,i],'k--')
                    plt.plot(X[:,i],U_x_pred[:,i],'r')
                    plt.ylim(1.2*np.min(nn.U_x_true[:,i]),1.2*np.max(nn.U_x_true[:,i]))
                    plt.legend(['True','Pred'])
                    plt.title('u_x')

                    ax = fig.add_subplot(236)
                    plt.plot(X[:,i],U_xx_true[:,i],'k--')
                    plt.plot(X[:,i],U_xx_pred[:,i],'r')
                    plt.ylim(1.2*np.min(nn.U_xx_true[:,i]),1.2*np.max(nn.U_xx_true[:,i]))
                    plt.legend(['True','Pred'])
                    plt.title('u_xx')

                    name = 'animations/plot_'+str(i).zfill(3)+'.png'
                    plt.savefig(name)
                    plt.cla()
                    plt.clf()
                    plt.close('all')

                # animation save path
                path = 'animations'
                name = dataset

                # construct a gif out of the saved pngs
                images = []
                filenames = glob.glob('animations/plot_*.png')
                filenames.sort()
                duration = 5.0/float(len(filenames))
                for filename in filenames:
                    images.append(imageio.imread(filename))
                imageio.mimsave(path+'/'+name+'.gif', images, duration=duration)
            except:
                pass

            # last delete plots from other animations
            filenames = glob.glob('animations/plot_*.png')
            for f in filenames:
                os.remove(f)
                
            print ''
