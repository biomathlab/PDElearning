import numpy as np
import tensorflow as tf
import matplotlib, os, math, sys
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import convolve2d
from IPython.core.debugger import Tracer
from keras.layers import Input, Concatenate, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers
import keras.backend as K
import pdb
# import custom activations and loss functions
from custom_functions import *


class SurfNN():
    
    def __init__(self, data_name, model_name):
        
        # data parameters
        self.data_dir = 'data'
        self.data_name = data_name
        
        # network parameters
        self.num_neurons = 1000
        self.hidden_activation = softplus_layer
        self.output_activation = softplus
        self.reg = regularizers.l2
        self.reg_str = 0.0 
        
        # training parameters
        self.train_perc = 0.9
        self.loss = gls_thresh_loss
        self.optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        
        # plotting parameters
        self.plot_dir = 'plots'
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
            
        # saving parameters
        self.model_dir = 'checkpoints'
        self.model_name = model_name
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.animations_dir = 'animations'
        if not os.path.exists(self.animations_dir):
            os.makedirs(self.animations_dir)
        
        # build the surface fitter
        self.nn = self.build_nn()
        self.nn.compile(loss=self.loss, optimizer=self.optimizer, metrics=[gls_loss])

    def load_data(self):
        
        '''
        Loads coordinate vectors and surface data from .npy files at specified location. 
        Coordinate vectors are converted to mesh grids and flattened out. Surface data
        are flattened out and scaled to [0, 1]. Clean data are loaded in and scaled using
        min and max values from noisy surface data.
        
        Input:
            
        Output:
        
            x_flat  : array of ordered pairs of coordinate vectors [N x 2]
            y_flat  : array of flattened noisy surface values [N x 1]
            U_min   : minimum value of noisy surface
            U_max   : maximum value of (noisy surface - U_min)
            U.shape : surface shape  
        '''
        
        # import data file
        data = np.load(self.data_dir+'/'+self.data_name+'.npy').item()
        
        # load coordinate vectors (independent variables)
        x = data['x']
        t = data['t']
        x_coord_vecs = [x,t] # [space x time]
        
        # load scalar response surface (dependent variable)
        U = data['U'].T # [space x time]
        
        # compute extrema
        U_min = float(np.min(U))
        U_max = float(np.max(U - U_min))
        
        # normalize surface
        U = (U - U_min)/U_max

        # reshape inputs and outputs
        x_coord_mats = np.meshgrid(*x_coord_vecs)
        if x_coord_mats[0].shape != U.shape: # if shape mismatch
            x_coord_mats = [X.T for X in x_coord_mats] # fix it
        x_flat = np.asarray([X.reshape(-1) for X in x_coord_mats]).T # [N x 2]
        y_flat = U.reshape(-1)[:,np.newaxis] # [N x 1]
        
        # load analytic solutions
        name = self.data_name[:-2]+'00.npy'
        data = np.load(self.data_dir+'/'+name).item()
        self.U_true = (data['U'].T - U_min)/U_max
        self.U_t_true = data['U_t'].T/U_max
        self.U_x_true = data['U_x'].T/U_max
        self.U_xx_true = data['U_xx'].T/U_max

        return x_flat, y_flat, U_min, U_max, U.shape
        
    def build_nn(self):
        
        '''
        Builds the surface fitting neural network with options specified in __init__. 
        Also prints model summary to screen.
        
        Input:
            
        Output:
        
            model : Keras model
        '''
        
        # inputs
        x = Input(shape=(1,), name='x')
        t = Input(shape=(1,), name='t')
        X = Concatenate(name='X', axis=1)([x,t])
        layers = [X]
        
        # hidden layer 1
        layers.append(Dense(units=self.num_neurons, 
                            activation='linear', 
                            activity_regularizer=self.reg(self.reg_str),
                            name='z1')(layers[-1]))
        layers.append(self.hidden_activation(name='y1')(layers[-1]))
        
        # output layer
        y_pred = Dense(units=1, 
                       activation=self.output_activation, 
                       name='y2')(layers[-1]) 
        
        # build the model
        model = Model(inputs=[x,t], outputs=y_pred)
        
        return model
    
    def train(self, epochs, batch_size=32, early_stopper=100, new_model=True):
        
        '''
        Trains the surface fitting neural network. Data is loaded and split into
        training and validation sets. Then for some number of epochs, the surface
        fitter is trained until stopping criteria is reached. Every time 
        validation error improves, plots and saves.
        
        Input:
        
            epochs        : max number of epochs for training
            batch_size    : batch size for training
            early_stopper : early stopping based on validation loss
            new_model     : boolean for starting training from scratch
            
        Output:
        '''
        
        # load the data
        x_flat, y_flat, U_min, U_max, U_shape = self.load_data()
        X = x_flat[:,0].reshape(U_shape)
        T = x_flat[:,1].reshape(U_shape)
        Y = y_flat[:,0].reshape(U_shape)
        
        # shuffle and split into train/validation samples
        sample = np.random.choice([True, False], U_shape, p=[self.train_perc, 1.0-self.train_perc])
        x_train = X[sample]
        t_train = T[sample]
        y_train = Y[sample]
        x_val = X[~sample]
        t_val = T[~sample]
        y_val = Y[~sample]
        
        # plot initial network performance
        self.plotter(-1, x_flat, y_flat, U_shape, 0)
        
        # determine batch size
        if batch_size == None:
            n = len(x_flat)
        else:
            n = batch_size
        
        # initialize best validation loss
        best_val_loss = float('inf')
        
        # initialize plot counter
        plot_count = 1
        
        # continue training from last time?
        if new_model == False:
            self.loader()

        # begin training
        for epoch in range(epochs):
            
            # update the parameters
            self.nn.fit([x_train, t_train], [y_train], epochs=1, batch_size=n, verbose=0);
            
            # evaluate the model, first entry is loss+reg+penalty, second is loss
            evals_train = self.nn.evaluate([X[sample], T[sample]], [Y[sample]], verbose=0)[1]
            evals_val = self.nn.evaluate([x_val, t_val], [y_val], verbose=0)[1]
            
            # save model if val loss improved
            if evals_val < best_val_loss:
                
                # set the iteration for the last improvement to current
                last_improvement = epoch
                
                # update the best-known validation loss
                best_val_loss = evals_val
                
                # printed to show improvement found
                improved_str = '*'
                
                # save model to JSON
                self.saver()
                
                # plot learned surface
                self.plotter(epoch, x_flat, y_flat, U_shape, plot_count)
                
                # update counter
                plot_count += 1
                
            else:
                
                # printed to show no improvement 
                improved_str = ''
            
            # Plot the progress
            print ('%d [Train loss: %1.4e] [Val loss: %1.4e] ' \
                   %(epoch+1, evals_train, evals_val) + improved_str)
            
            # if no improvement found for some time, stop training
            if epoch - last_improvement > early_stopper:

                print ''
                print("No improvement found in a while, stopping training.")
                print ''

                break
                
            # if the cost explodes, kill the process
            if math.isnan(evals_train) or math.isinf(evals_train):
                sys.exit("Optimization failed, train cost = inf/nan.")
            if math.isnan(evals_val) or math.isinf(evals_val):
                sys.exit("Optimization failed, val cost = inf/nan.")
        
        print ''
    
    def saver(self):
        
        '''
        Saves weights of current surface fitting neural network. 
        '''
        
        # save names
        name = self.model_dir+'/'+self.model_name+'_'+self.data_name
        
        # save weights
        self.nn.save_weights(name+'.h5') 
                
    def loader(self):
        
        '''
        Loads weights into current surface fitting neural network. 
        '''
        
        # save names
        name = self.model_dir+'/'+self.model_name+'_'+self.data_name
        
        # load weights
        self.nn.load_weights(name+'.h5')
        
    def plotter(self, epoch, X, u_true, U_shape, plot_count):
        
        '''
        Plots current progress of surface fitter. 
        
        Input:
        
            epoch      : current epoch
            X          : array of ordered pairs of coordinate vectors
            y_true     : array of flattened noisy surface values
            U_shape    : surface shape
            plot_count : plot iteration
            
        Output:
        '''
        
        # save name
        name = self.plot_dir+'/'+self.model_name+'_'+self.data_name
        name += '_'+str(plot_count).zfill(3)+'.png'
        
        # input data
        x = X[:,0][:,np.newaxis]
        t = X[:,1][:,np.newaxis]
        
        # evaluate current network
        u_pred = self.nn.predict([x, t])
        
        # reshape into 2D arrays
        X = x.reshape(U_shape)           # (space x time)
        T = t.reshape(U_shape)           # (space x time)
        U_true = u_true.reshape(U_shape) # (space x time)
        U_pred = u_pred.reshape(U_shape) # (space x time)
        
        # compute maximum residual
        max_res = np.max(np.abs(U_true - U_pred))
        
        # plot the surface
        x_steps = 2
        t_steps = x_steps*max(X.shape[1]/X.shape[0], 1)
        fig = plt.figure(figsize=(16,9))
        
        # learned surface and data
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(X.T, T.T, U_pred.T, color='r', alpha=0.4)
        ax.scatter(X[::x_steps,::t_steps].reshape(-1), 
                   T[::x_steps,::t_steps].reshape(-1), 
                   U_true[::x_steps,::t_steps].reshape(-1), c='k', s=1)
        ax.set_zlim(0,1)
        plt.title('Learned Surface, Epoch = '+str(epoch+1))
        plt.xlabel('x')
        plt.ylabel('t')
        
        # surface and data residual 
        ax = fig.add_subplot(222)
        c = ax.pcolormesh(X.T, T.T, (U_true - U_pred).T, vmin=-max_res, vmax=max_res)
        ax.set_title('Residual')
        plt.xlabel('x')
        plt.ylabel('t')
        fig.colorbar(c, ax=ax)
        
        # save and close
        plt.savefig(name)
        plt.close(fig)
            
    def predict(self, X=None):
        
        '''
        Make predictions of denoised surface values and partial derivative approximations.
        
        Input:
        
            X : array of coordinate vectors at which to evaluate the trained network 
                if X = None, then use original mesh grid from load_data()
            
        Output:
        
            surface_data : dictionary of inputs and UN-NORMALIZED outputs for 
                           data-driven discovery methods like PDE-FIND or EQL
        '''
        
        # if no inputs specified, use default
        if X == None:
            X, _, U_min, U_max, U_shape = self.load_data()
        
        # load the pretrained model
        self.loader()
        
        # define Keras inputs
        x = self.nn.inputs[0]
        t = self.nn.inputs[1]
        
        # partial derivatives
        d0 = [self.nn.outputs[0]]
        dt = K.gradients(d0, t)
        dx = K.gradients(d0, x)
        dxx = K.gradients(dx, x)

        # reshape input arrays
        x_in = X[:,0][:,np.newaxis]
        t_in = X[:,1][:,np.newaxis]
        
        # build functions
        U = K.function([x, t], d0)
        U_t = K.function([x, t], dt)
        U_x = K.function([x, t], dx)
        U_xx = K.function([x, t], dxx)
        
        # compute solutions to partials
        U = U([x_in, t_in])[0]
        U_t = U_t([x_in, t_in])[0]
        U_x = U_x([x_in, t_in])[0]
        U_xx = U_xx([x_in, t_in])[0]
        
        # reshape into surfaces and un-normalize
        U = U_max*U.reshape(U_shape) + U_min
        U_t = U_max*U_t.reshape(U_shape)
        U_x = U_max*U_x.reshape(U_shape)
        U_xx = U_max*U_xx.reshape(U_shape)
        T = X[:,1].reshape(U_shape)
        X = X[:,0].reshape(U_shape)
        
        # convert to lists
        inputs = [U, U_x, U_xx]
        outputs = [U_t]
        indep_vars = [X, T]
        input_names = ['U','U_x','U_xx']
        output_names = ['U_t']
        indep_var_names = ['X', 'T']
        
        # store everything in dictionary
        surface_data = {}
        surface_data['inputs'] = inputs
        surface_data['outputs'] = outputs
        surface_data['input_names'] = input_names
        surface_data['output_names'] = output_names
        surface_data['indep_vars'] = indep_vars
        surface_data['indep_var_names'] = indep_var_names

        return surface_data
    
    def make_surface_data(self, X=None):
        
        '''
        Make predictions of denoised values / partial derivatives and save.
        
        Input:
        
            X : array of coordinate vectors at which to evaluate the trained network 
                if X = None, then use original mesh grid from load_data()
            
        Output:
        
        '''
            
        # compute partial derivatives
        surface_data = self.predict(X)
        
        # save the data
        np.save(self.data_dir+'/'+self.data_name+'_'+self.model_name, surface_data)
