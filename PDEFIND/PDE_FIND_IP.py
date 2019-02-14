import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from scipy import integrate
from scipy import sparse
from scipy.signal import savgol_filter
import time

#functions to be used with an inverse problem methodology

#RK4 integration
def RK4(f):
    return lambda t, y, dt: (
            lambda dy1: (
            lambda dy2: (
            lambda dy3: (
            lambda dy4: (dy1 + 2*dy2 + 2*dy3 + dy4)/6
            )( dt * f( t + dt  , y + dy3   ) )
            )( dt * f( t + dt/2, y + dy2/2 ) )
            )( dt * f( t + dt/2, y + dy1/2 ) )
            )( dt * f( t       , y         ) )


#take in a vector, q, and data string and compute RMSE cost
def fisher_nonlin_cost(q,sigma_str):
    model_str = 'fisher_nonlin'

    if q[0] < 0:
        #avoid backwards diffusion
        return 1e6
    
    else:
        #get data directory
        if model_str == 'diffadv':
            data_dir = "/home/john/f18/research/EQL/PDE-FIND/code/Data/properror_adasamp_12_7/advection_diffusion_"
        elif model_str == 'fisher':
            data_dir = "/home/john/f18/research/EQL/PDE-FIND/code/Data/fisher/fisher_"
        elif model_str == 'fisher_nonlin':
            data_dir = "/home/john/f18/research/EQL/PDE-FIND/code/Data/nonlin_fisher/fisher_nonlin_"

        #load in data file
        mat = np.load(data_dir + sigma_str + '.npy').item()

        #dependent variable
        U=mat['U']
        #independent variables
        x=mat['x']
        t=mat['t']

        dx = x[1]-x[0]

        #sparse matrices for differentiation
        
        #u_x matrix construction
        Ux_mat_row = np.hstack((np.arange(0,len(x)-1,dtype=int), np.arange(0,len(x)-1,dtype=int),len(x)-1))
        Ux_mat_col = np.hstack((np.arange(0,len(x)-1,dtype=int), np.arange(1,len(x),dtype=int),0))
        Ux_entry = (1/dx)*np.hstack((-1*np.ones(len(x)-1),(np.ones(len(x)-1)),0))
        Ux_mat = sparse.coo_matrix((Ux_entry,(Ux_mat_row,Ux_mat_col)))

        #u_{xx} matrix construction
        x_int_range = np.arange(1,len(x)-1,dtype=int)
        Uxx_mat_row = np.hstack((x_int_range,x_int_range,x_int_range,len(x)-1))
        Uxx_mat_col = np.hstack((x_int_range-1,x_int_range,x_int_range+1,0))
        Uxx_entry = (1/(dx**2))*np.hstack((np.ones(len(x)-2),-2*np.ones(len(x)-2),(np.ones(len(x)-2)),0))
        Uxx_mat = sparse.coo_matrix((Uxx_entry,(Uxx_mat_row,Uxx_mat_col)))

        #PDE model
        def PDE_RHS(t, y):
            return q[0]*Uxx_mat.dot(y) + q[1]*y*Uxx_mat.dot(y) + q[2]*Ux_mat.dot(y)**2 + q[3]*y + q[4]*y**2 + q[5]

        #y0 = savgol_filter(U[0,],31,3)    # smooth initial timepoint for initial value
        y0 = U[0,]    #initial value
        y = np.zeros((len(t), len(x)))   # array for solution
        y[0, :] = y0
        r = integrate.ode(PDE_RHS).set_integrator("dopri5")  # choice of method
        r.set_initial_value(y0, t[0])   # initial values for integration
        for i in range(1, t.size):
           y[i, :] = r.integrate(t[i]) # get one more value, add it to the array
           if not r.successful():
                #if integration breaks, then not enough diffusion and solution explodes -- set all remaining points to zero for high cost function evaluation
                y[i:,:] = 1e-5
                break 
        rel_ind = np.abs(y)>1e-6
        #return GLS cost function
        return sum((((y[rel_ind] - U[rel_ind])/y[rel_ind])**2))