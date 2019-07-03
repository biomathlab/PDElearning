import numpy as np
import time, surface_fitter, math, pdb
from scipy.signal import convolve2d
from numpy import unique as un
from scipy.interpolate import splrep, splev, bisplrep, bisplev
from spline_custom_functions import GLS_spline_train_val

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

'''
This function computes finite difference approximations
of partial derivatives from raw data using central
differences on interior points and forward differences
at the boundaries. Approximations are saved in the data
folder.
'''

def predict_finite_differences(dataset, model_name=None):
    
    print 'finite differences, '+dataset

    # load data using SurfNN class
    reload(surface_fitter)
    from surface_fitter import SurfNN
    nn = SurfNN(dataset, model_name)
    x_flat, y_flat, U_min, U_max, U_shape = nn.load_data()

    # rank-1 arrays
    x = x_flat[:,0]
    t = x_flat[:,1]
    u_noise = y_flat[:,0] 

    # rank-2 arrays
    X = x.reshape(U_shape)             # space mesh
    T = t.reshape(U_shape)             # time mesh
    U_noise = u_noise.reshape(U_shape) # normalized noisy data

    # finite difference params
    x_u = np.unique(x)
    t_u = np.unique(t)
    dx = float(x_u[1])-float(x_u[0])
    dt = float(t_u[1])-float(t_u[0])

    # finite difference filter
    Dx = np.zeros((3,3))
    Dx[0,1] = 1.0/(2*dx)
    Dx[2,1] = -1.0/(2*dx)
    Dt = np.zeros((3,3))
    Dt[1,0] = 1.0/(2*dt)
    Dt[1,2] = -1.0/(2*dt)

    # get timestamp
    t0 = time.time()

    # compute central diffs on interior, forward diffs on exterior
    U_pred = U_noise
    U_t_pred = convolve2d(U_pred, Dt, boundary='fill', fillvalue=0, mode='same')
    U_t_pred[:,0] = (U_pred[:,1]-U_pred[:,0])/dt
    U_t_pred[:,-1] = (U_pred[:,-1]-U_pred[:,-2])/dt
    U_x_pred = convolve2d(U_pred, Dx, boundary='fill', fillvalue=0, mode='same')
    U_x_pred[0,:] = (U_pred[1,:]-U_pred[0,:])/dx
    U_x_pred[-1,:] = (U_pred[-1,:]-U_pred[-2,:])/dx
    U_xx_pred = convolve2d(U_x_pred, Dx, boundary='fill', fillvalue=0, mode='same')
    U_xx_pred[0,:] = (U_x_pred[1,:]-U_x_pred[0,:])/dx
    U_xx_pred[-1,:] = (U_x_pred[-1,:]-U_x_pred[-2,:])/dx

    # print time
    print 'Elapsed time =', time.time() - t0, 'seconds.'
    print ''

    # bring predictions back to original scale
    U_pred = U_max*U_pred + U_min # un-normalized prediction
    U_x_pred = U_max*U_x_pred     # un-normalized prediction
    U_xx_pred = U_max*U_xx_pred   # un-normalized prediction
    U_t_pred = U_max*U_t_pred     # un-normalized prediction

    # store everything in dictionary
    surface_data = {}
    surface_data['inputs'] = [U_pred, U_x_pred, U_xx_pred]
    surface_data['outputs'] = [U_t_pred]
    surface_data['indep_vars'] = [X, T]
    surface_data['input_names'] = ['U','U_x','U_xx']
    surface_data['output_names'] = ['U_t']
    surface_data['indep_var_names'] = ['X', 'T']

    # save the data
    np.save('data/'+dataset+'_finite_differences', surface_data)
    
'''
This function computes polynomial approximations
of partial derivatives from raw data using univariate
splines centered at interior points. Approximations 
are saved in the data folder.
'''

def predict_splines(dataset, model_name=None):
    
    print 'univariate splines, '+dataset

    # load data using SurfNN class
    reload(surface_fitter)
    from surface_fitter import SurfNN
    nn = SurfNN(dataset, model_name)
    x_flat, y_flat, U_min, U_max, U_shape = nn.load_data()

    # spline parameters
    x_w = 5     # 1/2 line width in x direction
    t_w = 5     # 1/2 line width in x direction
    x_order = 3 # polynomial x degree
    t_order = 3 # polynomial t degree

    # rank-1 arrays
    x = x_flat[:,0]
    t = x_flat[:,1]
    u_noise = y_flat[:,0]

    # rank-2 arrays
    X = x.reshape(U_shape)             # space mesh
    T = t.reshape(U_shape)             # time mesh
    U_noise = u_noise.reshape(U_shape) # normalized noisy data, dims = (x,t)
    U_pred = U_noise                   # 

    # initialize empty interior arrays
    U_x_pred_int = np.zeros([U_shape[0]-2*x_w,U_shape[1]-2*t_w])
    U_xx_pred_int = np.zeros([U_shape[0]-2*x_w,U_shape[1]-2*t_w])
    U_t_pred_int = np.zeros([U_shape[0]-2*x_w,U_shape[1]-2*t_w])

    # initialize empty full arrays
    U_x_pred = np.zeros(U_shape)
    U_xx_pred = np.zeros(U_shape)
    U_t_pred = np.zeros(U_shape)

    # get timestamp
    t0 = time.time()

    # populate interiors with spline approximations
    for i in np.arange(x_w,U_shape[0]-x_w):
        for j in np.arange(t_w,U_shape[1]-t_w):

            # get lines
            X_line = X[i-x_w:i+x_w+1, j]
            T_line = T[i, j-t_w:j+t_w+1]
            U_line_x = U_noise[i-x_w:i+x_w+1, j]
            U_line_t = U_noise[i, j-t_w:j+t_w+1]

            # interpolate
            tck_x = splrep(X_line, U_line_x, 
                           k=x_order, w=np.ones(X_line.shape))
            tck_t = splrep(T_line, U_line_t, 
                           k=t_order, w=np.ones(T_line.shape))

            # predict
            U_x_pred_int[i-x_w, j-t_w] = splev(un(X_line), tck_x, der=1)[x_w]
            U_xx_pred_int[i-x_w, j-t_w] = splev(un(X_line), tck_x, der=2)[x_w]
            U_t_pred_int[i-x_w, j-t_w] = splev(un(T_line), tck_t, der=1)[t_w]

    # populate left/right boundaries (not corners) with splines
    for i in np.arange(x_w,U_shape[0]-x_w):

        #
        # t derivatives first
        #

        # column location
        j = t_w

        # get lines
        T_line = T[i, :j+t_w+1]
        U_line_t = U_noise[i, :j+t_w+1]

        # interpolate
        tck_t = splrep(T_line, U_line_t, 
                       k=t_order, w=np.ones(T_line.shape))

        # predict
        U_t_pred[i, :j] = splev(T_line, tck_t, der=1)[:t_w]

        # column location
        j = -t_w

        # get lines
        T_line = T[i, j-t_w-1:]
        U_line_t = U_noise[i, j-t_w-1:]

        # interpolate
        tck_t = splrep(T_line, U_line_t, 
                       k=t_order, w=np.ones(T_line.shape))

        # predict
        U_t_pred[i, j:] = splev(T_line, tck_t, der=1)[-t_w:]

        #
        # x derivatives next
        #

        # column location
        j = t_w

        # get tiles
        X_tile = X[i-x_w:i+x_w+1, :j+t_w+1]
        U_tile_x = U_noise[i-x_w:i+x_w+1, :j+t_w+1]

        # loop over columns
        for t_loc in range(t_w):

            # interpolate
            tck_x = splrep(X_tile[:,t_loc], U_tile_x[:,t_loc], 
                           k=x_order, w=np.ones(X_tile[:,t_loc].shape))

            # predict
            U_x_pred[i, t_loc] = splev(un(X_tile[:,t_loc]), tck_x, der=1)[x_w]
            U_xx_pred[i, t_loc] = splev(un(X_tile[:,t_loc]), tck_x, der=2)[x_w]

        # column location
        j = -t_w

        # get tiles
        X_tile = X[i-x_w:i+x_w+1, j-t_w-1:]
        U_tile_x = U_noise[i-x_w:i+x_w+1, j-t_w-1:]

        # loop over columns
        for t_loc in range(t_w):

            # interpolate
            tck_x = splrep(X_tile[:,t_loc+t_w], U_tile_x[:,t_loc+t_w], 
                           k=x_order, w=np.ones(X_tile[:,t_loc].shape))

            # predict
            U_x_pred[i, t_loc+j] = splev(un(X_tile[:,t_loc+t_w]), tck_x, der=1)[x_w]
            U_xx_pred[i, t_loc+j] = splev(un(X_tile[:,t_loc+t_w]), tck_x, der=2)[x_w]

    # populate top/bottom boundaries (not corners) with splines
    for j in np.arange(t_w,U_shape[1]-t_w):

        #
        # x derivatives first
        #

        # row location
        i = x_w

        # get lines
        X_line = X[:i+x_w+1, j]
        U_line_x = U_noise[:i+x_w+1, j]

        # interpolate
        tck_x = splrep(X_line, U_line_x, 
                       k=x_order, w=np.ones(X_line.shape))

        # predict
        U_x_pred[:i, j] = splev(X_line, tck_x, der=1)[:x_w]
        U_xx_pred[:i, j] = splev(X_line, tck_x, der=2)[:x_w]

        # row location
        i = -x_w

        # get tiles
        X_line = X[i-x_w-1:, j]
        U_line_x = U_noise[i-x_w-1:, j]

        # interpolate
        tck_x = splrep(X_line, U_line_x, 
                       k=x_order, w=np.ones(X_line.shape))

        # predict
        U_x_pred[i:, j] = splev(X_line, tck_x, der=1)[-x_w:]
        U_xx_pred[i:, j] = splev(X_line, tck_x, der=2)[-x_w:]

        #
        # t derivatives next
        #

        # row location
        i = x_w

        # get tiles
        T_tile = T[:i+x_w+1, j-t_w:j+t_w+1]
        U_tile_t = U_noise[:i+x_w+1, j-t_w:j+t_w+1]

        # loop over columns
        for x_loc in range(x_w):

            # interpolate
            tck_t = splrep(T_tile[x_loc,:], U_tile_t[x_loc,:], 
                           k=t_order, w=np.ones(T_tile[x_loc,:].shape))

            # predict
            U_t_pred[x_loc, j] = splev(un(T_tile[x_loc,:]), tck_t, der=1)[t_w]


        # row location
        i = -x_w

        # get tiles
        T_tile = T[i-x_w-1:, j-t_w:j+t_w+1]
        U_tile_t = U_noise[i-x_w-1:, j-t_w:j+t_w+1]

        # loop over columns
        for x_loc in range(x_w):

            # interpolate
            tck_t = splrep(T_tile[x_loc+x_w,:], U_tile_t[x_loc+x_w,:], 
                           k=t_order, w=np.ones(T_tile[x_loc+x_w,:].shape))

            # predict
            U_t_pred[x_loc+i, j] = splev(un(T_tile[x_loc+x_w,:]), tck_t, der=1)[t_w]

    # populate (0,0) corner with spline approximations
    for i in range(x_w):
        T_line = T[i, :2*t_w+1]
        U_line = U_noise[i, :2*t_w+1]
        tck = splrep(T_line, U_line, 
                     k=t_order, w=np.ones(T_line.shape))
        U_t_pred[i,:t_w] = splev(un(T_line), tck, der=1)[:t_w]
    for j in range(t_w):
        X_line = X[:2*x_w+1, j]
        U_line = U_noise[:2*x_w+1, j]
        tck = splrep(X_line, U_line, 
                     k=x_order, w=np.ones(X_line.shape))
        U_x_pred[:x_w,j] = splev(un(X_line), tck, der=1)[:x_w]
        U_xx_pred[:x_w,j] = splev(un(X_line), tck, der=2)[:x_w]

    # populate (0,-1) corner with spline approximations
    for i in range(x_w):
        T_line = T[i, -2*t_w-1:]
        U_line = U_noise[i, -2*t_w-1:]
        tck = splrep(T_line, U_line, 
                     k=t_order, w=np.ones(T_line.shape))
        U_t_pred[i,-t_w:] = splev(un(T_line), tck, der=1)[-t_w:]
    for j in range(-t_w,0):
        X_line = X[:2*x_w+1, j]
        U_line = U_noise[:2*x_w+1, j]
        tck = splrep(X_line, U_line, 
                     k=x_order, w=np.ones(X_line.shape))
        U_x_pred[:x_w,j] = splev(un(X_line), tck, der=1)[:x_w]
        U_xx_pred[:x_w,j] = splev(un(X_line), tck, der=2)[:x_w]

    # populate (-1,0) corner with spline approximations
    for i in range(-x_w,0):
        T_line = T[i, :2*t_w+1]
        U_line = U_noise[i, :2*t_w+1]
        tck = splrep(T_line, U_line, 
                     k=t_order, w=np.ones(T_line.shape))
        U_t_pred[i,:t_w] = splev(un(T_line), tck, der=1)[:t_w]
    for j in range(t_w):
        X_line = X[-2*x_w-1:, j]
        U_line = U_noise[-2*x_w-1:, j]
        tck = splrep(X_line, U_line, 
                     k=x_order, w=np.ones(X_line.shape))
        U_x_pred[-x_w:,j] = splev(un(X_line), tck, der=1)[-x_w:]
        U_xx_pred[-x_w:,j] = splev(un(X_line), tck, der=2)[-x_w:]

    # populate (-1,-1) corner with spline approximations
    for i in range(-x_w,0):
        T_line = T[i, -2*t_w-1:]
        U_line = U_noise[i, -2*t_w-1:]
        tck = splrep(T_line, U_line, 
                     k=t_order, w=np.ones(T_line.shape))
        U_t_pred[i,-t_w:] = splev(un(T_line), tck, der=1)[-t_w:]
    for j in range(-t_w,0):
        X_line = X[-2*x_w-1:, j]
        U_line = U_noise[-2*x_w-1:, j]
        tck = splrep(X_line, U_line, 
                     k=x_order, w=np.ones(X_line.shape))
        U_x_pred[-x_w:,j] = splev(un(X_line), tck, der=1)[-x_w:]
        U_xx_pred[-x_w:,j] = splev(un(X_line), tck, der=2)[-x_w:]

    # embed spline interiors inside full approximations
    U_t_pred[x_w:-x_w, t_w:-t_w] = U_t_pred_int
    U_x_pred[x_w:-x_w, t_w:-t_w] = U_x_pred_int
    U_xx_pred[x_w:-x_w, t_w:-t_w] = U_xx_pred_int

    # print time
    print 'Elapsed time =', time.time() - t0, 'seconds.'
    print ''

    # bring predictions back to original scale
    U_pred = U_max*U_pred + U_min # un-normalized prediction
    U_x_pred = U_max*U_x_pred     # un-normalized prediction
    U_xx_pred = U_max*U_xx_pred   # un-normalized prediction
    U_t_pred = U_max*U_t_pred     # un-normalized prediction

    # store everything in dictionary
    surface_data = {}
    surface_data['inputs'] = [U_pred, U_x_pred, U_xx_pred]
    surface_data['outputs'] = [U_t_pred]
    surface_data['indep_vars'] = [X, T]
    surface_data['input_names'] = ['U','U_x','U_xx']
    surface_data['output_names'] = ['U_t']
    surface_data['indep_var_names'] = ['X', 'T']

    # save the data
    np.save('data/'+dataset+'_splines', surface_data)
    
'''
This function computes polynomial approximations
of partial derivatives from raw data using bivariate
splines centered at interior points. Approximations 
are saved in the data folder.
'''

def predict_bisplines(dataset, model_name=None):
    
    print 'bivariate splines, '+dataset
        
    # load data using SurfNN class
    reload(surface_fitter)
    from surface_fitter import SurfNN
    nn = SurfNN(dataset, model_name)
    x_flat, y_flat, U_min, U_max, U_shape = nn.load_data()

    # spline parameters
    x_w = 5     # 1/2 tile width in x direction
    t_w = 5     # 1/2 tile width in x direction
    x_order = 3 # polynomial x degree
    t_order = 3 # polynomial t degree

    # rank-1 arrays
    x = x_flat[:,0]
    t = x_flat[:,1]
    u_noise = y_flat[:,0]

    # rank-2 arrays
    X = x.reshape(U_shape)             # space mesh
    T = t.reshape(U_shape)             # time mesh
    U_noise = u_noise.reshape(U_shape) # normalized noisy data

    # initialize empty interior arrays
    U_pred_int = np.zeros([U_shape[0]-2*x_w,U_shape[1]-2*t_w])
    U_x_pred_int = np.zeros([U_shape[0]-2*x_w,U_shape[1]-2*t_w])
    U_xx_pred_int = np.zeros([U_shape[0]-2*x_w,U_shape[1]-2*t_w])
    U_t_pred_int = np.zeros([U_shape[0]-2*x_w,U_shape[1]-2*t_w])

    # initialize empty full arrays
    U_pred = np.zeros(U_shape)
    U_x_pred = np.zeros(U_shape)
    U_xx_pred = np.zeros(U_shape)
    U_t_pred = np.zeros(U_shape)

    # get timestamp
    t0 = time.time()

    # populate interiors with spline approximations
    for i in np.arange(x_w,U_shape[0]-x_w):
        for j in np.arange(t_w,U_shape[1]-t_w):

            # get tiles
            X_tile = X[i-x_w:i+x_w+1, j-t_w:j+t_w+1]
            T_tile = T[i-x_w:i+x_w+1, j-t_w:j+t_w+1]
            U_tile = U_noise[i-x_w:i+x_w+1, j-t_w:j+t_w+1]

            # interpolate
            tck = bisplrep(X_tile, T_tile, U_tile, kx=x_order, ky=t_order)

            # predict
            U_pred_int[i-x_w, j-t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)[x_w,t_w] 
            U_x_pred_int[i-x_w, j-t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=1, dy=0)[x_w,t_w]
            U_xx_pred_int[i-x_w, j-t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=2, dy=0)[x_w,t_w]
            U_t_pred_int[i-x_w, j-t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=1)[x_w,t_w]

    # populate exteriors with spline approximations
    for i in np.arange(x_w,U_shape[0]-x_w):

        # column location
        j = t_w

        # get tiles
        X_tile = X[i-x_w:i+x_w+1, :j+t_w+1]
        T_tile = T[i-x_w:i+x_w+1, :j+t_w+1]
        U_tile = U_noise[i-x_w:i+x_w+1, :j+t_w+1]

        # interpolate
        tck = bisplrep(X_tile, T_tile, U_tile, kx=x_order, ky=t_order)
        #https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.bisplrep.html
        # predict
        U_pred[i, :j] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)[x_w, :t_w] 
        U_x_pred[i, :j] = bisplev(un(X_tile), un(T_tile), tck, dx=1, dy=0)[x_w, :t_w]
        U_xx_pred[i, :j] = bisplev(un(X_tile), un(T_tile), tck, dx=2, dy=0)[x_w, :t_w]
        U_t_pred[i, :j] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=1)[x_w, :t_w]

        # column location
        j = -t_w

        # get tiles
        X_tile = X[i-x_w:i+x_w+1, j-t_w-1:]
        T_tile = T[i-x_w:i+x_w+1, j-t_w-1:]
        U_tile = U_noise[i-x_w:i+x_w+1, j-t_w-1:]

        # interpolate
        tck = bisplrep(X_tile, T_tile, U_tile, kx=x_order, ky=t_order)

        # predict
        U_pred[i, j:] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)[x_w, t_w+1:] 
        U_x_pred[i, j:] = bisplev(un(X_tile), un(T_tile), tck, dx=1, dy=0)[x_w, t_w+1:]
        U_xx_pred[i, j:] = bisplev(un(X_tile), un(T_tile), tck, dx=2, dy=0)[x_w, t_w+1:]
        U_t_pred[i, j:] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=1)[x_w, t_w+1:]

    # populate exteriors with spline approximations
    for j in np.arange(t_w,U_shape[1]-t_w):

        # row location
        i = x_w

        # get tiles
        X_tile = X[:i+x_w+1, j-t_w:j+t_w+1]
        T_tile = T[:i+x_w+1, j-t_w:j+t_w+1]
        U_tile = U_noise[:i+x_w+1, j-t_w:j+t_w+1]

        # interpolate
        tck = bisplrep(X_tile, T_tile, U_tile, kx=x_order, ky=t_order)

        # predict
        U_pred[:i, j] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)[:x_w, t_w] 
        U_x_pred[:i, j] = bisplev(un(X_tile), un(T_tile), tck, dx=1, dy=0)[:x_w, t_w]
        U_xx_pred[:i, j] = bisplev(un(X_tile), un(T_tile), tck, dx=2, dy=0)[:x_w, t_w]
        U_t_pred[:i, j] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=1)[:x_w, t_w]

        # row location
        i = -x_w

        # get tiles
        X_tile = X[i-x_w-1:, j-t_w:j+t_w+1]
        T_tile = T[i-x_w-1:, j-t_w:j+t_w+1]
        U_tile = U_noise[i-x_w-1:, j-t_w:j+t_w+1]

        # interpolate
        tck = bisplrep(X_tile, T_tile, U_tile, kx=x_order, ky=t_order)

        # predict
        U_pred[i:, j] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)[x_w+1:, t_w] 
        U_x_pred[i:, j] = bisplev(un(X_tile), un(T_tile), tck, dx=1, dy=0)[x_w+1:, t_w]
        U_xx_pred[i:, j] = bisplev(un(X_tile), un(T_tile), tck, dx=2, dy=0)[x_w+1:, t_w]
        U_t_pred[i:, j] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=1)[x_w+1:, t_w]

    # populate 0,0 corner with spline approximations
    X_tile = X[:2*x_w+1, :2*t_w+1]
    T_tile = T[:2*x_w+1, :2*t_w+1]
    U_tile = U_noise[:2*x_w+1, :2*t_w+1]
    tck = bisplrep(X_tile, T_tile, U_tile, kx=x_order, ky=t_order)
    U_pred[:x_w, :t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)[:x_w, :t_w] 
    U_x_pred[:x_w, :t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=1, dy=0)[:x_w, :t_w]
    U_xx_pred[:x_w, :t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=2, dy=0)[:x_w, :t_w]
    U_t_pred[:x_w, :t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=1)[:x_w, :t_w]

    # populate 0,-1 corner with spline approximations
    X_tile = X[:2*x_w+1, -2*t_w-1:]
    T_tile = T[:2*x_w+1, -2*t_w-1:]
    U_tile = U_noise[:2*x_w+1, -2*t_w-1:]
    tck = bisplrep(X_tile, T_tile, U_tile, kx=x_order, ky=t_order)
    U_pred[:x_w, -t_w:] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)[:x_w, -t_w:]
    U_x_pred[:x_w, -t_w:] = bisplev(un(X_tile), un(T_tile), tck, dx=1, dy=0)[:x_w, -t_w:]
    U_xx_pred[:x_w, -t_w:] = bisplev(un(X_tile), un(T_tile), tck, dx=2, dy=0)[:x_w, -t_w:]
    U_t_pred[:x_w, -t_w:] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=1)[:x_w, -t_w:]

    # populate -1,0 corner with spline approximations
    X_tile = X[-2*x_w-1:, :2*t_w+1]
    T_tile = T[-2*x_w-1:, :2*t_w+1]
    U_tile = U_noise[-2*x_w-1:, :2*t_w+1]
    tck = bisplrep(X_tile, T_tile, U_tile, kx=x_order, ky=t_order)
    U_pred[-x_w:, :t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)[-x_w:, :t_w]
    U_x_pred[-x_w:, :t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=1, dy=0)[-x_w:, :t_w]
    U_xx_pred[-x_w:, :t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=2, dy=0)[-x_w:, :t_w]
    U_t_pred[-x_w:, :t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=1)[-x_w:, :t_w]

    # populate -1,-1 corner with spline approximations
    X_tile = X[-2*x_w-1:, -2*t_w-1:]
    T_tile = T[-2*x_w-1:, -2*t_w-1:]
    U_tile = U_noise[-2*x_w-1:, -2*t_w-1:]
    tck = bisplrep(X_tile, T_tile, U_tile, kx=x_order, ky=t_order)
    U_pred[-x_w:, -t_w:] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)[-x_w:, -t_w:]
    U_x_pred[-x_w:, -t_w:] = bisplev(un(X_tile), un(T_tile), tck, dx=1, dy=0)[-x_w:, -t_w:]
    U_xx_pred[-x_w:, -t_w:] = bisplev(un(X_tile), un(T_tile), tck, dx=2, dy=0)[-x_w:, -t_w:]
    U_t_pred[-x_w:, -t_w:] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=1)[-x_w:, -t_w:]

    # embed spline interiors inside full approximations
    U_pred[x_w:-x_w, t_w:-t_w] = U_pred_int
    U_t_pred[x_w:-x_w, t_w:-t_w] = U_t_pred_int
    U_x_pred[x_w:-x_w, t_w:-t_w] = U_x_pred_int
    U_xx_pred[x_w:-x_w, t_w:-t_w] = U_xx_pred_int

    # print time
    print 'Elapsed time =', time.time() - t0, 'seconds.'
    print ''

    # bring predictions back to original scale
    U_pred = U_max*U_pred + U_min # un-normalized prediction
    U_x_pred = U_max*U_x_pred     # un-normalized prediction
    U_xx_pred = U_max*U_xx_pred   # un-normalized prediction
    U_t_pred = U_max*U_t_pred     # un-normalized prediction

    # store everything in dictionary
    surface_data = {}
    surface_data['inputs'] = [U_pred, U_x_pred, U_xx_pred]
    surface_data['outputs'] = [U_t_pred]
    surface_data['indep_vars'] = [X, T]
    surface_data['input_names'] = ['U','U_x','U_xx']
    surface_data['output_names'] = ['U_t']
    surface_data['indep_var_names'] = ['X', 'T']

    # save the data
    np.save('data/'+dataset+'_bisplines', surface_data)

'''
This function computes polynomial approximations
of partial derivatives from raw data using bivariate
splines (with a GLS cost function) centered at interior points. Approximations 
are saved in the data folder.
'''


def predict_NCV_bisplines(dataset, model_name=None):
    
    print 'bivariate NCV splines, '+dataset
        
    # load data using SurfNN class
    reload(surface_fitter)
    from surface_fitter import SurfNN
    nn = SurfNN(dataset, model_name)
    x_flat, y_flat, U_min, U_max, U_shape = nn.load_data()

    # spline parameters
    x_w = 5     # 1/2 tile width in x direction
    t_w = 5     # 1/2 tile width in x direction
    x_order = 3 # polynomial x degree
    t_order = 3 # polynomial t degree

    # GLS parameters
    gamma = 1.0
    thres = 1e-4

    # rank-1 arrays
    x = x_flat[:,0]
    t = x_flat[:,1]
    u_noise = y_flat[:,0]

    # rank-2 arrays
    X = x.reshape(U_shape)             # space mesh
    T = t.reshape(U_shape)             # time mesh
    U_noise = u_noise.reshape(U_shape) # normalized noisy data

    # initialize empty interior arrays
    U_pred_int = np.zeros([U_shape[0]-2*x_w,U_shape[1]-2*t_w])
    U_x_pred_int = np.zeros([U_shape[0]-2*x_w,U_shape[1]-2*t_w])
    U_xx_pred_int = np.zeros([U_shape[0]-2*x_w,U_shape[1]-2*t_w])
    U_t_pred_int = np.zeros([U_shape[0]-2*x_w,U_shape[1]-2*t_w])

    # initialize empty full arrays
    U_pred = np.zeros(U_shape)
    U_x_pred = np.zeros(U_shape)
    U_xx_pred = np.zeros(U_shape)
    U_t_pred = np.zeros(U_shape)

    # get timestamp
    t0 = time.time()

    # populate interiors with spline approximations
    for i in np.arange(x_w,U_shape[0]-x_w):
        for j in np.arange(t_w,U_shape[1]-t_w):

            # get tiles
            X_tile = X[i-x_w:i+x_w+1, j-t_w:j+t_w+1]
            T_tile = T[i-x_w:i+x_w+1, j-t_w:j+t_w+1]
            U_tile = U_noise[i-x_w:i+x_w+1, j-t_w:j+t_w+1]

            # interpolate for OLS scenario
            tck = bisplrep(X_tile, T_tile, U_tile, kx=x_order, ky=t_order)

            #.predict from OLS model
            U_pred_OLS = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)
            ##construct weight matrix from U_pred
            W = 1/np.abs(U_pred_OLS.flatten()**gamma)
            #Values below thres will have OLS model
            W[np.abs(U_pred_OLS.flatten())<thres] = 1.0
            
            #re-do spline smooth with GLS cost function
            tck = bisplrep(X_tile.flatten(), T_tile.flatten(), U_tile.flatten(),w=W,kx=x_order,ky=t_order)


            # predict
            U_pred_int[i-x_w, j-t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)[x_w,t_w] 
            U_x_pred_int[i-x_w, j-t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=1, dy=0)[x_w,t_w]
            U_xx_pred_int[i-x_w, j-t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=2, dy=0)[x_w,t_w]
            U_t_pred_int[i-x_w, j-t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=1)[x_w,t_w]

    # populate exteriors with spline approximations
    for i in np.arange(x_w,U_shape[0]-x_w):

        # column location
        j = t_w

        # get tiles
        X_tile = X[i-x_w:i+x_w+1, :j+t_w+1]
        T_tile = T[i-x_w:i+x_w+1, :j+t_w+1]
        U_tile = U_noise[i-x_w:i+x_w+1, :j+t_w+1]

        # interpolate
        tck = bisplrep(X_tile, T_tile, U_tile, kx=x_order, ky=t_order)

        #.predict from OLS model
        U_pred_OLS = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)
        ##construct weight matrix from U_pred
        W = 1/np.abs(U_pred_OLS.flatten()**gamma)
        #Values below thres will have OLS model
        W[np.abs(U_pred_OLS.flatten())<thres] = 1.0
        
        #re-do spline smooth with GLS cost function
        tck = bisplrep(X_tile.flatten(), T_tile.flatten(), U_tile.flatten(),w=W,kx=x_order,ky=t_order)


        #https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.bisplrep.html
        # predict
        U_pred[i, :j] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)[x_w, :t_w] 
        U_x_pred[i, :j] = bisplev(un(X_tile), un(T_tile), tck, dx=1, dy=0)[x_w, :t_w]
        U_xx_pred[i, :j] = bisplev(un(X_tile), un(T_tile), tck, dx=2, dy=0)[x_w, :t_w]
        U_t_pred[i, :j] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=1)[x_w, :t_w]

        # column location
        j = -t_w

        # get tiles
        X_tile = X[i-x_w:i+x_w+1, j-t_w-1:]
        T_tile = T[i-x_w:i+x_w+1, j-t_w-1:]
        U_tile = U_noise[i-x_w:i+x_w+1, j-t_w-1:]

        # interpolate
        tck = bisplrep(X_tile, T_tile, U_tile, kx=x_order, ky=t_order)

        #.predict from OLS model
        U_pred_OLS = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)
        ##construct weight matrix from U_pred
        W = 1/np.abs(U_pred_OLS.flatten()**gamma)
        #Values below thres will have OLS model
        W[np.abs(U_pred_OLS.flatten())<thres] = 1.0
        
        #re-do spline smooth with GLS cost function
        tck = bisplrep(X_tile.flatten(), T_tile.flatten(), U_tile.flatten(),w=W,kx=x_order,ky=t_order)


        # predict
        U_pred[i, j:] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)[x_w, t_w+1:] 
        U_x_pred[i, j:] = bisplev(un(X_tile), un(T_tile), tck, dx=1, dy=0)[x_w, t_w+1:]
        U_xx_pred[i, j:] = bisplev(un(X_tile), un(T_tile), tck, dx=2, dy=0)[x_w, t_w+1:]
        U_t_pred[i, j:] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=1)[x_w, t_w+1:]

    # populate exteriors with spline approximations
    for j in np.arange(t_w,U_shape[1]-t_w):

        # row location
        i = x_w

        # get tiles
        X_tile = X[:i+x_w+1, j-t_w:j+t_w+1]
        T_tile = T[:i+x_w+1, j-t_w:j+t_w+1]
        U_tile = U_noise[:i+x_w+1, j-t_w:j+t_w+1]

        # interpolate
        tck = bisplrep(X_tile, T_tile, U_tile, kx=x_order, ky=t_order)

        #.predict from OLS model
        U_pred_OLS = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)
        ##construct weight matrix from U_pred
        W = 1/np.abs(U_pred_OLS.flatten()**gamma)
        #Values below thres will have OLS model
        W[np.abs(U_pred_OLS.flatten())<thres] = 1.0
        
        #re-do spline smooth with GLS cost function
        tck = bisplrep(X_tile.flatten(), T_tile.flatten(), U_tile.flatten(),w=W,kx=x_order,ky=t_order)


        # predict
        U_pred[:i, j] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)[:x_w, t_w] 
        U_x_pred[:i, j] = bisplev(un(X_tile), un(T_tile), tck, dx=1, dy=0)[:x_w, t_w]
        U_xx_pred[:i, j] = bisplev(un(X_tile), un(T_tile), tck, dx=2, dy=0)[:x_w, t_w]
        U_t_pred[:i, j] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=1)[:x_w, t_w]

        # row location
        i = -x_w

        # get tiles
        X_tile = X[i-x_w-1:, j-t_w:j+t_w+1]
        T_tile = T[i-x_w-1:, j-t_w:j+t_w+1]
        U_tile = U_noise[i-x_w-1:, j-t_w:j+t_w+1]

        # interpolate
        tck = bisplrep(X_tile, T_tile, U_tile, kx=x_order, ky=t_order)

        #.predict from OLS model
        U_pred_OLS = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)
        ##construct weight matrix from U_pred
        W = 1/np.abs(U_pred_OLS.flatten()**gamma)
        #Values below thres will have OLS model
        W[np.abs(U_pred_OLS.flatten())<thres] = 1.0
        
        #re-do spline smooth with GLS cost function
        tck = bisplrep(X_tile.flatten(), T_tile.flatten(), U_tile.flatten(),w=W,kx=x_order,ky=t_order)


        # predict
        U_pred[i:, j] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)[x_w+1:, t_w] 
        U_x_pred[i:, j] = bisplev(un(X_tile), un(T_tile), tck, dx=1, dy=0)[x_w+1:, t_w]
        U_xx_pred[i:, j] = bisplev(un(X_tile), un(T_tile), tck, dx=2, dy=0)[x_w+1:, t_w]
        U_t_pred[i:, j] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=1)[x_w+1:, t_w]

    # populate 0,0 corner with spline approximations
    X_tile = X[:2*x_w+1, :2*t_w+1]
    T_tile = T[:2*x_w+1, :2*t_w+1]
    U_tile = U_noise[:2*x_w+1, :2*t_w+1]
    tck = bisplrep(X_tile, T_tile, U_tile, kx=x_order, ky=t_order)
    #.predict from OLS model
    U_pred_OLS = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)
    ##construct weight matrix from U_pred
    W = 1/np.abs(U_pred_OLS.flatten()**gamma)
    #Values below thres will have OLS model
    W[np.abs(U_pred_OLS.flatten())<thres] = 1.0
    #re-do spline smooth with GLS cost function
    tck = bisplrep(X_tile.flatten(), T_tile.flatten(), U_tile.flatten(),w=W,kx=x_order,ky=t_order)
    U_pred[:x_w, :t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)[:x_w, :t_w] 
    U_x_pred[:x_w, :t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=1, dy=0)[:x_w, :t_w]
    U_xx_pred[:x_w, :t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=2, dy=0)[:x_w, :t_w]
    U_t_pred[:x_w, :t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=1)[:x_w, :t_w]

    # populate 0,-1 corner with spline approximations
    X_tile = X[:2*x_w+1, -2*t_w-1:]
    T_tile = T[:2*x_w+1, -2*t_w-1:]
    U_tile = U_noise[:2*x_w+1, -2*t_w-1:]
    tck = bisplrep(X_tile, T_tile, U_tile, kx=x_order, ky=t_order)
    U_pred_OLS = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)
    ##construct weight matrix from U_pred
    W = 1/np.abs(U_pred_OLS.flatten()**gamma)
    #Values below thres will have OLS model
    W[np.abs(U_pred_OLS.flatten())<thres] = 1.0
    #re-do spline smooth with GLS cost function
    tck = bisplrep(X_tile.flatten(), T_tile.flatten(), U_tile.flatten(),w=W,kx=x_order,ky=t_order)
    U_pred[:x_w, -t_w:] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)[:x_w, -t_w:]
    U_x_pred[:x_w, -t_w:] = bisplev(un(X_tile), un(T_tile), tck, dx=1, dy=0)[:x_w, -t_w:]
    U_xx_pred[:x_w, -t_w:] = bisplev(un(X_tile), un(T_tile), tck, dx=2, dy=0)[:x_w, -t_w:]
    U_t_pred[:x_w, -t_w:] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=1)[:x_w, -t_w:]

    # populate -1,0 corner with spline approximations
    X_tile = X[-2*x_w-1:, :2*t_w+1]
    T_tile = T[-2*x_w-1:, :2*t_w+1]
    U_tile = U_noise[-2*x_w-1:, :2*t_w+1]
    tck = bisplrep(X_tile, T_tile, U_tile, kx=x_order, ky=t_order)
    U_pred_OLS = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)
    ##construct weight matrix from U_pred
    W = 1/np.abs(U_pred_OLS.flatten()**gamma)
    #Values below thres will have OLS model
    W[np.abs(U_pred_OLS.flatten())<thres] = 1.0
    #re-do spline smooth with GLS cost function
    tck = bisplrep(X_tile.flatten(), T_tile.flatten(), U_tile.flatten(),w=W,kx=x_order,ky=t_order)
    U_pred[-x_w:, :t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)[-x_w:, :t_w]
    U_x_pred[-x_w:, :t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=1, dy=0)[-x_w:, :t_w]
    U_xx_pred[-x_w:, :t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=2, dy=0)[-x_w:, :t_w]
    U_t_pred[-x_w:, :t_w] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=1)[-x_w:, :t_w]

    # populate -1,-1 corner with spline approximations
    X_tile = X[-2*x_w-1:, -2*t_w-1:]
    T_tile = T[-2*x_w-1:, -2*t_w-1:]
    U_tile = U_noise[-2*x_w-1:, -2*t_w-1:]
    tck = bisplrep(X_tile, T_tile, U_tile, kx=x_order, ky=t_order)
    U_pred_OLS = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)
    ##construct weight matrix from U_pred
    W = 1/np.abs(U_pred_OLS.flatten()**gamma)
    #Values below thres will have OLS model
    W[np.abs(U_pred_OLS.flatten())<thres] = 1.0
    #re-do spline smooth with GLS cost function
    tck = bisplrep(X_tile.flatten(), T_tile.flatten(), U_tile.flatten(),w=W,kx=x_order,ky=t_order)
    U_pred[-x_w:, -t_w:] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=0)[-x_w:, -t_w:]
    U_x_pred[-x_w:, -t_w:] = bisplev(un(X_tile), un(T_tile), tck, dx=1, dy=0)[-x_w:, -t_w:]
    U_xx_pred[-x_w:, -t_w:] = bisplev(un(X_tile), un(T_tile), tck, dx=2, dy=0)[-x_w:, -t_w:]
    U_t_pred[-x_w:, -t_w:] = bisplev(un(X_tile), un(T_tile), tck, dx=0, dy=1)[-x_w:, -t_w:]



    # embed spline interiors inside full approximations
    U_pred[x_w:-x_w, t_w:-t_w] = U_pred_int
    U_t_pred[x_w:-x_w, t_w:-t_w] = U_t_pred_int
    U_x_pred[x_w:-x_w, t_w:-t_w] = U_x_pred_int
    U_xx_pred[x_w:-x_w, t_w:-t_w] = U_xx_pred_int

    # print time
    print 'Elapsed time =', time.time() - t0, 'seconds.'
    print ''

    # bring predictions back to original scale
    U_pred = U_max*U_pred + U_min # un-normalized prediction
    U_x_pred = U_max*U_x_pred     # un-normalized prediction
    U_xx_pred = U_max*U_xx_pred   # un-normalized prediction
    U_t_pred = U_max*U_t_pred     # un-normalized prediction

    # store everything in dictionary
    surface_data = {}
    surface_data['inputs'] = [U_pred, U_x_pred, U_xx_pred]
    surface_data['outputs'] = [U_t_pred]
    surface_data['indep_vars'] = [X, T]
    surface_data['input_names'] = ['U','U_x','U_xx']
    surface_data['output_names'] = ['U_t']
    surface_data['indep_var_names'] = ['X', 'T']

    # save the data
    np.save('data/'+dataset+'_NCV_bisplines', surface_data)


'''
This function computes partial derivatives using global splines

to do: Figure out convergence criteria, once we have a trained
global spline surface, get derivatives (Should be similar to local method)

'''


def predict_global_bisplines(dataset, model_name=None):
    

    import warnings
    warnings.simplefilter("ignore")

    t0 = time.time()

    # load data using SurfNN class
    reload(surface_fitter)
    from surface_fitter import SurfNN

    nn = SurfNN(dataset, model_name)
    x_flat, y_flat, U_min, U_max, U_shape = nn.load_data()

    #suggested smoothness upper bound ,
    #based on scipy's bisplrep documentation
    m = len(x_flat)
    s = m + math.sqrt(2*m)

    # spline parameters
    x_order = 3 # polynomial x degree
    t_order = 3 # polynomial t degree

    # GLS parameters
    gamma = 1.0
    thres = 1e-4
    iterMax = 100

    # rank-1 arrays
    x = x_flat[:,0]
    t = x_flat[:,1]
    u_noise = y_flat[:,0]

    # rank-2 arrays
    X = x.reshape(U_shape)             # space mesh
    T = t.reshape(U_shape)             # time mesh
    U_noise = u_noise.reshape(U_shape) # normalized noisy data

    tpts = len(np.unique(t))
    train_val_ind = np.random.permutation(tpts)
    train_ind = np.sort(train_val_ind[:np.int(.9*tpts)])
    val_ind = np.sort(train_val_ind[np.int(.9*tpts):])

    #subsample into train-val split
    X_train = X[:,train_ind]
    T_train = T[:,train_ind]
    U_train = U_noise[:,train_ind]
    X_val = X[:,val_ind]
    T_val = T[:,val_ind]
    U_val = U_noise[:,val_ind]

    s_convergence = False
    s_list = []

    #INITIALIZE optimization params
    GLS_error_opt = np.inf
    GLS_error_prev = np.inf
    GLS_list=[]
    tck_opt = []

    while s_convergence == False:

        s_list.append(s)

        print "attempting GLS global splines for s= " + str(s)

        try:
            
            GLS_error,tck = GLS_spline_train_val(s,X_train,T_train,U_train,X_val,T_val,U_val,iterMax,x_order=3,t_order=3,gamma=1.0,thres=1e-4)

            print "Error is " + str(GLS_error)

        except:
            GLS_error = np.inf
        
            print "Failed to run. s too large"

        #compare to current best value:
        if GLS_error < GLS_error_opt:
            #if smaller GLS value, then re-set
            GLS_error_opt = GLS_error
            tck_opt = tck
            s_opt = s
            
            
        if GLS_error > GLS_error_prev:
            #if the GLS error increases, and we've done at least 3 finite 
            #s values, then we're beginning to overfit,
            # so time to stop decreasing s
            if np.sum(np.isfinite(s_list)) >= 3:
                s_convergence = True
        else:
            #if not, keep going
            pass
        
        if np.isfinite(GLS_error):
            GLS_error_prev = GLS_error


        GLS_list.append(GLS_error)
        s = s/2.0
    

    #s_opt now gives us an approximate area to sample from. 
    #Now let's perform a slightly more refined search
    s_opt_loc = s_list.index(s_opt)
    print "now re-doing search in ["+str(s_list[s_opt_loc+1])+","+str(s_list[s_opt_loc-1])+"]"

    #look one value below s_opt and one above s_opt
    #and re-do search for s
    s_final_vec = np.linspace(s_list[s_opt_loc+1],s_list[s_opt_loc-1],10)
    GLS_final_vec = np.zeros(s_final_vec.shape)
    GLS_error_opt = np.inf
    tck_opt = []

    for i,s in enumerate(s_final_vec):

        print "attempting GLS global splines for s= " + str(s)

        GLS_error,tck = GLS_spline_train_val(s,X_train,T_train,U_train,X_val,T_val,U_val,iterMax,x_order=3,t_order=3,gamma=1.0,thres=1e-4)
        GLS_final_vec[i] = GLS_error

        #compare to current best value:
        if GLS_error < GLS_error_opt:
            #if smaller GLS value, then re-set
            GLS_error_opt = GLS_error
            tck_opt = tck
            s_opt = s


    with open("data/"+dataset+"_result.txt",'wb') as f:
        f.write("Global NCV splines completed for " + dataset + "at \n")
        f.write(time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())+ "\n")
        f.write("Global s search considered \n")
        f.write(str(s_list)+" \n")
        f.write("which resulted in errors \n")
        f.write(str(GLS_list) + "\n")
        f.write("Optimal s occured at s =" +str(s_opt) + "from local choice of \n")
        f.write(str(s_final_vec)+" \n")
        f.write("With GLS errors \n")
        f.write(str(GLS_final_vec)+" \n")

    U_pred = bisplev(un(X), un(T), tck_opt, dx=0, dy=0)
    U_x_pred = bisplev(un(X), un(T), tck_opt, dx=1, dy=0)
    U_xx_pred = bisplev(un(X), un(T), tck_opt, dx=2, dy=0)
    U_t_pred = bisplev(un(X), un(T), tck_opt, dx=0, dy=1)


    # bring predictions back to original scale
    U_pred = U_max*U_pred + U_min # un-normalized prediction
    U_x_pred = U_max*U_x_pred     # un-normalized prediction
    U_xx_pred = U_max*U_xx_pred   # un-normalized prediction
    U_t_pred = U_max*U_t_pred     # un-normalized prediction

    # store everything in dictionary
    surface_data = {}
    surface_data['inputs'] = [U_pred, U_x_pred, U_xx_pred]
    surface_data['outputs'] = [U_t_pred]
    surface_data['indep_vars'] = [X, T]
    surface_data['input_names'] = ['U','U_x','U_xx']
    surface_data['output_names'] = ['U_t']
    surface_data['indep_var_names'] = ['X', 'T']

    # save the data
    np.save('data/'+dataset+'_global_NCV_bisplines_'+str(x_order), surface_data)



    
'''
This function computes neural network approximations
of partial derivatives from raw data using the learned 
network. Approximations are saved in the data folder.
'''
    
def predict_neural_network(dataset, model_name):
    
    print 'neural network, '+dataset
        
    reload(surface_fitter)
    from surface_fitter import SurfNN
    nn = SurfNN(dataset, model_name)
    x_flat, y_flat, U_min, U_max, U_shape = nn.load_data()

    # rank-1 arrays
    x = x_flat[:,0]
    t = x_flat[:,1]
    u_noise = y_flat[:,0]

    # rank-2 arrays
    X = x.reshape(U_shape)             # space mesh
    T = t.reshape(U_shape)             # time mesh
    U_noise = u_noise.reshape(U_shape) # normalized noisy data

    # get timestamp
    t0 = time.time()

    # make neural net predictions
    surface_data = nn.predict()
    U_pred = surface_data['inputs'][0]    # un-normalized prediction
    U_x_pred = surface_data['inputs'][1]  # un-normalized prediction
    U_xx_pred = surface_data['inputs'][2] # un-normalized prediction
    U_t_pred = surface_data['outputs'][0] # un-normalized prediction

    # print time
    print 'Elapsed time =', time.time() - t0, 'seconds.'
    print ''

    # store everything in dictionary
    surface_data = {}
    surface_data['inputs'] = [U_pred, U_x_pred, U_xx_pred]
    surface_data['outputs'] = [U_t_pred]
    surface_data['indep_vars'] = [X, T]
    surface_data['input_names'] = ['U','U_x','U_xx']
    surface_data['output_names'] = ['U_t']
    surface_data['indep_var_names'] = ['X', 'T']

    # save the data
    np.save('data/'+dataset+'_'+model_name, surface_data)
