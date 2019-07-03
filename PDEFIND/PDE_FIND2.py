import numpy as np
from numpy import linalg as LA
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
from scipy.sparse import dia_matrix
import itertools
import operator, pdb
import math
from tqdm import tqdm

"""
A few functions used in PDE-FIND

Samuel Rudy.  2016

Updated February 13, 2019 by John Nardini
for the study "Learning partial differential
equations for biological transport models 
from noisy spatiotemporal data"
"""


def build_Theta(data, derivatives, derivatives_description, P, data_description = None):
    """
    builds a matrix with columns representing polynoimials up to degree P of all variables

    This is used when we subsample and take all the derivatives point by point or if there is an 
    extra input (Q in the paper) to put in.

    input:
        data: column 0 is U, and columns 1:end are Q
        derivatives: a bunch of derivatives of U and maybe Q, should start with a column of ones
        derivatives_description: description of what derivatives have been passed in
        P: max power of polynomial function of U to be included in Theta

    returns:
        Theta = Theta(U,Q)
        descr = description of what all the columns in Theta are
    """
    
    n,d = data.shape
    m,d2 = derivatives.shape
    if n != m: raise Exception('dimension error')
    if data_description is not None: 
        if len(data_description) != d: raise Exception('data descrption error')
    
    # Create a list of all polynomials in d variables up to degree P
    rhs_functions = {}
    f = lambda x, y : np.prod(np.power(list(x), list(y)))
    powers = []            
    for p in range(1,P+1):
            size = d + p - 1
            for indices in itertools.combinations(range(size), d-1):
                starts = [0] + [index+1 for index in indices]
                stops = indices + (size,)
                powers.append(tuple(map(operator.sub, stops, starts)))
    for power in powers: rhs_functions[power] = [lambda x, y = power: f(x,y), power]

    # First column of Theta is just ones.
    Theta = np.ones((n,1), dtype=np.complex64)
    descr = ['']
    
    # Add the derivaitves onto Theta
    for D in range(1,derivatives.shape[1]):
        Theta = np.hstack([Theta, derivatives[:,D].reshape(n,1)])
        descr.append(derivatives_description[D])
        
    # Add on derivatives times polynomials
    for D in range(derivatives.shape[1]):
        for k in rhs_functions.keys():
            func = rhs_functions[k][0]
            new_column = np.zeros((n,1), dtype=np.complex64)
            for i in range(n):
                new_column[i] = func(data[i,:])*derivatives[i,D]
            Theta = np.hstack([Theta, new_column])
            if data_description is None: descr.append(str(rhs_functions[k][1]) + derivatives_description[D])
            else:
                function_description = ''
                for j in range(d):
                    if rhs_functions[k][1][j] != 0:
                        if rhs_functions[k][1][j] == 1:
                            function_description = function_description + data_description[j]
                        else:
                            function_description = function_description + data_description[j] + '^' + str(rhs_functions[k][1][j])
                descr.append(function_description + derivatives_description[D])

    ## (Added 9-18-2018 by JTN) : Add on derivatives times derivatives
    for D1 in range(derivatives.shape[1]):
        for D2 in range(1,D1+1):
            Theta = np.hstack([Theta, np.multiply(derivatives[:,D1].reshape(n,1),derivatives[:,D2].reshape(n,1))])
            if D1==D2:
                descr.append(derivatives_description[D1] + '^2')
            else:
                descr.append(derivatives_description[D1] + derivatives_description[D2])
                
    return Theta, descr



    
def FoBaGreedy(X, y, epsilon = 0.1, maxit_f = 500, maxit_b = 5, backwards_freq = 5):
    """
    Forward-Backward greedy algorithm for sparse regression.

    See Zhang, Tom. 'Adaptive Forward-Backward Greedy Algorithm for Sparse Learning with Linear
    Models', NIPS, 2008
    """

    n,d = X.shape
    F = {}
    F[0] = set()
    w = {}
    w[0] = np.zeros((d,1), dtype=np.complex64)
    k = 0
    delta = {}

    for forward_iter in range(maxit_f):

        k = k+1

        # forward step
        non_zeros = np.where(w[k-1] == 0)[0]
        err_after_addition = []
        residual = y - X.dot(w[k-1])
        for i in range(len(non_zeros)):
            alpha = X[:,i].T.dot(residual)/np.linalg.norm(X[:,i])**2
            w_added = np.copy(w[k-1])
            w_added[i] = alpha
            err_after_addition.append(np.linalg.norm(X.dot(w_added)-y))
        i = np.argmin(err_after_addition)
        
        F[k] = F[k-1].union({i})
        w[k] = np.zeros((d,1), dtype=np.complex64)
        w[k][list(F[k])] = np.linalg.lstsq(X[:, list(F[k])], y)[0]

        # check for break condition
        delta[k] = np.linalg.norm(X.dot(w[k-1]) - y) - np.linalg.norm(X.dot(w[k]) - y)
        if delta[k] < epsilon: return w[k-1]

        # backward step, do once every few forward steps
        if forward_iter % backwards_freq == 0 and forward_iter > 0:

            for backward_iter in range(maxit_b):

                non_zeros = np.where(w[k] != 0)
                err_after_simplification = []
                for j in range(len(non_zeros)):
                    w_simple = np.copy(w[k])
                    w_simple[j] = 0
                    err_after_simplification.append(np.linalg.norm(X.dot(w_simple) - y))
                j = np.argmin(err_after_simplification)
                w_simple = np.copy(w[k])
                w_simple[j] = 0

                # check for break condition on backward step
                delta_p = err_after_simplification[j] - np.linalg.norm(X.dot(w[k]) - y)
                if delta_p > 0.5*delta[k]: break

                k = k-1;
                F[k] = F[k+1].difference({j})
                w[k] = np.zeros((d,1), dtype=np.complex64)
                w[k][list(F[k])] = np.linalg.lstsq(X[:, list(F[k])], y)[0]

    return w[k] 
    

def STRidge(X0, y, lam, maxit, tol, normalize = 2, print_results = False):
    """
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse 
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

    This assumes y is only one column
    """

    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0
    
    # Get the standard ridge esitmate
    if lam != 0: w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d),X.T.dot(y),rcond=None)[0]
    else: w = np.linalg.lstsq(X,y)[0]
    num_relevant = d
    biginds = np.where( abs(w) > tol)[0]
    
    # Threshold and continue
    for j in range(maxit):

        # Figure out which items to cut out
        smallinds = np.where( abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]
            
        # If nothing changes then stop
        if num_relevant == len(new_biginds): break
        else: num_relevant = len(new_biginds)
            
        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0: 
                #if print_results: print "Tolerance too high - all coefficients set below tolerance"
                return w
            else: break
        biginds = new_biginds
        
        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0: w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y),rcond=None)[0]
        else: w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]

    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=None)[0]
    
    if normalize != 0: return np.multiply(Mreg,w)
    else: return w

    
def Lasso(X0, Y, lam, w = np.array([0]), maxit = 100, normalize = 2):
    """
    Uses accelerated proximal gradient (FISTA) to solve Lasso
    argmin (1/2)*||Xw-Y||_2^2 + lam||w||_1
    """
    
    # Obtain size of X
    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    Y = Y.reshape(n,1)
    
    # Create w if none is given
    if w.size != d:
        w = np.zeros((d,1), dtype=np.complex64)
    w_old = np.zeros((d,1), dtype=np.complex64)
        
    # Initialize a few other parameters
    converge = 0
    objective = np.zeros((maxit,1))
    
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0

    # Lipschitz constant of gradient of smooth part of loss function
    L = np.linalg.norm(X.T.dot(X),2)
    
    # Now loop until converged or max iterations
    for iters in range(0, maxit):
         
        # Update w
        z = w + iters/float(iters+1)*(w - w_old)
        w_old = w
        z = z - X.T.dot(X.dot(z)-Y)/L
        for j in range(d): w[j] = np.multiply(np.sign(z[j]), np.max([abs(z[j])-lam/L,0]))

        # Could put in some sort of break condition based on convergence here.
    
    # Now that we have the sparsity pattern, used least squares.
    biginds = np.where(w != 0)[0]
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],Y)[0]

    # Finally, reverse the regularization so as to be able to use with raw data
    if normalize != 0: return np.multiply(Mreg,w)
    else: return w


def print_pde(w, rhs_description, ut = 'u_t',n=5,imag_print = 0):
    pde = ut + ' = '
    first = True
    for i in range(len(w)):
        if w[i] != 0:
            if not first:
                pde = pde + ' + '
            if imag_print == 0:
                pde = pde + "(%05f)" % (w[i].real) + rhs_description[i] + "\n   "
            else:
                pde = pde + "(%0"+str(n)+"f %+05fi)" % (w[i].real, w[i].imag) + rhs_description[i] + "\n   "
            first = False
    return(pde)
    
def print_pde_table(w, rhs_description, ut = 'u_t',n=3):
    pde = ut + ' = '
    first = True
    #pdb.set_trace()
    for i in range(len(w)):
        if w[i] != 0:
            if not first:
                pde = pde + ' + '        
            pde = pde + "%05f" % (w[i].real) + rhs_description[i]
            first = False
    if pde == ut + ' = ':
        pde = pde + "0"
    return(pde)

    
#functions added by JTN

#construct Theta for a given mat file
def diffadv_theta_construct_sf(mat,skip,sample_width,deg,flag='npy',normalize=0):
    
    #.npy input
    if flag == 'npy':
        U = mat['inputs'][0].T # [t x x]
        Ux = mat['inputs'][1].T # [t x x]
        Uxx = mat['inputs'][2].T # [t x x]
        Ut = mat['outputs'][0].T # [t x x]
        
    #.mat input
    elif flag == 'mat':
        U = mat['U'] # [t x x]
        Ut = mat['U_t'] # [t x x]
        Ux = mat['U_x'] # [t x x]
        Uxx = mat['U_xx'] # [t x x]
        
    #normalization implement
    if normalize == 1:
        umax = np.max(np.abs(U)) 
        U = U/float(umax)
        utmax = np.max(np.abs(Ut)) 
        Ut = Ut/float(utmax)
        uxmax = np.max(np.abs(Ux)) 
        Ux = Ux/float(uxmax)
        uxxmax = np.max(np.abs(Uxx)) 
        Uxx = Uxx/float(uxxmax)  
    else:
        umax = 1
        utmax = 1
        uxmax = 1
        uxxmax = 1
        
    #subsample
    U = U[skip:,:]
    U = U[0::sample_width,:]
    Ut = Ut[skip:,:]
    Ut = Ut[0::sample_width,:]
    Ux = Ux[skip:,:]
    Ux = Ux[0::sample_width,:]
    Uxx = Uxx[skip:,:]
    Uxx = Uxx[0::sample_width,:]
    
    #independent variables
    if flag == 'npy':
        x=mat['indep_vars'][0][:,0][:,np.newaxis]
        t=mat['indep_vars'][1][0,:][:,np.newaxis]
        t=t[skip:,:]
        t=t[0::sample_width,:]
    elif flag == 'mat':
        x=mat['x']
        t=mat['t']
        t=t[skip:,]
        t=t[0::sample_width,]

    #number of data points
    num_points = U.size
    
    #make sure the size of each dependent variable is consistent.
    if U.size != Ut.size:
        print('Something wrong with data size')
    elif Ut.size != Ux.size:
        print('Something wrong with data size')
    elif Ux.size != Uxx.size:
        print('Something wrong with data size')
    
    #re-shape into vectors
    U = np.reshape(U,(num_points,1), order='C')
    Ut = np.reshape(Ut,(num_points,1), order='C')
    Ux = np.reshape(Ux,(num_points,1), order='C')
    Uxx = np.reshape(Uxx,(num_points,1), order='C')
        
    # Form a huge matrix of all terms in the library.
    X_data = U
    X_ders = np.hstack([np.ones((num_points,1)), Ux, Uxx,])
    X_ders_descr = ['','u_{x}','u_{xx}']
    X, description = build_Theta(X_data, X_ders, X_ders_descr, deg, data_description = ['u'])
    
    return t,x,Ut,X,description

#shuffle the data in train, val, test data
def data_shuf(Ut,R,shufMethod,trainPerc,valPerc,xn,tn,stack=1,xbin=5,tbin=5):
    
    #number of total points
    p_length = len(Ut)//stack
    
    #permute all points randomly
    if shufMethod == 'perm':
        p = np.random.permutation(p_length)
        ptrain = p[:int(p_length*trainPerc)]
        pval = p[int(p_length*trainPerc):int(p_length*(trainPerc+valPerc))]
        ptest = p[int(p_length*(trainPerc+valPerc)):]

    #do not permute data at all
    elif shufMethod == 'noperm':
        p = np.arange(p_length)
        ptrain = p[:int(p_length*trainPerc)]
        pval = p[int(p_length*trainPerc):int(p_length*(trainPerc+valPerc))]
        ptest = p[int(p_length*(trainPerc+valPerc)):]
        
    #do not permute data, but take last time points first
    elif shufMethod == 'reverse':
        p = np.squeeze(np.fliplr(np.atleast_2d(np.arange(p_length))))
        ptrain = p[:int(p_length*trainPerc)]
        pval = p[int(p_length*trainPerc):int(p_length*(trainPerc+valPerc))]
        ptest = p[int(p_length*(trainPerc+valPerc)):]
        
    #split data into (tbin)x(xbin) tiles of adjacent spatiotemporal data
    elif shufMethod == 'bins':
        ptrain,pval,ptest = binshuffle(xbin,tbin,xn,tn,trainPerc,valPerc)
        p = np.concatenate((ptrain,pval,ptest))
                
    #number of time points in each data split
    ntrain = len(ptrain)
    nval = len(pval)
    ntest = len(ptest)
    
    #stack entries (not required when working with only 1 data set)!
    for i in np.arange(stack-1):
        ptrain = np.concatenate((ptrain,ptrain[:ntrain]+(i+1)*p_length))
        pval = np.concatenate((pval,pval[:nval]+(i+1)*p_length))
        ptest = np.concatenate((ptest,ptest[:ntest]+(i+1)*p_length))
    
    #split data into train, val, test
    UtTrain = Ut[ptrain,:]
    RTrain = R[ptrain,:]
    
    UtVal = Ut[pval,:]
    RVal = R[pval,:]
    
    UtTest = Ut[ptest,:]
    RTest = R[ptest,:]
    
    
    return UtTrain, RTrain, ptrain, UtVal, RVal, pval, UtTest, RTest, ptest 

def binshuffle (xbin,tbin,xn,tn,trainPerc=0.6,valPerc=0.2):
    
    #split the indices of U (assumed pre-vectorized) into separate bins of size (tbin,xbin)
    #for training, validating, and testing. Note that the first dimension of U is assumed to be t
    
    #Determine number of t bins, xbins, and then all bins
    Nt = np.ceil(tn/tbin) # number t bins
    Nx = np.ceil(xn/xbin) # number x bins
    Nbins = int(Nt*Nx)    # number total bins
    #permute these bins
    pbin = np.random.permutation(Nbins)
    
    #split into train, val, and test bins
    train_pbin = pbin[:int(Nbins*trainPerc)]
    val_pbin = pbin[int(Nbins*trainPerc):int(Nbins*(trainPerc+valPerc))]
    test_pbin = pbin[int(Nbins*(trainPerc+valPerc)):]
    
    #initialize indices
    ind = np.arange(xn*tn)
    indModtn = np.mod(ind,tn)
    
    #training indices
    ptrain = []
    for p in train_pbin:
        #t bin index 
        pmodt = np.mod(p,Nt)
        #x bin index
        pmodx = np.floor(p/Nt)
        
        #from the bin indices, find the indices in U
        ind_loc = ([pmodt*tbin <= indModtn , indModtn < (pmodt+1)*tbin ,
                   pmodx*tn*xbin <= ind, ind < (pmodx+1)*tn*xbin])
        #logical to arrays
        new_ind = np.where(np.all(ind_loc,axis=0))
        #add to train indices
        ptrain = np.concatenate((ptrain,new_ind[0]),axis=0)

    #validation indices
    pval = []
    for p in val_pbin:
        #t bin index
        pmodt = np.mod(p,Nt)
        #x bin index
        pmodx = np.floor(p/Nt)
        
        #from the bin indices, find the indices in U
        ind_loc = ([pmodt*tbin <= indModtn , indModtn < (pmodt+1)*tbin ,
                   pmodx*tn*xbin <= ind, ind < (pmodx+1)*tn*xbin])
                      
        #logical to arrays
        new_ind = np.where(np.all(ind_loc,axis=0))
        
        #add to validation indices
        pval = np.concatenate((pval,new_ind[0]),axis=0)
        
    #test indices
    ptest = []
    for p in test_pbin:
        #t bin index
        pmodt = np.mod(p,Nt)
        #x bin index
        pmodx = np.floor(p/Nt)
        
        #from the bin indices, find the indices in U
        ind_loc = ([pmodt*tbin <= indModtn , indModtn < (pmodt+1)*tbin ,
                   pmodx*tn*xbin <= ind, ind < (pmodx+1)*tn*xbin])
           
        #logical to arrays
        new_ind = np.where(np.all(ind_loc,axis=0))
        
        #add to test indices
        ptest = np.concatenate((ptest,new_ind[0]),axis=0)
    
    #convert to ints
    ptrain = ptrain.astype(int)
    pval = pval.astype(int)
    if ptest!=[]: ptest = ptest.astype(int)

    return ptrain, pval, ptest

#Find the value of w that best fits the train data and then perform validation over the validation data
def run_PDE_Find_train_val (RTrain,UtTrain,RVal,utVal,algoName,description,deriv_list):
        
    #use STRidge method
    if algoName == 'STRidge':

        #hyperparameters.
        lambda_vec = np.hstack([0,10**np.linspace(-3,2,20)])
        dtol_vec =  np.hstack([0,10**np.linspace(-3,2,20)])
        
        # convert to mesh
        [l_mesh,dtol_mesh] = np.meshgrid(lambda_vec,dtol_vec)
        # create ordered pairs in X_mesh
        l_mesh = l_mesh.reshape(-1)[:,np.newaxis]
        dtol_mesh = dtol_mesh.reshape(-1)[:,np.newaxis]
        X_mesh = np.concatenate([l_mesh,dtol_mesh],axis=1)

        #initialize vector of validation and TPR scores
        val_score = np.zeros(len(X_mesh))
        TP_FN_score = np.zeros(len(X_mesh))
        
        for i,hparams in enumerate(tqdm(X_mesh)):
            #sparse regression
            xi = STRidge(RTrain,UtTrain,hparams[0], 1000, hparams[1])
            #validation
            val_score[i] = run_PDE_Find_Test(RVal,utVal,xi)
            #TPR
            TP_FN_score[i] = TP_TPFPFN(xi,description,deriv_list,0)

        #find optimal hyper parameter
        hparams_opt = X_mesh[np.argmin(val_score),]
        #re-find best estimate
        xi_best = STRidge(RTrain,UtTrain,hparams_opt[0], 1000, hparams_opt[1])
        hparams_opt = X_mesh

    elif algoName == 'Greedy':
        
        #hyperparameters
        lambda_vec = np.hstack([0,10**np.linspace(-3,2,50)])
        
        #initialize validation, TPR vectors
        val_score = np.zeros(len(lambda_vec))
        TP_FN_score = np.zeros(len(lambda_vec))
        
        for j,l in enumerate(tqdm(lambda_vec)):
            #sparse regression
            xi = FoBaGreedy(RTrain, UtTrain,l)
            #validation
            val_score[j] = run_PDE_Find_Test(RVal,utVal,xi)
            #TPR
            TP_FN_score[j] = TP_TPFPFN(xi,description,deriv_list,0)
            
        #find optimal hyperparameter
        hparams_opt = lambda_vec[np.argmin(val_score)]
        #re-find best estimate
        xi_best = FoBaGreedy(RTrain, UtTrain,hparams_opt)
        
        
    elif algoName == 'Lasso':
        
        lambda_vec = np.hstack([0,10**np.linspace(-2,3,50)])
        
        val_score = np.zeros(len(lambda_vec))
        TP_FN_score = np.zeros(len(lambda_vec))
        for j,l in enumerate(lambda_vec):
            #sparse regression
            xi = Lasso(RTrain, UtTrain, l, normalize = 1,maxit = 100)
            #validation
            val_score[j] = run_PDE_Find_Test(RVal,utVal,xi)
            #TPR
            TP_FN_score[j] = TP_TPFPFN(xi,description,deriv_list,0)
            
        #find optimal hyperparameter
        hparams_opt = lambda_vec[np.argmin(val_score)]
        #re-find best estimate
        xi_best = Lasso(RTrain, UtTrain, hparams_opt, normalize = 1,maxit = 100)
        
    return xi_best, hparams_opt, np.min(val_score), TP_FN_score


#Test  a value of w using MSE
def run_PDE_Find_Test (RTest,UtTest,w):
   
    #length of vector
    test_length = len(RTest)
    #Compute test score
    score = np.linalg.norm(UtTest - np.matmul(RTest,w))/test_length
    
    return score

#compute the TPR for a given w estimate
def TP_TPFPFN (w,rhs_des,deriv_list,thres=1e-4):
    
    #zero gets a score of 0
    if len(w) == 0:
        return 0.0
    
    #initialize   
    TP = 0
    FP = 0
    FN = 0

    #it's possible that an entry in deriv_list may not be in rhs_des
    #so check that each entry is in deriv_list. If not, ignore it but
    #add one to FN
    deriv_list_c = deriv_list[:]
    remove_from_deriv_list = []
    #account for any deriv_list entries not in rhs_des
    for i in deriv_list_c:
        if i not in rhs_des:
            FN+=1
            remove_from_deriv_list.append(deriv_list_c.index(i))            
    #now remove terms (in reverse direction to not accidentally remove other terms)
    for i in sorted(remove_from_deriv_list,reverse=True):
        del deriv_list_c[i]
    
    
    #Now find locations of true terms in w
    deriv_inds = [rhs_des.index(j) for j in deriv_list_c]
    
    #update TP, FN    
    for i in deriv_inds:
        if np.abs(w[i]) > thres:
            TP += 1
        else:
            FN += 1
    
    #remove deriv_inds from consideration now that we've covered them
    all_ind = np.arange(len(w))
    extra_ind = np.delete(all_ind,deriv_inds)
    
    #loop through remaining terms, penalize if not zero
    for i in extra_ind:
        if np.abs(w[i]) > thres:
            FP += 1
            
    return(float(TP)/float(TP + FP + FN))


#perform pruning step after PDE-FIND
def PDE_FIND_prune_lstsq(xi,utTrain,utVal,thetaTrain,thetaVal,description,val_score_init,prune_level):
    
    #reduce library to nonzero terms in xi
    xi_tilde = xi[xi!=0]
    thetaTrain_tilde = thetaTrain[:,np.squeeze(xi!=0)]
    thetaVal_tilde = thetaVal[:,np.squeeze(xi!=0)]
    description_tilde = [description[i] for i in np.where(np.squeeze(xi!=0))[0]]

    #loop through each remaining entry and see how much the validation score changes when this entry is set to zero
    val_score_vec = np.zeros(np.squeeze(xi_tilde.shape))
    for i in np.argsort(np.abs(xi_tilde)):
        #copy theta_tilde without ith column (further reduced library)
        thetaTrain_hat = np.delete(thetaTrain_tilde,i,1)
        thetaVal_hat = np.delete(thetaVal_tilde,i,1)
        
        #nonsparse regression
        xi_hat = np.linalg.lstsq(thetaTrain_hat,utTrain)[0]
        #get new validation score
        val_score_new = run_PDE_Find_Test(thetaVal_hat,utVal,xi_hat)
        val_score_vec[i] = val_score_new
    
    #indices of params to keep (keep if validation score goes up more than (1+alpha)*100%)
    keep_ind = np.squeeze(val_score_vec >= (1+prune_level)*val_score_init)
    
    #final theta matrices
    thetaTrain_tilde = thetaTrain_tilde[:,keep_ind]
    thetaVal_tilde = thetaVal_tilde[:,keep_ind]
    #final training on final library
    if thetaTrain_tilde.shape[1] > 0:
        xi_tilde = np.linalg.lstsq(thetaTrain_tilde,utTrain)[0]#xi_tilde[keep_ind]
    else:
        xi_tilde = []
    #final description
    description_tilde = [description_tilde[j] for j in np.where(keep_ind)[0]]
    
    # re-convert xi_tilde to same length as initial description
    xi_tilde = xi_convert_full(xi_tilde,description_tilde,description)
       
    return xi_tilde, description_tilde, thetaTrain_tilde, thetaVal_tilde


#convert xi from partial library to full library
def xi_convert_full(xi,desc,desc_full):
    xi_full = np.zeros((len(desc_full),1))
    
    for i,x in enumerate(xi):
        #where is this term located in full description
        desc_ind = desc_full.index(desc[i])
        #now put it in the full vector
        xi_full[desc_ind] = xi[i]
        
    return xi_full

#go from decimal number to its binary N-dimensiona vector of 0's and 1's
def trans(x,N):
    y=np.copy(x)
    if y == 0: return[0]
    bit = []
    for i in np.arange(N):
        bit.append(y % 2)
        y >>= 1
    return np.atleast_2d(np.asarray(bit[::-1]))

#go from N-dimnsional vector of 0's and 1's to corresponding decimal number
def trans_rev(x):
    n = len(x)-1
    dec = 0
    for i in np.arange(n+1):
        dec = dec + x[i]*2**(n-i)
    return dec

#find most common entry from list
def most_common(lst):
    return max(lst, key=lst.count)