import numpy as np
from scipy.interpolate import splrep, splev, bisplrep, bisplev
from numpy import unique as un
import pdb

def GLS_spline_train_val(s,X_train,T_train,U_train,X_val,T_val,U_val,iterMax,x_order=3,t_order=3,gamma=1.0,thres=1e-4):


	#try:
	#fit spline using OLS
	tck = bisplrep(X_train,T_train,U_train,kx=x_order, ky=t_order,s=s)

	#now we will get the knots by fitting splines (with GLS model) 
	#until the number of knots has converged

	knot_convergence = False
	knot_count_list = []
	knot_iter_count = 0

	#wait for GLS modeling to converge to same number of knots:
	while(knot_convergence==False):
	    
	    #perform GLS splines
	    U_pred_train = bisplev(un(X_train), un(T_train), tck, dx=0, dy=0)
	    #weight matrix
	    W_train = 1/np.abs(U_pred_train.flatten()**gamma)
	    #Values below thres will have OLS model
	    W_train[np.abs(U_pred_train.flatten())<thres] = 1.0
	    #re-do spline smooth with GLS cost function, let it find the knots
	    tck = bisplrep(X_train.flatten(), T_train.flatten(), U_train.flatten(),w=W_train,kx=x_order,ky=t_order,s=s)
	    

	    #count number of x,y knots
	    knot_count_list.append(len(tck[0])+len(tck[1]))
	    knot_iter_count += 1

	    #if the number of knots has remained the same (plus/minus 1) for 5 
	    # iterations, then we say it has converged

	    if len(knot_count_list)>5:
	        last_knots_equal = [knot_entry in np.arange(knot_count_list[-1]-1,knot_count_list[-1]+2) for knot_entry in knot_count_list[-5:]]
	        if all(last_knots_equal):
	            knot_convergence = True
	            print "knot locations converged"

	    #but don't let it run too long
	    if knot_iter_count > iterMax:
	        print "Exiting knot location convergence due to too many iterations"
	        break

	#now that the knots have converged, let's find the spline coefficients
	count = 0
	converge_critera = False

	#knot locations
	xKnot = tck[0]
	yKnot = tck[1]
	#current coeff values
	cOld = tck[2]


	#perform GLS splines (fixed knot locations) until the coeffs
	#appear to converge in inf-norm
	while(converge_critera==False):

		#predict u
		U_pred_train = bisplev(un(X_train), un(T_train), tck, dx=0, dy=0)
		#weight matrix
		W_train = 1/np.abs(U_pred_train.flatten()**gamma)
		#Values below thres will have OLS model
		W_train[np.abs(U_pred_train.flatten())<thres] = 1.0
		#re-do spline smooth with GLS cost function and fixed knot locations
		tck = bisplrep(X_train.flatten(), T_train.flatten(), U_train.flatten(),w=W_train,kx=x_order,ky=t_order,task=-1,tx=xKnot,ty=yKnot,s=s)


		#only consider params that were larger than 1e-1 for tolerance check
		param_tol_ind = np.abs(cOld) > 1e-1
		param_tol_ratio = (cOld - tck[2])/cOld
		if np.sum(param_tol_ind) == 0:
			print "Exiting, as coeffs are converging to zero"
			break
		else:
			param_tol = np.linalg.norm(param_tol_ratio[param_tol_ind],np.inf)
		
		#if the weights don't change much, we have converged
		if param_tol < 1e-2:

			print "Spline parameters appear to have converged, now exiting"
			converge_critera = True

		#otherwise, keep going
		else:

			cOld = tck[2]
			count += 1
			#print "completed " +str(count)+ "iterations of GLS estimation" 

		#if we exceed itermax, we can leave
		if count >= iterMax:
			print "Exiting due to exceeding iterMax"
			break


	#predict U for the validation data from the final spline params
	U_pred_val = bisplev(un(X_val), un(T_val), tck, dx=0, dy=0)

	#create weight matrix from U_pred
	W_val = np.abs(U_pred_val.flatten()**gamma)
	#Values below thres will have OLS model
	W_val[np.abs(U_pred_val.flatten()<thres)] = 1.0

	#compute error
	GLS_error = np.linalg.norm((U_pred_val - U_val).flatten()/W_val)


	#except:
	#	GLS_error = np.inf
	#
	#	print "Failed to run. s too large"

	return GLS_error, tck