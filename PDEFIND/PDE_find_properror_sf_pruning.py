#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os

from PDE_FIND2 import *


# In[9]:


#computational method to consider
comp_str = 'global_NCV_bisplines_3' #options are 'nn','finite_differences','splines', 'NCV_bisplines' ,'global_NCV_bisplines_3'

#mathematical model
model_str = 'diffadv' #options are 'diffadv','fisher','fisher_nonlin'


# In[10]:


#create and format data
skip = 20 #number of initial timepoints to skip
sample_width = 5 #how much to subsample by (timepoints)
normalize = 0 #to normalize data or not during PDE-FIND implementation
deg = 2 # degree of polynomial to use in library
    
#training-validation split
trainPerc = .5      # must be between 0 and 1
valPerc = 1-trainPerc

#number of training-validation splits per data set
reals = 1000

#how to permute the data
shufMethod = 'bins' #options are 'perm' (each point randomly split) , 'noperm' (first 
                    #trainPerc of timepoints given to training data, rest to validation),
                    #'reverse' (last trainperc of timepoints given to training data, rest
                    # to validation), 'bins' (grouping local spatiotemporal points randomly)

#optimization algorithm
algoName = 'Greedy' #options: 'STRidge','Lasso','Greedy'

#where to write result
write_dir = 'pickle_data/'


# In[11]:


#load data directory, true eqn form, and pruning level for different models
if model_str == 'diffadv':
    data_dir = "Data/diffadv/advection_diffusion_"
    deriv_list = ['u_{xx}','u_{x}']
    prune_level = 0.25 
    
elif model_str == 'fisher':
    data_dir = "Data/fisher/fisher_"
    deriv_list = ['u_{xx}','u','u^2']
    prune_level = 0.25
    
elif model_str == 'fisher_nonlin':
    data_dir = "Data/nonlin_fisher/fisher_nonlin_"
    deriv_list = ['uu_{xx}','u_{x}^2','u','u^2']
    prune_level = 0.05
    
#data files (based on different noise levels) to consider
data_files = ['00_' + comp_str,'01_' + comp_str,'05_' + comp_str,'10_' + comp_str,'25_' + comp_str,'50_' + comp_str]


# In[12]:


for d in data_files:

    #filename to save at
    filename = write_dir + algoName + '_' + d + '_' + shufMethod + '_'+model_str+'_prune_deg_' +str(deg)+ '.npz'
    
    #list of xi estimates from PDE-FIND with pruning
    xi_list = []
    #list of xi estimates from PDE-FIND (no pruning)
    xi_list_no_prune = []
    #list of selected hyperparameters from each simulation
    hparams_list = []
    #validation score
    val_score_list = []
    #list of TPR scores for each realization
    TP_score_list = []

    #load in file
    mat = np.load(data_dir + d + '.npy').item()
    #create indep. variable grids, ut, theta
    t_samp,x_samp,ut,theta,description = diffadv_theta_construct_sf(mat,skip,sample_width,deg)
    
    #loop through reals
    for real in np.arange(reals):
    
        #split data into train and validation data
        # ptrain, pval are indices pertaining to train / validation data : 
        # i.e., ut[ptrain] = utTrain
        utTrain,thetaTrain,ptrain,utVal,thetaVal,pval,utTest,thetaTest,ptest = data_shuf(ut,
             theta,shufMethod,trainPerc,valPerc,len(x_samp),len(t_samp),stack=1)

        #perform training and validation for given data
        xi, hparams, val_score, TP_score = run_PDE_Find_train_val(thetaTrain, utTrain, thetaVal, utVal, algoName,description,deriv_list)
                
        print "initial equation is " + print_pde(xi,description)
        print "initial TPR score is " + str(TP_TPFPFN(xi,description,deriv_list,0))
        
        #implement pruning if xi has more than 1 nonzero entry
        if len(xi[xi!=0]) > 1:
            #perform pruning methodology
            xi_new, description_new, thetaTrain_new, thetaVal_new = PDE_FIND_prune_lstsq(xi,utTrain,
                                         utVal,thetaTrain,thetaVal,description,val_score,prune_level)
            #obtain final validation score
            val_score = run_PDE_Find_Test(thetaVal,utVal,xi_new)
        else:
            xi_new = xi
            
        print "updated equation is " + print_pde(xi_new,description)
        print "Final TP score is " + str(TP_TPFPFN(xi_new,description,deriv_list,0))
        
        #add new info to lists
        xi_list.append(xi_new)
        xi_list_no_prune.append(xi)
        hparams_list.append(hparams)
        val_score_list.append(val_score)
        TP_score_list.append(TP_TPFPFN(xi_new,description,deriv_list,0))
           
        #save
        #np.savez(filename,xi_list = xi_list,xi_list_no_prune=xi_list_no_prune,hparams_list=hparams_list,val_score_list=val_score_list,TP_score_list=TP_score_list,
         #       description=description,deriv_list=deriv_list)
        


# In[ ]:




