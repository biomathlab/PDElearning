import numpy as np
from PDE_FIND2 import *
import statistics, os, pdb

write_dir = 'pickle_data/'
#math model
#options are 'diffadv','fisher','fisher_nonlin'
model_str_list = ['fisher','fisher_nonlin']#['diffadv']#,'fisher','fisher_nonlin']

#noise levels that were considered
data_files_1 = ['00_','01_','05_','10_','25_','50_']
data_file_strings = ['0.0','0.01','0.05','0.10','0.25','0.50']

#other specs of implementation
algoName = 'Greedy'
shufMethod = 'bins'
deg = 2

#methods used for computation, as well as their abbreviations
methods = ['finite_differences','splines','NCV_bisplines',
           'global_NCV_bisplines_3','nn']
method_strings = ['FD','LCVSP','LNCVSP','GNCVSP','ANN']

for model_str in model_str_list:
    #load in true equation form
    if model_str == 'diffadv':
        deriv_list = ['u_{xx}','u_{x}']
        true_params = np.array([.01,-0.8])
    elif model_str == 'fisher':
        deriv_list = ['u_{xx}','u','u^2']
        true_params = np.array([.02,10,-10])
    elif model_str == 'fisher_nonlin':
        deriv_list = ['uu_{xx}','u_{x}^2','u','u^2']
        true_params = np.array([.02,.02,10,-10])

    #table heading
    print '\\begin{tabular}{|c|c|c|}'
    print '    \\hline'
    print '      &  & \\textbf{True Equation} \\\\ '
    print '    \\hline'
    print '      &  & $' + print_pde_table(true_params,deriv_list) + '$  \\\\  ' 
    print '    \\hline'
    print '    $\\boldsymbol{\\sigma}$ & \\textbf{Method} & \\textbf{Learned Equation} \\\\ '
    print '    \\hline'


    
    #vote for the equation form
    xi_vote = []
    #vote for each param individually
    xi_vote_params = []

    
    for j,m in enumerate(methods):
        for k,d in enumerate(data_files_1):   
        

            #load in data
            if 'NCV_bisplines' not in m:
                filename = write_dir + algoName + '_' +d+m+ '_' + shufMethod + '_'+model_str+'_prune_tv_5050_1_21_deg_' +str(deg)+ '.npz'
            else:
                filename = write_dir + algoName + '_' +d+m+ '_' + shufMethod + '_'+model_str+'_prune_deg_' +str(deg)+ '.npz'

            if os.path.isfile(filename):

                #load in data
                data = np.load(filename)
                
                # Determine the most-common equation
                xi_vote_tmp = []
                for i in range(len(data['xi_list'])):
                    #convert nonzero entries of xi to a decimal number, signifying which equation
                    #form we are voting for
                    xi_vote_tmp.append(trans_rev((data['xi_list'][i] != 0)*1))
                #append the most common equation form
                xi_vote.append(most_common(xi_vote_tmp))

                matrix_vote_initialized = False
                #now generate parameter estimates from the most common equation form
                #by looping back through each xi value. If it matches the most commonly
                #chosen xi form, then we add its parameter estimates to the final list
                #
                # each column of A corresponds to one such xi estimate
                # each row of A corresponds to different terms in A
                for i in range(len(data['xi_list'])):
                    xi_full = data['xi_list'][i]
                    #if current xi matches most common xi
                    if xi_vote[-1]==trans_rev(xi_full != 0)*1:
                        if not matrix_vote_initialized:
                            A = xi_full
                            matrix_vote_initialized = True
                        else:
                            A = np.hstack((A,xi_full))

                print ('    '+ data_file_strings[k] +'  &   ' + method_strings[j] + '   &   $' + print_pde_table(np.mean(A,axis=1),data['description']) + '$ \\\\ ')
                print '    \\hline'
                

    print '\\end{tabular}'
    print ''


