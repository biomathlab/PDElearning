This folder contains the code to perform the PDE learning aspect of our study.

PDE_find_properror_sf_pruning.ipynb - ipython notebook to run PDE-FIND implementations. In the second cell, the variable "comp_str"specifies which denoising strategy one wishes to use (nn, finite_differences, splines (meaning cubic bisplines), NCV_busplines (meaning local cubic bisplines with a GLS error model), or global_NCV_bisplines_3 (meaning global cubic bisplines with a GLS error model) ) and the variable "model_str" specifies which model one wants to consider (diffadv, fisher, fisher_nonlin). The third cell then updates other various aspects of the study, as detailed throughout the paper.

Properror analyze results.ipynb - Once one has performed the PDE-FIND calculations and saved results in the folder "pickle_data", they can plot their results by using this ipython notebook. In the second cell here, the list "model_str_list" specifies which mathematical models will be considered. 

nonlin_fisher_IP.ipynb - ipython notebook to perform the inverse problem methodology discussed in Section 3(e) of our study.

Properror analyze results_spline_compare.ipynb - ipython notebook to plot the results for comparing 1d- and bispline computations with PDE-FIND, as demonstrated in the supplementary material.

make_learned_eqn_tables.py - Create the tables of learned equations provided in our supplementary material

PDE_FIND2.py -- Our code used to implement the PDE-FIND algorithm
