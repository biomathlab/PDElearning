This file contains the code to perform the PDE learning aspect of our study.

PDE_find_properror_sf_pruning.ipynb - ipython notebook to run PDE-FIND implementations. In the second cell, the variable "comp_str"specifies which denoising strategy one wishes to use (nn, finite_differences, or splines) and the variable "model_str" specifies which model one wants to consider (diffadv, fisher, fisher_nonlin). The third cell then updates other various aspects of the study, as detailed throughout the paper.

Properror analyze results.ipynb - Once one has performed the PDE-FIND calculations and saved results in the folder "pickle_data", they can plot their results by using this ipython notebook. In the third cell here, the variable "model_str" specifies which mathematical model will be used. 

nonlin_fisher_IP.ipynb - ipython notebook to perform the inverse problem methodology discussed in Section 3(e) of our study.

Properror analyze results_spline_compare.ipynb - ipython notebook to plot the results for comparing 1d- and bispline computations with PDE-FIND, as demonstrated in the supplementary material.
