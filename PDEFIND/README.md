This file contains the code to perform the equation learning aspect of our study.

Run the file "PDE_find_properror_sf_pruning.ipynb" to perform PDE-FIND implementations. In the second cell, the variable "comp_str"
specifies which denoising strategy one wishes to use (nn, finite_differences, or splines) and the variable "model_str"
specifies which model one wants to consider (diffadv, fisher, fisher_nonlin). The third cell then updates other various
aspects of the study, as detailed throughout the paper.

Once one has performed their PDE-FIND calculations and saved results in the folder "pickle_data", they can plot their results
by running "Properror analyze results.ipynb". In the third cell here, the variable "model_str" specifies which mathematical
model will be used. 

To perform the inverse problem methodology discussed in Section 3(e) of our study, run the file "nonlin_fisher_IP.ipynb".
