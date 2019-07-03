import numpy as np
import surface_fitter

data_names = ['advection_diffusion','advection_diffusion']#,'fisher','fisher_nonlin']
inds = ['$\sigma = 00$','$\sigma = 01$','$\sigma = 05$','$\sigma = 10$','$\sigma = 25$','$\sigma = 50$']
model_names = ['finite_differences','bisplines','NCV_bisplines','global_NCV_bisplines_3','nn']
print_names = ['FD','LCVSP','LNCVSP','GNCVSP','ANN']
skip = 20
threshold = 0.0001

for data_name in data_names:
    
    print '\\verb+' + data_name + '+'
    print '\\begin{tabular}{cccccc}'
    print '    Error & Method & $u$ RMSE & $u_{t}$ RMSE & $u_{x}$ RMSE & $u_{xx}$ RMSE \\\\ '
    print '    \\hline'

    for ind in inds:
        
        for model_name in model_names:
            
            dataset = data_name+'_'+ind[-3:-1]+'_'+model_name

            reload(surface_fitter)
            from surface_fitter import SurfNN
            nn = SurfNN(data_name+'_'+ind[-3:-1], None)
            x_flat, y_flat, U_min, U_max, U_shape = nn.load_data()

            # rank-1 arrays
            x = x_flat[:,0]
            t = x_flat[:,1]
            u_noise = y_flat[:,0]

            # rank-2 arrays
            X = x.reshape(U_shape)             # space mesh
            T = t.reshape(U_shape)             # time mesh
            U_noise = u_noise.reshape(U_shape) # normalized noisy data
            U_true = nn.U_true                 # normalized clean data
            U_x_true = nn.U_x_true             # normalized clean data
            U_xx_true = nn.U_xx_true           # normalized clean data
            U_t_true = nn.U_t_true             # normalized clean data

            # load predictions
            surface_data = np.load('data/'+dataset+'.npy').item()
            U_pred = surface_data['inputs'][0]    # un-normalized prediction
            U_x_pred = surface_data['inputs'][1]  # un-normalized prediction
            U_xx_pred = surface_data['inputs'][2] # un-normalized prediction
            U_t_pred = surface_data['outputs'][0] # un-normalized prediction
            
            # scale predictions to (0, 1)
            U_pred = (U_pred - U_min)/U_max       # normalized prediction
            U_x_pred = (U_x_pred - U_min)/U_max   # normalized prediction
            U_xx_pred = (U_xx_pred - U_min)/U_max # normalized prediction
            U_t_pred = (U_t_pred - U_min)/U_max   # normalized prediction
            
            # skip initial timepoints if necessary 
            U_true = U_true[:, skip:]
            U_t_true = U_t_true[:, skip:]
            U_x_true = U_x_true[:, skip:]
            U_xx_true = U_xx_true[:, skip:]
            U_pred = U_pred[:, skip:]
            U_t_pred = U_t_pred[:, skip:]
            U_x_pred = U_x_pred[:, skip:]
            U_xx_pred = U_xx_pred[:, skip:]
            
            # compute relative mean square error
            U_mse = np.mean(np.square((U_true - U_pred)[np.abs(U_true>threshold)] / \
                                       U_true[np.abs(U_true>threshold)]))
            U_t_mse = np.mean(np.square((U_t_true - U_t_pred)[np.abs(U_t_true>threshold)] / \
                                         U_t_true[np.abs(U_t_true>threshold)]))
            U_x_mse = np.mean(np.square((U_x_true - U_x_pred)[np.abs(U_x_true>threshold)] / \
                                         U_x_true[np.abs(U_x_true>threshold)]))
            U_xx_mse = np.mean(np.square((U_xx_true - U_xx_pred)[np.abs(U_xx_true>threshold)] / \
                                          U_xx_true[np.abs(U_xx_true>threshold)]))
            
            print '    '+ind+' & '+print_names[model_names.index(model_name)]+ \
                  ' & %1.2e & %1.2e & %1.2e & %1.2e \\\\' \
                  %(U_mse, U_t_mse, U_x_mse, U_xx_mse)

        print '    \\hline'
    print '\\end{tabular}'
    print ''