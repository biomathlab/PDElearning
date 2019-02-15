import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# generate grids
dt = 0.003
dx = 0.01
num_t = 300
num_x = 100
t = np.arange(dt, num_t*dt + dt, dt)
x = np.arange(0.0, num_x*dx + dx, dx)
[X,T] = np.meshgrid(x,t);

# equation params
u0 = 1.00 # initial condition
x0 = 0.20 # initial location
D = 0.01 # diffusion coefficient
c = 0.80 # advection coefficient

# analytic solutions 
U = (u0*np.exp(-(x0 - X + c*T)**2.0/(4.0*D*T))) / (2.0*np.pi**(0.5)*(D*T)**(0.5))
U_x = (u0*np.exp(-(x0 - X + c*T)**2.0/(4.0*D*T))*(2.0*x0 - 2.0*X + 2.0*c*T)) / \
      (8.0*D*T*np.pi**(0.5)*(D*T)**(0.5))
U_xx = (u0*np.exp(-(x0 - X + c*T)**2.0/(4.0*D*T)) * \
       (c**2.0*T**2.0 - 2.0*c*T*X + 2.0*c*T*x0 - 2.0*D*T + X**2 - 2.0*X*x0 + x0**2.0)) / \
       (8.0*D**2.0*T**2.0*np.pi**(0.5)*(D*T)**(0.5))
U_t = -(u0*np.exp(-(x0 - X + c*T)**2.0/(4.0*D*T)) * \
       (c**2.0*T**2.0 + 2.0*D*T - X**2.0 + 2.0*X*x0 - x0**2.0)) / \
       (8.0*T*np.pi**(0.5)*(D*T)**(1.5))

# save the surfaces
data = {}
data['D'] = D
data['c'] = c
data['x'] = x
data['t'] = t
data['U'] = U
data['U_t'] = U_t
data['U_x'] = U_x
data['U_xx'] = U_xx
np.save('data/advection_diffusion_00',data)

# 
# noisy GLS solutions
# 

U_median = np.median(U[np.abs(U)>0.001])
gamma = 1.0 # proportional error constant

# 1% error
noise_level = 0.01
U_noise = U + noise_level * np.abs(U)**gamma * np.random.normal(size=U.shape)
U_noise = (U_noise>0)*U_noise
data = {}
data['D'] = D
data['c'] = c
data['x'] = x
data['t'] = t
data['U'] = U_noise
data['gamma'] = gamma
np.save('data/advection_diffusion_01',data)

# 5% error
noise_level = 0.05
U_noise = U + noise_level * np.abs(U)**gamma * np.random.normal(size=U.shape)
U_noise = (U_noise>0)*U_noise
data = {}
data['D'] = D
data['c'] = c
data['x'] = x
data['t'] = t
data['U'] = U_noise
data['gamma'] = gamma
np.save('data/advection_diffusion_05',data)

# 10% error
noise_level = 0.10
U_noise = U + noise_level * np.abs(U)**gamma * np.random.normal(size=U.shape)
U_noise = (U_noise>0)*U_noise
data = {}
data['D'] = D
data['c'] = c
data['x'] = x
data['t'] = t
data['U'] = U_noise
data['gamma'] = gamma
np.save('data/advection_diffusion_10',data)

# 25% error
noise_level = 0.25
U_noise = U + noise_level * np.abs(U)**gamma * np.random.normal(size=U.shape)
U_noise = (U_noise>0)*U_noise
data = {}
data['D'] = D
data['c'] = c
data['x'] = x
data['t'] = t
data['U'] = U_noise
data['gamma'] = gamma
np.save('data/advection_diffusion_25',data)

# 50% error
noise_level = 0.50
U_noise = U + noise_level * np.abs(U)**gamma * np.random.normal(size=U.shape)
U_noise = (U_noise>0)*U_noise
data = {}
data['D'] = D
data['c'] = c
data['x'] = x
data['t'] = t
data['U'] = U_noise
data['gamma'] = gamma
np.save('data/advection_diffusion_50',data)