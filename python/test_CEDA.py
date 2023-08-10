get_ipython().magic('pylab inline')
get_ipython().magic('matplotlib inline')
pylab.rcParams['figure.figsize'] = (16, 9)

# numpy, matplotib and others
import numpy as np
from numpy.random import RandomState
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import os
import pickle

# L63 and L96 models using fortran (fast)
import models.l63f as mdl_l63
import models.l96f as mdl_l96

# L63 and L96 models using python (slow)
from models.L63 import l63_predict, l63_jac
from models.L96 import l96_predict, l96_f

# data assimilation routines
from algos.EM_EnKS import EnKS, EM_EnKS
from algos.EM_EKS import EKS, EM_EKS
from algos.utils import climat_background, RMSE, gen_truth, gen_obs

### GENERATE SIMULATED DATA (LORENZ-63 MODEL)

# random number generator
prng = RandomState(9)

# dynamical model
Nx = 3 # dimension of the state
dt = .01 # integration time
sigma = 10;rho = 28;beta = 8./3 # physical parameters
#f = lambda x: l63_predict(x, dt, sigma, rho, beta) # python version (slow)
fmdl=mdl_l63.M(sigma=sigma, rho=rho, beta=beta, dtcy=dt)
f = lambda x: fmdl.integ(x) # fortran version (fast)
jacF = lambda x: l63_jac(x, dt, sigma, rho, beta) # python version (slow)

# observation operator
No = 3 # dimension of the observations
h = lambda x: x
jacH = lambda x: np.eye(Nx)
H = np.eye(Nx)

# background state
x0_true = np.r_[6.39435776, 9.23172442, 19.15323224]
xb, B = climat_background(f, x0_true, 5000)

# size of the sequence
T = 1000 # 10 Lorenz-63 times
time = range(T)*array([dt])

# generate state
Q_true = np.eye(Nx) * .05
X_true = gen_truth(f, x0_true, T, Q_true, prng)

# generate observations
dt_obs = 5 # 1 observation every dt_obs time steps
R_true = np.eye(No) * 2
Yo = gen_obs(h, X_true, R_true, dt_obs, prng)

### PLOT STATE AND OBSERVATIONS

line1,=plt.plot(time,X_true[0,1:T+1],'-r');plt.plot(time,Yo[0,:],'.r')
line2,=plt.plot(time,X_true[1,1:T+1],'-b');plt.plot(time,Yo[1,:],'.b')
line3,=plt.plot(time,X_true[2,1:T+1],'-g');plt.plot(time,Yo[2,:],'.g')
plt.title('Lorenz-63 true (continuous lines) and observed trajectories (points)')
plt.xlabel('Lorenz-63 times')
plt.legend([line1, line2, line3], ['$x_1$', '$x_2$', '$x_3$'])

### INITIALIZE THE EM ALGORITHMS

# EM parameters
N_iter = 500
Q_init = np.eye(Nx)
R_init = np.eye(No)
N = 100 # number of members (only in EM-EnKS)

### APPLY THE EM ALGORITHM ON EXTENDED KALMAN SMOOTHER (EM-EKS)

# parameters
params = { 'initial_background_state'                 : xb,
           'initial_background_covariance'            : B,
           'initial_model_noise_covariance'           : Q_init,
           'initial_observation_noise_covariance'     : R_init,
           'model_dynamics'                           : f,
           'model_jacobian'                           : jacF,
           'observation_operator'                     : h,
           'observation_jacobian'                     : jacH,
           'observations'                             : Yo,
           'nb_EM_iterations'                         : N_iter,
           'true_state'                               : X_true,
           'state_size'                               : Nx,
           'observation_size'                         : No,
           'temporal_window_size'                     : T,
           'model_noise_covariance_structure'         : 'full',
           'is_model_noise_covariance_estimated'      : True,
           'is_observation_noise_covariance_estimated': True,
           'is_background_estimated'                  : True,
           'inflation_factor'                         : 1 }

# function
res_EM_EKS = EM_EKS(params)

### APPLY THE EM ALGORITHM ON ENSEMBLE KALMAN SMOOTHER (EM-EnKS)

# parameters
params = { 'initial_background_state'                 : xb,
           'initial_background_covariance'            : B,
           'initial_model_noise_covariance'           : Q_init,
           'initial_observation_noise_covariance'     : R_init,
           'model_dynamics'                           : f,
           'observation_matrix'                       : H,
           'observations'                             : Yo,
           'nb_particles'                             : N,
           'nb_EM_iterations'                         : N_iter,
           'true_state'                               : X_true,
           'inflation_factor'                         : 1,
           'temporal_window_size'                     : T,
           'state_size'                               : Nx,
           'observation_size'                         : No,
           'is_model_noise_covariance_estimated'      : True,
           'is_observation_noise_covariance_estimated': True,
           'is_background_estimated'                  : True,
           'model_noise_covariance_structure'         : 'full'}

# function
res_EM_EnKS = EM_EnKS(params, prng)

### COMPARE RESULTS BETWEEN THE TWO STRATEGIES (EM-EKS AND EM-EnKS)

# extract outputs
Q_EKS = res_EM_EKS['EM_model_noise_covariance']
Q_EnKS = res_EM_EnKS['EM_model_noise_covariance']
R_EKS = res_EM_EKS['EM_observation_noise_covariance']
R_EnKS = res_EM_EnKS['EM_observation_noise_covariance']
loglik_EKS=res_EM_EKS['loglikelihood']
loglik_EnKS=res_EM_EnKS['loglikelihood']
RMSE_EKS=res_EM_EKS['RMSE']
RMSE_EnKS=res_EM_EnKS['RMSE']

# plot trace of Q
plt.subplot(2,2,1)
line1,=plt.plot(np.trace(Q_EKS)/Nx,'b')
line2,=plt.plot(np.trace(Q_EnKS)/Nx,'r')
line3,=plt.plot((1,N_iter),(np.trace(Q_true)/Nx,np.trace(Q_true)/Nx),'--k')
plt.title('Q estimates')
plt.legend([line1, line2, line3], ['EM-EKS', 'EM-EnKS', 'True Q'])

# plot trace of R
plt.subplot(2,2,2)
line1,=plt.plot(np.trace(R_EKS)/No,'b')
line2,=plt.plot(np.trace(R_EnKS)/No,'r')
line3,=plt.plot((1,N_iter),(np.trace(R_true)/No,np.trace(R_true)/No),'--k')
plt.title('R estimates')
plt.legend([line1, line2, line3], ['EM-EKS', 'EM-EnKS', 'True R'])

# plot log-likelihood
plt.subplot(2,2,3)
line1,=plt.plot(loglik_EKS,'b')
line2,=plt.plot(loglik_EnKS,'r')
plt.title('Log-likelihood')
plt.legend([line1, line2], ['EM-EKS', 'EM-EnKS'])

# plot Root Mean Square Error
plt.subplot(2,2,4)
line1,=plt.plot(RMSE_EKS,'b')
line2,=plt.plot(RMSE_EnKS,'r')
plt.title('RMSE')
plt.legend([line1, line2], ['EM-EKS', 'EM-EnKS'])

### GENERATE SIMULATED DATA (LORENZ 96 MODEL)

# random number generator
prng = RandomState(1)

# dynamical model
Nx = 40 # dimension of the state
dt = .05 # integration time
F = 8 # physical parameter
# f = lambda x: l96_predict(x,dt,F) # python version (slow)
fmdl=mdl_l96.M(dtcy=dt, force=F, nx=Nx)
f = lambda x: fmdl.integ(x) # fortran version (fast)

# observation operator
No = 40 # dimension of the observations
h = lambda x: x
jacH = lambda x: np.eye(Nx)
H = np.eye(Nx)

# background state
x0_true = array(zeros(Nx))
xb, B = climat_background(f, x0_true, 5000)

# size of the sequence
T = 400 # 20 Lorenz-96 times
time = range(T+1)*array([dt])

# generate state
Q_true = np.eye(Nx) * .1
X_true = gen_truth(f, x0_true, T+50, Q_true, prng)

# generate observations
dt_obs = 2 # 1 observation every dt_obs time steps
R_true = np.eye(No) * 2
Yo = gen_obs(h, X_true, R_true, dt_obs, prng)

# remove first part of the sequence (time to converge to the attractor)
X_true = X_true[:,50:T+50+1]
Yo = Yo[:,50:T+50+1]

### PLOT STATE AND OBSERVATIONS

[X,Y]=meshgrid(range(Nx),time)
subplot(1,2,1);pcolor(X,Y,X_true.T);xlim([0,Nx-1]);clim([-10,10]);ylabel('Lorenz-96 times');title('True trajectories')
subplot(1,2,2);pcolor(X,Y,ma.masked_where(isnan(Yo.T),Yo.T));xlim([0,Nx-1]);clim([-10,10]);ylabel('Lorenz-96 times');title('Observed trajectories')

### APPLY THE EM ALGORITHM ON ENSEMBLE KALMAN SMOOTHER (EM-EnKS)

# EM parameters
N_iter = 50
Q_init = np.eye(Nx)
R_init = np.eye(No)
N = 100 # number of members

# parameters
params = { 'initial_background_state'                 : xb,
           'initial_background_covariance'            : B,
           'initial_model_noise_covariance'           : Q_init,
           'initial_observation_noise_covariance'     : R_init,
           'model_dynamics'                           : f,
           'observation_matrix'                       : H,
           'observations'                             : Yo,
           'nb_particles'                             : N,
           'nb_EM_iterations'                         : N_iter,
           'true_state'                               : X_true,
           'inflation_factor'                         : 1,
           'temporal_window_size'                     : T,
           'state_size'                               : Nx,
           'observation_size'                         : No,
           'is_model_noise_covariance_estimated'      : True,
           'is_observation_noise_covariance_estimated': True,
           'is_background_estimated'                  : True,
           'model_noise_covariance_structure'         : 'full',
           #'model_noise_covariance_matrix_template'   : np.eye(Nx) # only for constant model noise covariance
         }

# function
res_EM_EnKS = EM_EnKS(params, prng)

### PLOT RESULTS OF EM-EnKS

# extract outputs
Q_EnKS = res_EM_EnKS['EM_model_noise_covariance']
R_EnKS = res_EM_EnKS['EM_observation_noise_covariance']
loglik_EnKS=res_EM_EnKS['loglikelihood']
RMSE_EnKS=res_EM_EnKS['RMSE']
Xs=res_EM_EnKS['smoothed_ensemble']

# plot trace of Q
plt.subplot(2,2,1)
line2,=plt.plot(np.trace(Q_EnKS)/Nx,'r')
line3,=plt.plot((0,N_iter),(np.trace(Q_true)/Nx,np.trace(Q_true)/Nx),'--k')
plt.title('Q estimates')
ylim([0,1])
plt.legend([line2, line3], ['EM-EnKS', 'True Q'])

# plot trace of R
plt.subplot(2,2,2)
line2,=plt.plot(np.trace(R_EnKS)/No,'r')
line3,=plt.plot((0,N_iter),(np.trace(R_true)/No,np.trace(R_true)/No),'--k')
plt.title('R estimates')
plt.legend([line2, line3], ['EM-EnKS', 'True R'])

# plot log-likelihood
plt.subplot(2,2,3)
plt.plot(loglik_EnKS,'r')
plt.title('Log-likelihood')

# plot Root Mean Square Error
plt.subplot(2,2,4)
plt.plot(RMSE_EnKS,'r')
plt.title('RMSE')

