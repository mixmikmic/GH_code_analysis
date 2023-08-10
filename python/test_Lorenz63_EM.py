#fortran files in "models" folder are required compiled before running this "ipynb".
get_ipython().run_line_magic('load_ext', 'fortranmagic')

import numpy as np 
from numpy.random import RandomState 
from numpy.linalg import cholesky
import matplotlib.pyplot as plt 
import seaborn as sns 
from save_load import loadTr

# L63 models 
import models.l63f as mdl_l63
from models.L63 import l63_predict, l63_jac

# import EM algorithms
from algos.CPF_BS_SEM import CPF_BS_SEM
from algos.CPF_AS_SEM import CPF_AS_SEM
from algos.PF_BS_EM import PF_BS_EM
from algos.EnKS_EM import EnKS_EM
from algos.EKS_EM import EKS_EM
from algos.utils import climat_background, RMSE, CV95, gen_truth, gen_obs

### GENERATE DATA OF LORENZ-63 MODEL

# random number generator
prng = RandomState(9)

dx = 3 # dimension of the state
dt = .15 # chosen model step dt \in [0.01, 0.25]-the larger dt the more nonliner model
sigma = 10;rho = 28;beta = 8./3 # physical parameters
#m = lambda x: l63_predict(x, dt, sigma, rho, beta) # python version (slow)
fmdl=mdl_l63.M(sigma=sigma, rho=rho, beta=beta, dtcy=dt)
m = lambda x: fmdl.integ(x) # dynamical model (fortran version (fast))
jacM = lambda x: l63_jac(x, dt, sigma, rho, beta) # Jacobian matrix (for EKS_EM only) of the dynamical model

dy = 2 # dimension of the observations
H = np.eye(dx)
H = H[(0,2),:] # first and third variables are observed
h = lambda x: H.dot(x)  # observation model
jacH = lambda x: H # Jacobian matrix  of the observation model(for EKS_EM only)

# Setting covariances
sig2_Q =1; sig2_R = 2 # parameters
Q_true = np.eye(dx) *sig2_Q # model covariance
R_true = np.eye(dy) *sig2_R # observation covariance

# prior state
x0_true = np.r_[8, 0, 30]


# generate data (true states + observations)
T0 = 1400
time = range(T0)*np.array([dt])
X = gen_truth(m, x0_true, T0, Q_true, prng)
dt_obs = 1
Y = gen_obs(h, X, R_true, dt_obs, prng)

data = loadTr('Lorenz_dataQ1dt15.pkl') # lock in case of runing on a different simulated data
X = data['X']; Y = data['Y'] # lock in case of runing on a different simulated data

# learning data
T = 100
T_burnin = 300
X_learning = X[:,T_burnin:T_burnin+T+1]
Y_learning = Y[:,T_burnin:T_burnin+T]

# Background information
xb, B, x0_tru = climat_background(m,Q_true, x0_true, 1000,prng)
# (In case of given the true initial state)
B = np.eye(dx) *sig2_Q
xb= X_learning[:,0]

# Plot simulated data
time_learning = np.arange(T+1)
sns.set_style("white")
plt.rcParams['figure.figsize'] = (15, 5)
plt.figure()
plt.plot(time_learning[1:],Y_learning.T, 'k.', markersize = 6); 
plt.plot(time_learning[1:],X_learning[[0,2],1:].T, 'k-', linewidth =1)
plt.plot(time_learning[1:],X_learning[1,1:].T, '-', color='gray', linewidth =1)
plt.title('${}$-simulated data with $dt={}, \sigma _Q^2 ={},  \sigma _R^2= {}$: observations (points) and true state variables (curves) where $x_2$ (gray curve) is unobserved component.'.format(T,dt, sig2_Q,sig2_R))
plt.xlim([1,T])
plt.xlabel('time ($t$)')
plt.grid()
plt.show()

### SETTING PARAMETERS FOR EM ALGORITHMS
N_iter =100 # number of iterations of EM algorithms

# Step functions
gam1 = np.ones(N_iter)  #for SEM (stochastic EM)
gam2 = np.ones(N_iter)
for k in range(50,N_iter):
    gam2[k] = k**(-0.7) # for SAEM (stochastic approximation EM)
#gam1 = gam2

rep = 100 # number of repetitions for each algorithm
Q_rep = np.zeros([3,rep,N_iter+1])
R_rep = np.zeros([3,rep,N_iter+1])
loglik_rep = np.zeros([3,rep,N_iter])
X_conditioning = np.zeros([dx,T]) # the conditioning trajectory (only for CPF-BS-SEM and CPF-AS-SEM)

# initial parameters
aintQ = .5; bintQ = 2
aintR = 1; bintR = 4
baseQ = np.eye(dx) # for fixed-constant form of model covariance
baseR = np.eye(dy) # for fixed-constant form of observation covariance 

Nf = 20 # number of particles
Ns = 20 # number of realizations
N = 20 # number of members (only for EnKS-EM)

### COMPUTE ESTIMATES WITH CPF-BS-SEM (our approach), CPF-AS-SEM, PF-BS-EM, EnKS-EM, EKS-EM
sns.set_style("white")
plt.rcParams['figure.figsize'] = (15, 5)
plt.figure()
estimateQ = True; estimateR = True; estimateX0 = True
structQ = 'const'; structR = 'const'
for i in range(rep):
    # uniformly sample the initial covariances
    Q_init = np.random.uniform(aintQ,bintQ)*np.eye(dx)
    R_init = np.random.uniform(aintR,bintR)*np.eye(dy)
    
    ### SEM ALGORITHM using CONDITIONAL PARTICLE FILTERING-BACKWARD SIMULATION SMOOTHER(CPF_BS_SEM) (proposed) 
    res_CPF_BS_SEM = CPF_BS_SEM(X_conditioning,Y_learning,X_learning,Q_init,m,H,R_init,xb,B,dx,dy, Nf, Ns,T,N_iter,gam1,estimateQ,estimateR,estimateX0,structQ,structR,baseQ,baseR, prng)

    ### SEM ALGORITHM using CONDITIONAL PARTICLE FILTERING-ANCESTOR SAMPLING SMOOTHER (CPF_AS_SEM) (new approach in DA)
    res_CPF_AS_SEM = CPF_AS_SEM(X_conditioning,Y_learning,X_learning,Q_init,m,H,R_init,xb,B,dx,dy, Nf, Ns,T,N_iter,gam1,estimateQ,estimateR,estimateX0, structQ, structR,baseQ,baseR, prng)

    ### SEM ALGORITHM using PARTICLE FILTERING-BACKWARD SMOOTHER (PF_BS_EM) (regular)
    res_PF_BS_EM = PF_BS_EM(Y_learning,X_learning,Q_init,m,H,R_init,xb,B,dx,dy, Nf, Ns,T,N_iter,gam1,estimateQ,estimateR,estimateX0, structQ, structR,baseQ,baseR, prng)

    ### EM ALGORITHM using ENSEMBLE KALMAN SMOOTHER (EnKS_EM) (regular)
    res_EnKS_EM = EnKS_EM(Y_learning,X_learning,Q_init,m,H,R_init,xb,B,dx,dy,N,T,N_iter,gam1,estimateQ,estimateR,estimateX0, structQ, structR,baseQ,baseR, 1,prng) # 1 is inflation factor chosen
   
    ### EM ALGORITHM using EXTENDED KALMAN SMOOTHER (EKS_EM) (regular)
    res_EKS_EM = EKS_EM(Y_learning,X_learning,Q_init,m,jacM,h,jacH,R_init,xb,B,dx,dy,T,N_iter,gam1,estimateQ,estimateR,estimateX0, structQ, structR,baseQ,baseR,1)

    ii =0; ilim =0
    #CPF-BSi
    Q_CPF_BS = res_CPF_BS_SEM['EM_model_noise_covariance'][:,:,ii:]
    R_CPF_BS = res_CPF_BS_SEM['EM_observation_noise_covariance'][:,:,ii:]
    loglik_CPF_BS= res_CPF_BS_SEM['loglikelihood'][ii:]
    Xs_CPF_BS = res_CPF_BS_SEM['smoothed_sample_all']
    #CPF-AS
    Q_CPF_AS = res_CPF_AS_SEM['EM_model_noise_covariance'][:,:,ii:]
    R_CPF_AS = res_CPF_AS_SEM['EM_observation_noise_covariance'][:,:,ii:]
    loglik_CPF_AS= res_CPF_AS_SEM['loglikelihood'][ii:]
    Xs_CPF_AS = res_CPF_AS_SEM['smoothed_sample_all']
    #PF-BSi
    Q_PF_BS = res_PF_BS_EM['EM_model_noise_covariance'][:,:,ii:]
    R_PF_BS = res_PF_BS_EM['EM_observation_noise_covariance'][:,:,ii:]
    loglik_PF_BS= res_PF_BS_EM['loglikelihood'][ii:]
    #EnKS
    Q_EnKS =  res_EnKS_EM['EM_model_noise_covariance'][:,:,ii:]
    R_EnKS =  res_EnKS_EM['EM_observation_noise_covariance'][:,:,ii:]
    loglik_EnKS= res_EnKS_EM['loglikelihood'][ii:]
    Xs_EnKS = res_EnKS_EM['smoothed_ensemble_all']
    #EKS
    Q_EKS =  res_EKS_EM['EM_model_noise_covariance'][:,:,ii:]
    R_EKS =  res_EKS_EM['EM_observation_noise_covariance'][:,:,ii:]
    loglik_EKS= res_EKS_EM['loglikelihood'][ii:]
    
    Q_rep[1,i,:] = np.trace(Q_CPF_BS)/dx
    Q_rep[0,i,:] = np.trace(Q_CPF_AS)/dx
    R_rep[1,i,:] = np.trace(R_CPF_BS)/dx
    R_rep[0,i,:] = np.trace(R_CPF_AS)/dx   
    loglik_rep[1,i,:] = loglik_CPF_BS
    loglik_rep[0,i,:] = loglik_CPF_AS
    
    # plot log-likelihood estimates
    plt.subplot(131)
    line0,=plt.plot(loglik_EKS,'xkcd:green')
    line1,=plt.plot(loglik_EnKS,'g')
    line2,=plt.plot(loglik_PF_BS,'xkcd:orange')
    line3,=plt.plot(loglik_CPF_AS,'b')
    line4,=plt.plot(loglik_CPF_BS,'r')
    plt.title('Log-likelihood')
    plt.xlabel('iteration($r$)')
    plt.xlim([ilim, N_iter])
    
    # plot of sig2_Q estimates
    plt.subplot(132)
    line0,=plt.plot(np.trace(Q_EKS)/dy,'xkcd:green')
    line1,=plt.plot(np.trace(Q_EnKS)/dx,'g')
    line2,=plt.plot(np.trace(Q_PF_BS)/dx, 'xkcd:orange')
    line3,=plt.plot(np.trace(Q_CPF_AS)/dx,'b')
    line4,=plt.plot(np.trace(Q_CPF_BS)/dx,'r')
    line5,=plt.plot((1,N_iter),(np.trace(Q_true)/dx,np.trace(Q_true)/dx),'k')
    plt.xlabel('iteration($r$)')
    plt.title('$\sigma_Q^2$')
    plt.xlim([ilim, N_iter])
    
    # plot sig2_R estimates
    plt.subplot(133)
    line0,=plt.plot(np.trace(R_EKS)/dy,'xkcd:green')
    line1,=plt.plot(np.trace(R_EnKS)/dy,'g')
    line2,=plt.plot(np.trace(R_PF_BS)/dy,'xkcd:orange')
    line3,=plt.plot(np.trace(R_CPF_AS)/dy,'-b')
    line4,=plt.plot(np.trace(R_CPF_BS)/dy,'-r')
    line5,=plt.plot((1,N_iter),(np.trace(R_true)/dy,np.trace(R_true)/dy),'k')
    plt.xlabel('iteration($r$)')
    plt.title('$\sigma_R^2$ estimates')
    plt.xlim([ilim, N_iter])
    plt.legend([line0, line1, line2, line3, line4, line5], ['EKS_EM','EnKS_EM', 'PF_BS_SEM','CPF_AS_SEM', 'CPF_BS_SEM', 'true parameter'])
plt.show()

print("EKS_EM, EnKS_EM with {} members and PF_BS_SEM with {} particles give biased and/or very noisy estimates".format(N, Nf))

### VIOLIN PLOT of ESTIMATION PERFORMANCES between CPF-BS-SEM and CPF-AS-SEM

ind_box = np.arange(0,N_iter+1,10)
ind_box[1:] = ind_box[1:]-1

data_CAS_Q = Q_rep[0,:,ind_box].T
data_CAS_R = R_rep[0,:,ind_box].T
data_CAS_llh = loglik_rep[0,:,ind_box].T

data_CPS_Q = Q_rep[1,:,ind_box].T 
data_CPS_R = R_rep[1,:,ind_box].T
data_CPS_llh = loglik_rep[1,:,ind_box].T

my_color = "Reds_r" 
my_labels = ["CPF-BS-SEM", "CPF-AS-SEM"]
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style("white")
plt.rcParams['figure.figsize'] = (15, 4)
ind_box = np.arange(0,N_iter+1,10)
ind_box[0] =1
iterations = (np.repeat(ind_box.T,(rep))).reshape(len(ind_box)*rep)
iterations = np.concatenate((iterations,iterations))

est = np.concatenate((data_CPS_llh.T.reshape(len(ind_box)*rep), data_CAS_llh.T.reshape(len(ind_box)*rep)))
methods = np.concatenate((np.repeat("CPF-BS-SEM",len(ind_box)*rep), np.repeat("CPF-AS-SEM",len(ind_box)*rep)))
df = {'iterations':iterations,'est':est,'methods': methods}
DAT = pd.DataFrame(data=df)
plt.figure()
plt.subplot(131)
line2 = sns.violinplot(x ='iterations',y= 'est', hue= 'methods', data= DAT, bw=0.3,palette= my_color, linewidth=1).set_ylabel('')
plt.legend('',frameon=False, loc= 4)
plt.xlabel('iteration ($r$)')
plt.title('log likelihood')
plt.grid()


est = np.concatenate((data_CPS_Q.T.reshape(len(ind_box)*rep), data_CAS_Q.T.reshape(len(ind_box)*rep)))
methods = np.concatenate((np.repeat("CPF-BS-SEM",len(ind_box)*rep), np.repeat("CPF-AS-SEM",len(ind_box)*rep)))
df = {'iterations':iterations,'est':est,'methods': methods}
DAT = pd.DataFrame(data=df)
plt.subplot(132)
line2 = sns.violinplot(x ='iterations',y= 'est', hue= 'methods', data= DAT, palette= my_color, linewidth=1).set_ylabel('')
line1 = plt.plot((-0.5,10.5*N_iter/100),(sig2_Q,sig2_Q),'-k',linewidth=1, label = 'True parameter' )
plt.legend(frameon=False, loc= 'best')
plt.xlabel('iteration ($r$)')
plt.title(r'$\sigma_Q^2$')
plt.grid()


est = np.concatenate(((data_CPS_R.T).reshape(len(ind_box)*rep), (data_CAS_R.T).reshape(len(ind_box)*rep)))
methods = np.concatenate((np.repeat("CPF-BS-SEM",len(ind_box)*rep), np.repeat("CPF-AS-SEM",len(ind_box)*rep)))
df = {'iterations':iterations,'est':est,'methods': methods}
DAT = pd.DataFrame(data=df)
plt.subplot(133)
line2 = sns.violinplot(x ='iterations',y= 'est', hue= 'methods', data= DAT, palette= my_color, linewidth=1).set_ylabel('')
line1 = plt.plot((-0.5,10.5*N_iter/100),(sig2_R,sig2_R),'-k',linewidth=1, label = 'True parameter' )
plt.legend('',frameon=False, loc= 'best')
plt.xlabel('iteration ($r$)')
plt.title(r'$\sigma_R^2$')
plt.grid()
plt.show()

print("CPF_BS_SEM with {} particles gives smaller bias and variance of parameter estimates than CPF_AS_SEM with {} particles.".format(Nf,Nf))

### COMPARE RECONSTRUCTION QUALITY (RMSE and COVERAGE PROBABILITY) between CPF-BS SMOOTHER and CPF-AS SMOOTHER with THEIR ESTIMATED PARAMETERS ON A TESTING DATA
from algos.utils import RMSE, CV95
from algos.CPF_BS_SEM import CPF_BS
from algos.CPF_AS_SEM import CPF_AS
# random number generator
#prng = RandomState(1)

#testing data
T_testing =1000 # 10 Lorenz-63 times
time = range(T_testing)*np.array([dt])
X_testing = X[:,T0-T_testing:]
Y_testing =  Y[:,T0-T_testing:]
B = np.eye(dx) *sig2_Q
xb= X_testing[:,0]

Nf=20; Ns=20; Niter=10
X_conditioning = np.zeros([dx,T_testing])

Q_CPF_AS = np.mean(Q_rep[0,:,-1])*baseQ
R_CPF_AS = np.mean(R_rep[0,:,-1])*baseR

Q_CPF_BS = np.mean(Q_rep[1,:,-1])*baseQ
R_CPF_BS = np.mean(R_rep[1,:,-1])*baseR

# CPF-BS
res_CPF_BS = CPF_BS(X_conditioning,Y_testing, X_testing,Q_CPF_BS,m,H,R_CPF_BS,xb,B,dx,dy, Nf, Ns,T_testing,Niter,prng)
Xs_CPF_BS = res_CPF_BS['smoothed_sample_all']

#CPF-AS
res_CPF_AS = CPF_AS(X_conditioning,Y_testing, X_testing,Q_CPF_BS,m,H,R_CPF_BS,xb,B,dx,dy, Nf, Ns,T_testing,Niter,prng)  
Xs_CPF_AS = res_CPF_AS['smoothed_sample_all']

#plot
time = np.arange(T_testing+1)
var = 1; num =Niter-1; tlim= [1,T_testing]
num1=Niter;num2= num +Niter+1-num1
XsBS = np.squeeze(Xs_CPF_BS[var,:,:,Niter+1-num1:num2]).transpose((2,0,1))
XsAS = np.squeeze(Xs_CPF_AS[var,:,:,Niter+1-num1:num2]).transpose((2,0,1))
XsBS = XsBS.reshape(Ns*num,T_testing+1)
XsAS = XsAS.reshape(Ns*num,T_testing+1)

XsBS_mean = XsBS.mean(axis=0); XsAS_mean = XsAS.mean(axis=0); 
cov_prob, CIlowBS, CIupBS = CV95(XsBS[:,1:],X_testing[var,1:])
RMSE_CPF_BS= RMSE(X_testing[var,1:] - XsBS[:,1:].mean(0))
print('RMSE_CPF_BS of $x_{}$= {}'.format(var+1,RMSE_CPF_BS))
print('cov_prob CPF_BS of $x_{}$ ={}%'.format(var+1,cov_prob))

cov_prob, CIlowAS, CIupAS = CV95(XsAS[:,1:],X_testing[var,1:])
RMSE_CPF_AS= RMSE(X_testing[var,1:] - XsAS[:,1:].mean(0))
print('RMSE_CPF_AS of $x_{}$ = {}'.format(var+1,RMSE_CPF_AS))
print('cov_prob CPF_AS of $x_{}$ = {}%'.format(var+1,cov_prob))

plt.rcParams['figure.figsize'] = (15, 6)
plt.figure()
plt.subplot(211)
line = plt.fill_between(time[1:], CIlowBS,CIupBS, color="xkcd:light red", alpha=0.3)#,sig2_Qha =0.3 )
#line0 = plt.plot(time[1:],Y_learning[var-1,:], '.k', markersize = 4, label ='observation')
line1 = plt.plot(time[1:],X_testing[var,1:], '-k', linewidth =1, label = 'true state')
line2 = plt.plot(time[1:],XsBS_mean[1:],'red',linewidth =1, label ='smoothed mean')
plt.xlabel('time ($t$)')
plt.ylabel('CPF-BS')
plt.xlim([1,T_testing])
plt.grid()
plt.title('Recontruction of the unobserved state ($x_2$) given the estimated parameters')

plt.subplot(212)
line = plt.fill_between(time[1:], CIlowAS,CIupAS, color="xkcd:light red", alpha=0.3)#,sig2_Qha =0.3 )
#line0 = plt.plot(time[1:],Y_learning[var-1,:], '.k', markersize = 4, label ='observation')
line1 = plt.plot(time[1:],X_testing[var,1:], '-k', linewidth =1, label = 'true state')
line2 = plt.plot(time[1:],XsAS_mean[1:],'red',linewidth =1, label ='smoothed mean')
plt.xlabel('time ($t$)')
plt.ylabel('CPF-AS')
plt.xlim([1,T_testing])
plt.grid()
plt.show()
print("For {} iterations, CPF_BS with {} particles gives larger cover probability and smaller of RMSE than CPF_AS {} particles, hence provides better state reconstruction.".format(Niter,Nf,Nf))

