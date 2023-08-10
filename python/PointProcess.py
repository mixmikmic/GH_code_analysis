#IPython is what you are using now to run the notebook
import IPython
print ("IPython version:      %6.6s (newest at 6.1.0)" % IPython.__version__)

# Numpy is a library for working with Arrays
import numpy as np
print ("Numpy version:        %6.6s (newest at 1.13.1)" % np.__version__)

# SciPy implements many different numerical algorithms
import scipy as sp
print ("SciPy version:        %6.6s (newest at 0.19.1)" % sp.__version__)

# Pandas makes working with data tables easier
import pandas as pd
print ("Pandas version:       %6.6s (newest at 0.20.3)" % pd.__version__)

# Module for plotting
import matplotlib 
print ("Mapltolib version:    %6.6s (newest at 2.0.2)" % matplotlib.__version__)

# SciKit Learn implements several Machine Learning algorithms
import sklearn
print ("Scikit-Learn version: %6.6s (newest at 0.19.0)" % sklearn.__version__)

# MNE is a package for processing (EEG) and (MEG) data 
import mne
print ("MNE version:          %6.6s (newest at 0.14.1)" % mne.__version__)

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import peakutils as pku
from peakutils.plot import plot as pplot
import seaborn as sns
from scipy import signal
from scipy.optimize import minimize,fmin_l_bfgs_b

class IGPointProcess():
    """Class for Renewal Inverse Gaussian Point Process"""
    def __init__(self, RTsInInterval):
        self.RTsInInterval = [float(i) for i in RTsInInterval]
        self.times = np.cumsum(RTsInInterval)
        self.theta0 = None
        self.theta1 = None
        self.mu = None
        self.sig = None
        self.df = None

    def renewalIG(self):
        """Determines the parameters of the Inverse Gaussian waiting time
        Dist: MLE estimate of distribution mean (theta0), MLE estimate of 
        shape parameter (theta1), the mean (mu), and variance (sig)

        Returns:
        -------
        -theta0: MLE estimate of distribution mean
        -theta1: MLE estimate of distribution shape parameter
        -mu: mean of IG(theta0, theta1)
        -sig: variance of IG(theta0, theta1)
        """
        #MLE estimaton
        theta0=np.mean(self.RTsInInterval)
        sums = 0
        for i in self.RTsInInterval:
            sums += np.reciprocal(i)-np.reciprocal(theta0)
        theta1 = np.reciprocal(sums/float(len(self.RTsInInterval)))
        #Now using the definition of mean and variance for IG
        mu=theta0
        sig=1./theta1*theta0**3
        self.theta0 = theta0
        self.theta1 = theta1
        self.mu = mu
        self.sig = sig
        return theta0, theta1, mu, sig
    
    def print_parameters(self):
        """Print out fitted parameters"""
        print("Theta: ", str(self.theta0), str(self.theta1))
        print('Mean: '+str(self.mu))
        print('Variance: '+str(self.sig))
        
    def plot_reaction_times(self):
        """Plots the raw reaction times"""
        plt.figure()
        plt.plot(range(len(self.RTsInInterval)), self.RTsInInterval)
        plt.ylabel('Reaction Times [s]')
        plt.xlabel('Time')
        plt.show()
        
    def plot_distribution(self):
        """Plot a histogram of reaction times and the fitted
        distribution"""
        plt.figure()
        wald=np.random.wald(self.theta0,self.theta1,(10000,))
        sns.distplot(self.RTsInInterval, kde=False, norm_hist = True)
        sns.distplot(wald, hist=False)
        plt.xlabel('Reaction Time [s]')
        plt.ylabel("Probability")
        plt.show()
        
    def sample_meanInverse(self):
        """Sample the distribution to produce the Mean Inverse
        RT metric"""
        wald=np.random.wald(self.theta0,self.theta1,(1000,))
        inv = 1.0/wald
        return 1000*np.mean(inv)
    
class Collect_IG_Data():
    def __init__(self):
        self.df = None
        
    def collect_IG_data(self, participantDict):
        data = []
        for participant in participantDict:
            pvtDetail = participantDict[participant].pvtDetail
            pvt = participantDict[participant].pvt
            FDData = pvtDetail[(pvtDetail.DecimalTime >= participantDict[participant].startFDtime) & (pvtDetail.DecimalTime <= participantDict[participant].endFDtime)]
            FDData_summary = pvt[(pvt.DecimalTime >= participantDict[participant].startFDtime) & (pvt.DecimalTime <= participantDict[participant].endFDtime)]

            for session in sorted(list(set(FDData.SESSION))):
                current_session_RT = FDData[FDData.SESSION == session]['RT'].values
                current_session_RT2 = np.array([i for i in current_session_RT if i >= 100])
                if len(current_session_RT2) != 0:
                    inverseGaussianPP = IGPointProcess(current_session_RT2)
                    theta0, theta1, mu, sig = inverseGaussianPP.renewalIG()
                    data.append([participant,session,inverseGaussianPP.sample_meanInverse(),theta0,theta1])
                        
        a = pd.DataFrame(data, columns=['SUBJECT','SESSION','IGMeanInverseRT','IGMean','IGShape'])
        a.to_csv("Datasets/IGPointProcessData.csv", index=False)
        self.df = a
        
    def saveToPVTSummaryFile(self):
        """Write the new features to the PVT Summary File (only
        done if the features are not already present)"""
        
        pvtTest = pd.read_csv("PVTSummaryData.csv", na_values = ['','.'], low_memory = False,encoding="latin-1")
        if 'IGMeanInverseRT' not in list(pvtTest):
            merged = pd.merge(pvtTest, self.df, on=['SUBJECT', 'SESSION'])
            merged.to_csv("PVTSummaryData.csv", index=False)
        
    def collectAndSaveData(self,participantDict):
        """Collect and save data"""
        self.collect_IG_data(participantDict)
        self.saveToPVTSummaryFile()
        

class HDIGPointProcess():
    """Class for History-Dependent Inverse Gaussian
    Point Process"""
    def __init__(self, RTsInInterval,p,alpha):
        self.RTsInInterval = [i for i in RTsInInterval]
        self.ul = np.cumsum(RTsInInterval)
        self.full_ul = [i for i in self.ul]
        self.theta0 = None
        self.theta1 = None
        self.mu = None
        self.sig = None
        self.p = p
        self.alpha = alpha
        
    def HDIG(self,theta):
        """Return the mean and standard deviation of the
        distribution"""
        scale=theta[-1]
        theta0=theta[0]
        H=self.history(self.ul,self.p)
        mu=theta0+np.dot(H[1:self.p+1],theta[1:self.p+1])
        sig=1./scale*mu**3
        return mu,sig
    
    def HDIGpdf(self,v,H,theta):
        """Return value of the probability distribution"""
        scale = theta[-1]
        theta0 = theta[0]
        mu = theta0+np.dot(H[1:self.p+1],theta[1:self.p+1])
        a = float(scale)
        b = (2*np.pi*v**3)
        c = -0.5*float(scale)*(v-mu)**2
        d = float(v*mu**2)
        return 0.5*(np.log(a)-np.log(b))+c/d
    
    def w(self,t,ui):
        """Weighting function"""
        return np.exp(-1*self.alpha*(t-ui))

    def history(self, u):
        """Calculates the history vector"""
        uk = u[-1] #last heartbeat
        w = np.flip(np.diff(u),0)
        if self.p > len(w):
            wp = np.append(w,np.zeros(self.p-len(w)))
        else:
            wp = w[0:self.p]
        H = np.append(uk, wp)
        return H
    
    def localML(self,theta, t):
        """Calculates Log Liklihood value"""
        nt = len(self.ul)
        f1=0
        for i in range(2,nt+1):
            H = self.history(self.ul[:i-1])  
            weight = self.w(t,self.ul[i-1])
            fx = self.HDIGpdf(self.ul[i-1]-self.ul[i-2],H,theta)
            f1 += weight*fx
        return -1*f1
    
    def meanConstraint(self,theta, t):
        """Constraint that mean must be >= 100 msec as anything less
        is not a 'valid' reaction time """
        H = self.history(self.ul)
        mu = theta[0]+np.dot(H[1:self.p+1],theta[1:self.p+1])
        return mu-100
    
    def meanConstraint2(self,theta, t):
        """Constraint that mean must be <= 10000 msec to ensure 
        that optimize routine doesn't blow up"""
        H = self.history(self.ul)
        mu = theta[0]+np.dot(H[1:self.p+1],theta[1:self.p+1])
        return -mu+10000
    
    def scaleConstraint(self,theta,t):
        """Constraint that scale parameter must be positive"""
        return theta[-1]

    def optML(self,x0,args): 
        """Optimization routine: minimizes the negative log liklihood
        function with the constraints """
        con = [{'type': 'ineq', 'fun': self.meanConstraint2, 'args': args},{'type': 'ineq', 'fun': self.meanConstraint, 'args': args},{'type': 'ineq', 'fun': self.scaleConstraint, 'args': args}]
        res=minimize(self.localML,x0,args,constraints=con,options={'disp':False})              
        theta_est=np.array(res.x)
        self.theta0 = theta_est[0]
        self.theta1 = theta_est[-1]
        H = self.history(self.ul)
        mu = theta_est[0]+np.dot(H[1:self.p+1],theta_est[1:self.p+1])
        self.mu = mu
        return self.theta0, self.theta1, res.status, self.mu, theta_est
    
    def iterateOverTs(self,theta0_start,theta1_start):
        """Iterate over the reaction times and fit a distribution at each.
        Then average the distribution parameters."""
        theta0_list = []
        theta1_list = []
        status_list = []
        mu_list = []
        for t in self.full_ul[self.p:]:
            theta_start = [0.01]*(self.p+2)
            theta_start[0] = theta0_start
            theta_start[-1] = theta1_start
            self.ul = self.full_ul[:self.full_ul.index(t)]
            theta0,theta1,status,mu, thetas = self.optML(theta_start,(t,))
            theta0_list.append(theta0)
            theta1_list.append(theta1)
            status_list.append(status)
            mu_list.append(mu)

        res = {'status':sum(status_list),'mu':np.mean(mu_list),'theta0':np.mean(theta0_list),'theta1':np.mean(theta1_list)}
        return res
        
    def plot_distribution(self, theta0, theta1):
        """Plot the raw reaction times and the fitted distribution"""
        plt.figure()
        sns.distplot(self.RTsInInterval, kde=False, norm_hist = True)
        wald=np.random.wald(theta0,theta1,(1000,))
        sns.distplot(wald, hist=False)
        plt.xlabel('Reaction Time [s]')
        plt.ylabel("Probability")
        plt.show()
        
    def sample_meanInverse(self,theta0,theta1):
        """Sample the distribution to produce the Mean Inverse
        RT metric"""
        wald=np.random.wald(theta0,theta1,(10000,))
        inv = np.reciprocal(wald)
        return 1000*np.mean(inv)
    
class Collect_HDIG_Data():
    def __init__(self, p, alpha):
        self.p = p
        self.alpha = alpha
        self.data = []
        self.df = None
    
    def collect_HDIG_data(self,participantDict):
        """Collect the HDIG parameters per session"""
        for participant in participantDict:
            pvtDetail = participantDict[participant].pvtDetail
            pvt = participantDict[participant].pvt
            FDData = pvtDetail[(pvtDetail.DecimalTime >= participantDict[participant].startFDtime) & (pvtDetail.DecimalTime <= participantDict[participant].endFDtime)]
            for session in sorted(list(set(FDData.SESSION))):
                current_session_RT = FDData[FDData.SESSION == session]['RT'].values
                current_session_RT2 = np.array([i for i in current_session_RT if i >= 100])
                if len(current_session_RT2) != 0:
                    inverseGaussianPP = IGPointProcess(current_session_RT2)
                    theta0, theta1, mu, sig = inverseGaussianPP.renewalIG()

                    HDIGinverseGaussianPP = HDIGPointProcess(current_session_RT2,self.p,self.alpha)
                    results = HDIGinverseGaussianPP.iterateOverTs(theta0,theta1)
                    meaninverse = HDIGinverseGaussianPP.sample_meanInverse(results['mu'],results['theta1'])
                    self.data.append([participant,session,meaninverse,results['theta0'],results['theta1'],results['status']])

        a = pd.DataFrame(self.data, columns=['SUBJECT','SESSION','HDIGMeanInverseRT','HDIGTheta0','HDIGTheta1','status'])
        a.to_csv("Datasets/HDIGPointProcessData.csv", index=False)
        self.df = a
        
    def saveToPVTSummaryFile(self):
        """Write the new features to the PVT Summary File (only
        done if the features are not already present)"""
        
        pvtTest = pd.read_csv("PVTSummaryData.csv", na_values = ['','.'], low_memory = False,encoding="latin-1")
        if 'HDIGMeanInverseRT' not in list(pvtTest):
            merged = pd.merge(pvtTest, self.df, on=['SUBJECT', 'SESSION'])
            merged.to_csv("PVTSummaryData.csv", index=False)
        
    def collectAndSaveData(self,participantDict):
        """Collect and save data"""
        self.collect_HDIG_data(participantDict)
        self.saveToPVTSummaryFile()

