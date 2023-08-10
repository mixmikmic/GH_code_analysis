import numpy as np
import numpy.linalg
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import celerite as ce
from celerite import terms
from celerite.modeling import Model

#defining helper functions for model
def expf(t,a,b):
    return np.exp(a*(t-b))

def linef(t, x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    b = y2 - (m*x2)
    return ((m*t)+b)

#defining the model for our fit
class PWModel(Model):
    parameter_names = ("xl", "xr", "al", "bl", "ar", "br")
    
    def get_value(self, t):
        result = np.empty(len(t))
        for i in range(len(t)): #had to tweak this to accept t-arrays, may affect speed...
            if(t[i]<self.xl):
                result[i] = expf(t[i], self.al, self.bl)
            elif(self.xl<=t[i] and t[i]<=self.xr):
                result[i] = linef(t[i], self.xl, expf(self.xl, self.al, self.bl), self.xr, expf(self.xr, self.ar, self.br))
            elif(self.xr<t[i]):
                result[i] = expf(t[i], self.ar, self.br)
        return result
    
    #the gradient terms were manually calculated
    def compute_gradient(self, t):
        yl = np.exp(self.al*(self.xl-self.bl))
        yr = np.exp(self.ar*(self.xr-self.br))
        result = np.empty([len(t), 6])
        result2 = np.empty([6, len(t)])
        for i in range(len(t)):
            ylt = np.exp(self.al*(t[i]-self.bl))
            yrt = np.exp(self.ar*(t[i]-self.br))
            if(t[i]<self.xl):
                dxl = 0.
                dxr = 0.
                dal = (t[i]-self.bl) * ylt
                dbl = -1* self.al * ylt
                dar = 0.
                dbr = 0.
                result[i] = np.array([dxl, dxr, dal, dbl, dar, dbr])
                result2[:,i] = result[i]

            elif(self.xl<=t[i] and t[i]<=self.xr):
                term = (t[i]-self.xr)
                dxl = ((term)/((self.xr-self.xl)**2)) * ((yr-yl) - (self.al * yl * (self.xr-self.xl)))
                dxr = (((term)/((self.xr-self.xl)**2)) * ((self.ar * yr * (self.xr-self.xl))-(yr-yl))) - ((yr-yl)/(self.xr-self.xl)) + (self.ar * yr)
                dal = ((term)/(self.xr-self.xl)) * (yl * (self.bl-self.xl))
                dbl = ((term)/(self.xr-self.xl)) *(self.al * yl)
                dar = (((term)/(self.xr-self.xl))+1) * ((self.xr-self.br)*yr)
                dbr = (((term)/(self.xr-self.xl))+1) * (-1*(self.ar*yr))
                result[i] = np.array([dxl, dxr, dal, dbl, dar, dbr])
                result2[:,i] = result[i]
        

            elif(self.xr<t[i]):
                dxl = 0.
                dxr = 0.
                dal = 0.
                dbl = 0.
                dar = (t[i]-self.br) * yrt
                dbr = -1 * self.ar * yrt
                result[i] = np.array([dxl, dxr, dal, dbl, dar, dbr])
                result2[:,i] = result[i]

        return result2

#defining double exponential model
class DExpModel(Model):
    parameter_names = ("xc", "al", "bl", "ar", "br")
    
    def get_value(self, t):
        result = np.empty(len(t))
        for i in range(len(t)): #had to tweak this to accept t-arrays, may affect speed...
            if(t[i]<self.xc):
                result[i] = expf(t[i], self.al, self.bl)
            elif(self.xc<t[i]):
                result[i] = expf(t[i], self.ar, self.br)
        return result
    
    #the gradient terms were manually calculated
    def compute_gradient(self, t):
        result = np.empty([len(t), 5])
        result2 = np.empty([5, len(t)])
        for i in range(len(t)):
            ylt = np.exp(self.al*(t[i]-self.bl))
            yrt = np.exp(self.ar*(t[i]-self.br))
            
            if(t[i]<self.xc):
                dxc = 0
                dal = (t[i]-self.bl) * ylt
                dbl = -1* self.al * ylt
                dar = 0.
                dbr = 0.
                result[i] = np.array([dxc, dal, dbl, dar, dbr])
                result2[:,i] = result[i]

            elif(self.xc<=t[i]):
                dxc = 0.
                dal = 0.
                dbl = 0.
                dar = (t[i]-self.br) * yrt
                dbr = -1 * self.ar * yrt
                result[i] = np.array([dxc, dal, dbl, dar, dbr])
                result2[:,i] = result[i]

        return result2
    
class CTSModel(Model):
    parameter_names = ("A", "tau1", "tau2")
    def get_value(self, t):
        lam = np.exp(np.sqrt(2*(self.tau1/self.tau2)))
        return self.A*lam*np.exp((-self.tau1/t)-(t/self.tau2))
    #the gradient terms were manually calculated
    def compute_gradient(self, t):
        lam = np.exp(np.sqrt(2*(self.tau1/self.tau2)))
        dA = (1./self.A) * self.get_value(t)
        dtau1 = ((1/(self.tau2 * np.log(lam))) - (1/t)) * self.get_value(t)
        dtau2 = ((t/(self.tau2**2)) - (self.tau1/((self.tau2**2) * np.log(lam)))) * self.get_value(t)
        return np.array([dA, dtau1, dtau2])

#previously used paramter estimation
params0 = [4.08344998e+02, 7.15585975e+02, 4.26056582e-02, 1.61781302e+02, -3.64834836e-03, 3.94850711e+03]
TPW_Model = PWModel(xl = params0[0], xr = params0[1], al = params0[2], bl = params0[3], ar = params0[4], br = params0[5])
params_vec = TPW_Model.get_parameter_vector()

paramsDExp = [7.00000000e+02, 5.07236898e-03, -1.64964096e+03, -3.53546301e-03, 4.04367967e+03]
TDExp_Model = DExpModel(xc = paramsDExp[0], al = paramsDExp[1], bl = paramsDExp[2], ar = paramsDExp[3], br = paramsDExp[4])
params_vec_DExp = TDExp_Model.get_parameter_vector()

paramsCTS = [3.2e+06, 3.9e+03, 1e+02]
TCTS_Model = CTSModel(A=paramsCTS[0], tau1 = paramsCTS[1], tau2= paramsCTS[2])
params_vec__cts = TCTS_Model.get_parameter_vector()

data1 = "/Users/chris/Documents/QPP/SolarFlareGPs/data/120704187_ctime_lc.txt"
t1, I1 = np.loadtxt(data1, unpack=True)

x = t1
y = I1
yinit = TPW_Model.get_value(x)
yinitDExp = TDExp_Model.get_value(x)
yinitCTS = TCTS_Model.get_value(x)

yerr = np.sqrt(10000 + I1)

#values from prior work 
A0 = 44.48325411e+07
tau0 = 1.36841695e+00



kernel = terms.RealTerm(log_a = np.log(A0), log_c = np.log(1./tau0))
kernel2 = terms.RealTerm(log_a = np.log(A0), log_c = np.log(1./tau0))
kernel3 = terms.RealTerm(log_a = np.log(A0), log_c = np.log(1./tau0))

gp = ce.GP(kernel, mean=TPW_Model, fit_mean=True)
gp2 = ce.GP(kernel2, mean=TDExp_Model, fit_mean=True)
gp3 = ce.GP(kernel3, mean=TCTS_Model, fit_mean=True)

gp.compute(x, yerr)
gp2.compute(x, yerr)
gp3.compute(x, yerr)

#defining cost function:
def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

def grad_neg_log_like(params, y, gp):   
    gp.set_parameter_vector(params)
    return -gp.grad_log_likelihood(y)[1]

#fitting
initial_params = gp.get_parameter_vector()
bounds = gp.get_parameter_bounds()
soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like, method ="L-BFGS-B", args = (y, gp))

initial_paramsD = gp2.get_parameter_vector()
boundsD = gp2.get_parameter_bounds()
soln2 = minimize(neg_log_like, initial_paramsD, jac=grad_neg_log_like, method ="L-BFGS-B", args = (y, gp2))

initial_paramsCTS = gp3.get_parameter_vector()
boundsCTS = gp3.get_parameter_bounds()
soln3 = minimize(neg_log_like, initial_paramsCTS, jac = grad_neg_log_like, method = "L-BFGS-B", args = (y, gp3))

final_params = soln.x
gp.set_parameter_vector(final_params)
print soln
print"\n"

final_paramsD = soln2.x
gp2.set_parameter_vector(final_paramsD)
print soln2
print '\n'

final_paramsCTS = soln3.x
gp3.set_parameter_vector(final_paramsCTS)
print soln3
print '\n'
print paramsCTS
print soln3.x[2:]

#max prediction
t = np.linspace(x[0], x[-1:][0], 500)
mu, var = gp.predict(y, t, return_var=True)
mu2, var2 = gp2.predict(y, t, return_var=True)
mu3, var3 = gp3.predict(y, t, return_var=True)
std = np.sqrt(var)
std2 = np.sqrt(var2)
std3 = np.sqrt(var3)



#plot data
color = "#ff7f0e"
plt.figure(figsize = (15,10))
plt.errorbar(x, y, yerr=yerr, fmt="-k", capsize=0)
plt.plot(t, mu, color=color)
plt.fill_between(t, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.xlim(0, 1800)
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
plt.title("maximum likelihood prediction")

#plot prediction
plt.plot(x, yinit, 'r-')
fpwmodel = PWModel(xl=final_params[2], xr=final_params[3], al=final_params[4], bl=final_params[5], ar=final_params[6], br=final_params[7])
ytest = fpwmodel.get_value(x)
plt.plot(x, ytest, 'b--')

#plot2
plt.figure(figsize = (15,10))
plt.errorbar(x, y, yerr=yerr, fmt="-k", capsize=0)
plt.plot(t, mu2, color=color)
plt.fill_between(t, mu2+std2, mu2-std2, color=color, alpha=0.3, edgecolor="none")
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.xlim(0, 1800)
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
plt.title("maximum likelihood prediction")

#plot prediction
plt.plot(x, yinitDExp, 'r-')
fdexpmodel = DExpModel(xc=final_paramsD[2], al=final_paramsD[3], bl=final_paramsD[4], ar=final_paramsD[5], br=final_paramsD[6])
ytest2 = fdexpmodel.get_value(x)
plt.plot(x, ytest2, 'b--')
plt.show()

#plot3
plt.figure(figsize = (15,10))
plt.errorbar(x, y, yerr=yerr, fmt="-k", capsize=0)
plt.plot(t, mu3, color=color)
plt.fill_between(t, mu3+std3, mu3-std3, color=color, alpha=0.3, edgecolor="none")
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.xlim(0, 1800)
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
plt.title("maximum likelihood prediction")

#plot prediction
plt.plot(x, yinitCTS, 'r-')
fCTSmodel = CTSModel(A=final_paramsCTS[2], tau1=final_paramsCTS[3], tau2=final_paramsCTS[4])
ytest3 = fCTSmodel.get_value(x)
plt.plot(x, ytest3, 'b--')
plt.show()

#testing out our likelihood function
def man_log_like(y, gp):
    resid = y - gp.mean.get_value(y)
    V = gp.get_matrix()
    t1 = np.dot(resid, np.dot(np.linalg.inv(V), resid))
    t2 = np.log(2 * np.pi * np.linalg.det(V))
    result = t1+t2
    print t1
    print t2
    print result
    print np.exp(-(result/2))
  

man_log_like(y, gp)
print '\n'
man_log_like(y, gp2)
print '\n'
man_log_like(y, gp3)

