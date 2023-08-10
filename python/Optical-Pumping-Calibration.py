import matplotlib.pyplot as plt
#make pretty comments
from IPython.display import Latex 

#Make plots show up in notebook
get_ipython().magic('pylab inline --no-import-all')

#Import various things for later
from scipy import linspace,sqrt, exp,log #linspace allows us to generate linear array between start and stop points
import scipy.optimize as optimization #curve fitting
from numpy import array
#testing goodness of fit
from scipy.stats import chi2 
from scipy.stats import chisqprob 

freq_sweep85=array([.05,.1,.15,.2,.25,.3]) #frequency for sweep calibration
I_sweep85=array([0.32683,0.44222,0.55979,0.674415,0.789305,0.90482])
s_sweep85=array([0.0025,0.0012,0.0011,0.0015,0.0012,0.0011])
freq_sweep87=array([0.05,.1,.15,.20])
I_sweep87=array([0.38504,0.55861,0.73190,0.90486])
s_sweep87=array([0.0011,0.0010,0.0011,0.0012])

muh=1.3996 #mu/h in units of MHz/gauss
gf85=1./2.
gf87=1./3.
B_res85=freq_sweep85/gf85/muh
B_res87=freq_sweep87/gf87/muh

def lin_fit(x, a, b):
    return a+b*x #Our linear function.

x0=array([-10.,1.]) #Intial guess

#calls the solver and puts the solution into the array solution, the uncertainties are put into the
#array covar. 
solution,covar= optimization.curve_fit(lin_fit, B_res85, I_sweep85, x0, s_sweep85)
cal_slope85=1./solution[1]
s_cal_slope85=sqrt((1./solution[1]**2)*covar[1,1])
cal_intercept85=-solution[0]/solution[1]
s_cal_intercept85= sqrt(((solution[0]/solution[1]**2)**2)*covar[1,1]+((1./solution[1]**2)*covar[0,0]))
print "cal_intercept85= %f +/- %f " % (cal_intercept85,s_cal_intercept85)
print "cal_slope85b= %f +/- %f " %(cal_slope85,s_cal_slope85)
plt.xlabel('Magnetic field (gauss)')
plt.ylabel('Current (amps)')
plt.errorbar(B_res85, I_sweep85, fmt = 'rx', yerr =s_sweep85,label='Data') 
plt.plot(B_res85,lin_fit(B_res85,solution[0],solution[1]),label='Fit')
plt.legend(loc='upper right')
plt.show()
chisqr=numpy.sum((I_sweep85-lin_fit(B_res85,solution[0],solution[1]))/s_sweep85)**2
dof=len(I_sweep85)-len(x0)
GOF = 1. - chi2.cdf(chisqr,dof)
print "Reduced Chi Square=%f"%(chisqr/dof)
print "Probability to obtain a higher Chi Square=%f"%(GOF)

x0=array([-10.,1.]) #Intial guess

#calls the solver and puts the solution into the array solution, the uncertainties are put into the
#array covar. 
solution,covar= optimization.curve_fit(lin_fit, B_res87, I_sweep87, x0, s_sweep87)
cal_intercept87=-solution[0]/solution[1]
s_cal_intercept87= sqrt(((solution[0]/solution[1]**2)**2)*covar[1,1]+((1./solution[1]**2)*covar[0,0]))
print "cal_intercept87= %f +/- %f " % (cal_intercept87,s_cal_intercept87)
print "cal_slope87b= %f +/- %f " %(cal_slope87,s_cal_slope87)

plt.xlabel('Magnetic field (gauss)')
plt.ylabel('Current (amps)')
plt.errorbar(B_res87, I_sweep87, fmt = 'rx', yerr =s_sweep87,label='Data') 
plt.plot(B_res87,lin_fit(B_res87,solution[0],solution[1]),label='Fit')
plt.legend(loc='upper right')
plt.show()
chisqr=numpy.sum((I_sweep87-lin_fit(B_res87,solution[0],solution[1]))/s_sweep87)**2
dof=len(I_sweep87)-len(x0)
GOF = 1. - chi2.cdf(chisqr,dof)
print "Reduced Chi Square=%f"%(chisqr/dof)
print "Probability to obtain a higher Chi Square=%f"%(GOF)

Sweep_cal_intercept=(cal_intercept85+cal_intercept87)/2
s_Sweep_cal_intercept=sqrt(s_cal_intercept85**2+s_cal_intercept87**2)
Sweep_cal_slope=(cal_slope85+cal_slope87)/2
s_Sweep_cal_slope=sqrt(s_cal_slope85**2+s_cal_slope87**2)

def B_cal(I,s_I):
    B=lin_fit(I,Sweep_cal_intercept,Sweep_cal_slope)
    s_B=sqrt(I**2*s_Sweep_cal_slope**2+Sweep_cal_slope**2*s_I**2+s_Sweep_cal_intercept**2)
    return (B,s_B)

I_Sweep_Horz_Cal85=array([0.430115,0.24649,0.16479,0.10695,0.06498,0.032055,0.22337,0.055625])
s_I_Sweep_Horz_Cal85=array([0.0012,0.0005,0.0011,0.0011,0.0014,0.0016,.002,0.0018])
Sweep_B, s_Sweep_B=B_cal(I_Sweep_Horz_Cal85,s_I_Sweep_Horz_Cal85)

Sweep_B

s_Sweep_B

I_Horz_Cal85=array([0.026572,0.047042,0.08556,0.10607,0.12552,0.14438,0.14759,0.17557])
s_I_Horz_Cal85=array([2.8e-6,2.8e-6,2.8e-6,2.8e-6,2.8e-6,2.8e-6,2.8e-6,2.8e-6])
freq_Horz_Cal85=array([.25,.3,.5,.6,.7,.8,.9,1.0])

B_res_Horz85=freq_Horz_Cal85/gf85/muh

B_Horz85=B_res_Horz85-Sweep_B

x0=array([-10.,1.]) #Intial guess

#calls the solver and puts the solution into the array solution, the uncertainties are put into the
#array covar. 
solution,covar= optimization.curve_fit(lin_fit, I_Horz_Cal85, B_Horz85, x0, s_Sweep_B)
Horz_cal_slope85=solution[1]
s_Horz_cal_slope85=sqrt(covar[1,1])
Horz_cal_intercept85=solution[0]
s_Horz_cal_intercept85= sqrt(covar[0,0])
print "cal_intercept85= %f +/- %f " % (Horz_cal_intercept85,s_Horz_cal_intercept85)
print "cal_slope85b= %f +/- %f " %(Horz_cal_slope85,s_Horz_cal_slope85)
plt.ylabel('Magnetic field (gauss)')
plt.xlabel('Current (amps)')
plt.errorbar( I_Horz_Cal85, B_Horz85,fmt = 'rx', yerr =s_Sweep_B,label='Data') 
plt.plot(I_Horz_Cal85,lin_fit(I_Horz_Cal85,solution[0],solution[1]),label='Fit')
plt.legend(loc='upper right')
plt.show()
chisqr=numpy.sum((B_Horz85-lin_fit(I_Horz_Cal85,solution[0],solution[1]))/s_Sweep_B)**2
dof=len(I_Horz_Cal85)-len(x0)
GOF = 1. - chi2.cdf(chisqr,dof)
print "Reduced Chi Square=%f"%(chisqr/dof)
print "Probability to obtain a higher Chi Square=%f"%(GOF)

def BR_eqn(S,L,F,J,I,m,B):
    A=2.26435e-24 #hyperfine splitting for ground state of Rb87 Steck
    Ehfs=A*(I+1./2.)
    gJ=1+(J*(J+1)+S(S+1)-L(L+1))/(2*J*(J+1))
    gF=gJ*(F*(F+1)-I(I+1)+J(J+1))/(2*F*(F+1))
    gI=-0.000995141
    x=(gJ-gI)*muB*B/Ehfs
    if F-I>0:
        Eng=-Ehfs/(2*(2*I+1))+gI*ub*m*B+Ehfs/2*sqrt(1+4*M*x/(2*I+1)+x**2)
        

I=3./2
J=1./2
S=1./2
L=0
F=1
B=7e-4
muB=9.724e-24
A=2.26435e-24 #hyperfine splitting for ground state of Rb87 Steck
Ehfs=A*(I+1./2.)
gJ=1+(J*(J+1)+S*(S+1)-L*(L+1))/(2*J*(J+1))
gF=gJ*(F*(F+1)-I*(I+1)+J*(J+1))/(2*F*(F+1))
gI=-0.000995141
x=(gJ-gI)*muB*B/Ehfs

x



