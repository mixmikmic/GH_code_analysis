import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')
plt.style.use('notebook');
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
colors = ["#2078B5", "#FF7F0F", "#2CA12C", "#D72827", "#9467BE", "#8C574B",
            "#E478C2", "#808080", "#BCBE20", "#17BED0", "#AEC8E9", "#FFBC79", 
            "#98E08B", "#FF9896", "#C6B1D6", "#C59D94", "#F8B7D3", "#C8C8C8", 
           "#DCDC8E", "#9EDAE6"]

# We defined these functions last time
def Σ(σ,q):
    '''Compute the Σ function needed for linear fits.'''
    return np.sum(q/σ**2)

def get_a(x,y,σ):
    '''Get the χ^2 best fit value of a0 and a1.'''

    # Get the individual Σ values
    Σy,Σx,Σx2,Σ1,Σxy = Σ(σ,y),Σ(σ,x),Σ(σ,x**2),Σ(σ,np.ones(x.size)),Σ(σ,x*y)

    # the denominator
    D = Σ1*Σx2 - Σx**2

    # compute the best fit coefficients
    a = np.array([Σy*Σx2 - Σx*Σxy,Σ1*Σxy - Σx*Σy])/D

    # Compute the error in a
    aErr = np.array([np.sqrt(Σx2/D),np.sqrt(Σ1/D)])

    return a,aErr

def linear(x,a):
    '''Return a polynomial of order'''
    return a[0] + a[1]*x

data = np.array([[  1.00000000e-01,   2.04138220e+00,   9.73629324e-03],
       [  3.57894737e-01,   1.30119078e+00,   1.12856801e-01],
       [  6.15789474e-01,   8.42154689e-01,   9.83121201e-02],
       [  8.73684211e-01,   5.45192601e-01,   1.29185248e-01],
       [  1.13157895e+00,   3.59854509e-01,   2.15797712e-01],
       [  1.38947368e+00,   2.23469107e-01,   2.81326486e-01],
       [  1.64736842e+00,   1.47065865e-01,   3.29001539e-01],
       [  1.90526316e+00,   9.42222066e-02,   4.24678699e-01],
       [  2.16315789e+00,   6.29051329e-02,   4.21412036e-01],
       [  2.42105263e+00,   3.49731098e-02,   6.30215689e-01],
       [  2.67894737e+00,   2.37533207e-02,   5.75886026e-01],
       [  2.93684211e+00,   1.54965698e-02,   7.41655404e-01],
       [  3.19473684e+00,   9.73289991e-03,   8.07876091e-01],
       [  3.45263158e+00,   8.36780173e-03,   8.86926901e-01],
       [  3.71052632e+00,   3.90242054e-03,   9.43666008e-01],
       [  3.96842105e+00,   1.61554262e-03,   1.13278970e+00],
       [  4.22631579e+00,   2.59857424e-03,   1.14161518e+00],
       [  4.48421053e+00,   1.73614348e-03,   1.23615458e+00],
       [  4.74210526e+00,   1.63584150e-03,   1.25628767e+00],
       [  5.00000000e+00,   1.59297834e-03,   1.27049395e+00]])

plt.plot(x,np.log(y1),'o')
plt.plot(x,linear(x,a1),'-')

# get the data
x,y1,y2,σ = data[:,0],data[:,1],data[:,2],np.ones_like(data[:,0])

# plot the data
plt.plot(x,y1,'o',mfc=colors[0], mec='None',label='set 1')
plt.plot(x,y2,'s',mfc=colors[1], mec='None', label='set 2')

# peform the fits
a1,a1_err = get_a(x[:-10],np.log(y1[:-10]),σ[:-10])
a2,a2_err = get_a(np.log(x),np.log(y2),σ)

# plot the fit results
fx = np.linspace(0,5,100)
plt.plot(fx,np.exp(a1[0])*np.exp(a1[1]*fx), color=colors[0], linewidth=1.5, zorder=0, label='set 1 fit')
plt.plot(fx,np.exp(a2[0])*fx**a2[1], color=colors[1],linewidth=1.5, zorder=0, label='set 2 fit')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')

from scipy.optimize import curve_fit

def exp_func(x,*a):
    '''exponential function'''
    return a[0]*np.exp(a[1]*x)

def power_func(x,*a):
    '''power function.'''
    return a[0]*x**a[1]

# perform the fits
a1,a1_cov = curve_fit(exp_func,x,y1,p0=(1,1))
a2,a2_cov = curve_fit(power_func,x,y2,p0=(1,1))

# plot the data
plt.plot(x,y1,'o',mfc=colors[0], mec='None',label='set 1')
plt.plot(x,y2,'s',mfc=colors[1], mec='None', label='set 2')

# plot the fit results
fx = np.linspace(0,5,100)
plt.plot(fx,exp_func(fx,*a1), color=colors[0], linewidth=1.5, zorder=0, label = 'set 1 fit')
plt.plot(fx,power_func(fx,*a2), color=colors[1],linewidth=1.5, zorder=0, label='set 2 fit')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')

def κ_theory(T,*a):
    '''Model for thermal conductivity'''
    return 1.0/(a[0]/T + a[1]*T**2)

# perform the fit
T,κ,σ = np.loadtxt('data/kappa.dat',unpack=True)
a,a_cov = curve_fit(κ_theory,T,κ,sigma=σ,p0=(1,1))

# plot the original data
plt.errorbar(T,κ,yerr=σ, marker='o', mfc=colors[2], mec='w', markersize=8, 
             linestyle='none', capsize=0, elinewidth=1.5, ecolor=colors[2], label='Cu Data')

# plot the line of best fit
fit_T = np.linspace(0.1,60,1000)
plt.plot(fit_T, κ_theory(fit_T,*a), '-', color=colors[2], linewidth=2, 
         zorder=0, label=r'$(a_0/T + a_1 T\ {}^2)^{-1}$')

# add axis labels and legend
plt.xlabel('Temperature [K]')
plt.ylabel('Thermal Conductivity [W/cm K]')
plt.legend()

# fit values
a_err = np.sqrt(np.diag(a_cov))
print('a_0 = %8.2E ± %7.1E' % (a[0],a_err[0]))
print('a_1 = %8.2E ± %7.1E' % (a[1],a_err[1]))

def χ2(x,y,Y,σ=None):
    '''Return the value of χ².'''
    if σ.any():
        return np.sum(((Y-y)/σ)**2)
    else: 
        return np.sum((Y-y)**2)
   
print('κ-fit: χ² = %5.3E' % χ2(T,κ,κ_theory(T,*a),σ))
print(T.size)

def poly_fit(x,*par):
    '''A par.size-1 order polynomial'''
    poly = np.poly1d(par)
    return poly(x)

# perform a pathalogical fit to the data
M = 7
ap,ap_cov = curve_fit(poly_fit,T,κ,p0=np.ones(M))

# plot the original data
plt.errorbar(T,κ,yerr=σ, marker='o', mfc=colors[2], mec='w', markersize=8, 
             linestyle='none', capsize=0, elinewidth=1.5, ecolor=colors[2], label='Cu Data')

# plot the fits
fit_T = np.linspace(0.1,60,1000)
plt.plot(fit_T, κ_theory(fit_T,*a), '-', color=colors[2], linewidth=2, 
         zorder=0, label=r'$(a_0/T + a_1 T\ {}^2)^{-1}$')
plt.plot(fit_T, poly_fit(fit_T,*ap), '--', color=colors[2], linewidth=2, 
         zorder=0, label='Order-%d polynomial'%(M-1))

# add axis labels and legend
plt.xlabel('Temperature [K]')
plt.ylabel('Thermal Conductivity [W/cm K]')
plt.legend()

print('κ-fit      : χ² = %5.3E' % (χ2(T,κ,κ_theory(T,*a),σ)/(T.size-2)))
print('%d-poly-fit : χ² = %5.3E' % (M-1,χ2(T,κ,poly_fit(T,*ap),σ)/(T.size-M)))

from scipy.special import gammaincc
print('κ-fit      : Q = %5.3f' % gammaincc(0.5*(T.size-2), 0.5*χ2(T,κ,κ_theory(T,*a),σ)))
print('%d-poly-fit : Q = %5.3f' % (M-1,gammaincc(0.5*(T.size-M), 0.5*χ2(T,κ,poly_fit(T,*ap),σ))))



