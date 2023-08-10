#importing packages

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from stingray import Lightcurve, Powerspectrum

#opening data
data1 = "/Users/chris/Documents/QPP/SolarFlareGPs/data/120704187_ctime_lc.txt"
t1, I1 = np.loadtxt(data1, unpack=True)

#basic plot
print t1[len(t1)-1]-t1[0]
plt.plot(t1, I1, 'b-')
plt.xlabel("time")
plt.ylabel("counts")
plt.show()

def gauss(x, a, b, c):
    return ((a/np.sqrt(2*np.pi*(c**2)) * np.exp(-1*np.power((x-b),2)/(2*(c**2)))))

#doing a simple fit to the gaussian using a least-squares fit built into scipy
t = t1
popt, pcov = sp.optimize.curve_fit(gauss, t1, I1, p0=[400000, 1200, 600])
Igauss = gauss(t, popt[0], popt[1], popt[2])
plt.plot(t1, I1, 'b-')
plt.xlabel("time")
plt.ylabel("counts")
plt.plot(t, Igauss, '-r')
plt.show()

def lognorm(x, a, b, c):
    return ((a/(x*np.sqrt(2*np.pi*(c**2)))) * np.exp(-1*np.power((np.log(x)-b),2)/(2*(c**2))))

#we can approximate the values from our previous fit, using the form:
a = popt[0]
val = (1+((popt[2]**2)/(popt[1]**2)))
mu = np.log((popt[1])/np.sqrt(val))
sigma = np.sqrt(np.log(val))


#repeating the above step for the lognormal function
popt2, pcov2 = sp.optimize.curve_fit(lognorm, t1, I1, p0=[a, mu, sigma])
Ilognorm = lognorm(t, popt2[0], popt2[1], popt2[2])
plt.plot(t1, I1, 'b-')
plt.plot(t, Ilognorm, '-r')
plt.xlabel("time")
plt.ylabel("counts")
plt.show()

def lorentzian(x, a, b, c):
    return ((a/(np.pi*c))*((c**2)/(np.power((x-b),2)+(c**2))))

#again, repeating the above steps
popt3, pcov3 = sp.optimize.curve_fit(lorentzian, t1, I1, p0=[400000, 1100, 400])
Ilorentz = lorentzian(t, popt3[0], popt3[1], popt3[2])
plt.plot(t1, I1, 'b-')
plt.plot(t, Ilorentz, '-r')
plt.xlabel("time")
plt.ylabel("counts")
plt.show()

def expf(x, a, b):
    return np.exp((a*(x-b)))

def linef(x, a, b):
    return a*x + b

def linef2 (x, x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    b = y2 - (m*x2)
    return ((m*x) + b)


#rough "by eye" approximation of the function using a piecewise exp-linear-exp function
xl = 600
xr = 800
a1 = 2e-2
b1 = 0
a3 = -5.5e-3
b3 = 2900

x1 = np.linspace(0, xl, 500)
x2 = np.linspace(xl, xr, 400)
x3 = np.linspace(xr, 1800, 300)
y1 = expf(x1, a1, b1)
y2 = linef2(x2, xl, expf(xl, a1, b1), xr, expf(xr, a3, b3))
y3 = expf(x3, a3, b3)
plt.ylim(0,450000)
plt.xlabel("time")
plt.ylabel("counts")
plt.plot(t1, I1, 'b-')
plt.plot(x1,y1,'g-')
plt.plot(x2,y2,'c-')
plt.plot(x3,y3,'m-')
plt.show()

#super slow implementation of a piecewise function
def piecewise(x, xl, xr, a1, b1, a3, b3):
            result = np.empty(len(x))
            for i in range(len(x)):
                    if(x[i]<=xl):
                        result[i] = expf(x[i], a1, b1)
                    elif (x[i]>xl and x[i]<=xr):
                        result[i] = linef2(x[i], xl, expf(xl,a1,b1), xr, expf(xr, a3, b3))
                    elif (x[i]>xr):
                        result[i] = expf(x[i], a3, b3)
            return result
        
def hoggmodel(t, tl, tr, al, yl, ar, yr):
    result = np.empty(t.shape)
    for i in range(len(t)):
        if(t[i]<tl):
            result[i] = np.exp(yl + al*(t[i]-tl))
        elif(t[i]>tr):
            result[i] = np.exp(yr + ar*(tr-t[i]))
        else:
            result[i] = np.exp(yl + (t[i]-tl)*(yr-yl)/(tr-tl))
    return result
        
#parameters pulled from the "by eye" fit in prior cell
params = [xl, xr, a1, b1, a3, b3]
hoggparams = [1050, 1500, a1, np.log(410000), 0.2*a1, np.log(260000)]

popt4, pcov = sp.optimize.curve_fit(piecewise, t1, I1, p0=params)
popt5, pcov = sp.optimize.curve_fit(hoggmodel, t1, I1, p0=hoggparams)
print popt4
              
Ipw = piecewise(t, popt4[0], popt4[1], popt4[2], popt4[3], popt4[4], popt4[5])
Ihogg = hoggmodel(t, popt5[0], popt5[1], popt5[2], popt5[3], popt5[4], popt5[5])

plt.ylim(0,450000)
plt.xlabel("time")
plt.ylabel("counts")
plt.plot(t1, I1, 'b-')
plt.plot(t,Ipw,'r-')
#plt.plot(t,Ihogg,'g-')
plt.show()

def dexp(x, xc, al, bl, ar, br):
    result = np.empty(len(x))
    for i in range(len(x)):
            if(x[i]<=xc):
                result[i] = expf(x[i], al, bl)
            elif (x[i]>xc):
                result[i] = expf(x[i], ar, br)
    return result

paramsD = [700, 1.71e-2, 10, -1.6e-2, 1450]
popt6, pcov = sp.optimize.curve_fit(dexp, t1, I1, p0=paramsD)
Idexp = dexp(t1, popt6[0], popt6[1], popt6[2], popt6[3], popt6[4])
print popt6
plt.plot(t1, I1, 'b-')
plt.plot(t1, Idexp, 'r-')
plt.show()

def ctsmodel(t, A, tau1, tau2):
    lam = np.exp(np.sqrt(2*(tau1/tau2)))
    return A*lam*np.exp((-tau1/t)-(t/tau2))

paramsc = [140000, 1000, 1000]
Icts = ctsmodel(t1, paramsc[0], paramsc[1], paramsc[2])

popt7, pcov = sp.optimize.curve_fit(ctsmodel, t1, I1, p0=paramsc)
Ictsf = ctsmodel(t1, popt7[0], popt7[1], popt7[2])
print popt7

plt.plot(t1, I1, 'r-')
plt.plot(t1, Icts, 'b--')
plt.plot(t1, Ictsf, 'g--')
plt.show()

Iarray = np.array([I1, Igauss, Ilognorm, Ilorentz, Ipw, Ihogg])

plt.ylim(0,450000)

plt.xlabel("time")
plt.ylabel("counts")
plt.plot(t1, Iarray[0], 'b-')
plt.plot(t,Iarray[1],'m-', label="Normal")
plt.plot(t,Iarray[2],'c-', label="LogNormal")
plt.plot(t,Iarray[3],'g-', label="Lorentzian")
plt.plot(t,Iarray[4],'y-', label="Piecewise")
#plt.plot(t,Iarray[5],'r-', label="Hogg")
plt.legend()
plt.show()

Iarray = np.array([I1, Igauss, Ilognorm, Ilorentz, Ipw]) #removed hogg
lc = []
ps = []
for i in range(len(Iarray)):
    lc.append(Lightcurve(t1, Iarray[i]))
    ps.append(Powerspectrum(lc[i]))


plt.xlabel("log-frequency")
plt.ylabel("log-intensity")
plt.loglog(ps[0].freq, ps[0].power, 'b-', label="Data")
plt.loglog(ps[1].freq, ps[1].power,'m-', label="Normal")
plt.loglog(ps[2].freq, ps[2].power,'c-', label="LogNormal")
plt.loglog(ps[3].freq, ps[3].power,'g-', label="Lorentzian")
plt.loglog(ps[4].freq, ps[4].power,'y-', label="Piecewise")
#plt.loglog(ps[5].freq, ps[5].power,'r-', label="Hogg")
plt.legend()
plt.show()

Resarray = np.empty([len(Iarray),len(Iarray[0])])
for i in range(len(Iarray)):
    Resarray[i] = Iarray[i] - Iarray[0]

plt.xlabel("time")
plt.ylabel("counts (model - data)")    
plt.plot(t,Resarray[1],'m-', label="Normal")
plt.plot(t,Resarray[2],'c-', label="LogNormal")
plt.plot(t,Resarray[3],'g-', label="Lorentzian")
plt.plot(t,Resarray[4],'y-', label="Piecewise")
#plt.plot(t,Resarray[5],'r-', label="Hogg")
plt.legend()
plt.show()

lcr = []
psr = []
for i in range(1,len(Resarray)):
    lcr.append(Lightcurve(t1, Resarray[i], input_counts=False, err_dist="gauss"))
    psr.append(Powerspectrum(lcr[i-1]))

plt.xlabel("log-frequency")
plt.ylabel("log-intensity (counts-data)")
plt.loglog(ps[0].freq, ps[0].power, 'b-', label="Data")   
plt.loglog(psr[0].freq, psr[0].power,'m-', label="Normal")
plt.loglog(psr[1].freq, psr[1].power,'c-', label="LogNormal")
plt.loglog(psr[2].freq, psr[2].power,'g-', label="Lorentzian")
plt.loglog(psr[3].freq, psr[3].power,'y-', label="Piecewise")
#plt.loglog(psr[4].freq, psr[4].power,'r-', label="Hogg")
plt.legend()
plt.show()

