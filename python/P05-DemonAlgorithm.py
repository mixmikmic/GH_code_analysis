get_ipython().magic('pylab inline')

#
# rand() returns a single random number:
#

print(rand())

#
# hist plots a histogram of an array of numbers
#

print(hist(normal(size=1000)))

m=28*1.67e-27  # mass of a molecule (e.g., Nitrogen)
g=9.8          # grav field strength
kb=1.67e-23    # boltzman constant
demonE = 0.0   # initial demon energy
N=10000        # number of molecules
M=400000       # number of iterations
h=20000.0      # height scale

def setup(N=100,L=1.0):
    y=L*rand(N)     # put N particles at random heights (y) between 0 and L
    return y

yarray = setup(N=1000,L=2.0)
hist(yarray)

def shake(y, demonE, delta=0.1):
    """
    Pass in the current demon energy as an argument.
    delta is the size of change in y to generate, more or less.
    randomly choose a particle, change it's position slightly (around delta)
    return the new demon energy and a boolean (was the change accepted?)
    """
    ix = int(rand()*len(y))
    deltaY = delta*normal()
    deltaE = deltaY*m*g
    accept=False
    if deltaE < demonE and (y[ix]+deltaY>0):
        demonE -= deltaE  # take the energy from the demon, or give it if deltaE<0.
        y[ix] += deltaY
        accept=True
        
    return demonE, accept

y = setup(N,L=h)

acceptCount = 0

demonList = []
for i in range(M):
    demonE,accept = shake(y, demonE, delta=0.2*h)
    demonList.append(demonE)
    if accept:
        acceptCount += 1

title("Distribution of heights")
xlabel("height (m)")
ylabel("number in height range")
hist(y,bins=40)
print(100.0*acceptCount/M, "percent accepted")
print("Averge height=%4.3fm" % (y.sum()/len(y),))

#
# Build a histogram of Demon Energies
#

title("Distribution of Demon Energies")
xlabel("Energy Ranges (J)")
ylabel("Number in Energy Ranges")
ns, bins, patches = hist(demonList, bins=60)

#
# Use a "curve fit" to find the temperature of the demon
#

from scipy.optimize import curve_fit

def fLinear(x, m, b):
    return m*x + b

energies = (bins[:-1]+bins[1:])/2.0
xvals = array(energies)  # fit log(n) vs. energy
yvals = log(array(ns))
sig = 1.0/sqrt(array(ns))

#
# make initial estimates of slope and intercept.
#

m0 = (yvals[-1]-yvals[0])/(xvals[-1]-xvals[0])
b0 = yvals[0]-m0*xvals[0]

popt, pcov = curve_fit(fLinear, xvals, yvals, p0=(m0, b0), sigma=sig)

m=popt[0]          # slope
dm=sqrt(pcov[0,0]) # sqrt(variance(slope))
b=popt[1]          # int
db=sqrt(pcov[1,1]) # sqrt(variance(int))
Temp=-1.0/(m*kb)   # temperature
dT = abs(dm*Temp/m)# approx uncertainty in temp

print("slope=", m, "+/-", dm )
print("intercept=", b, "+/-", db)
print("Temperature=", Temp, "+/-", dT, "K")
title("Demon Energy Distribution")
xlabel("Energy (J)")
ylabel("log(n) (number of demon visit to energy)")
errorbar(xvals, yvals, sig, fmt='r.')
plot(xvals,yvals,'b.',label="Demon Energies")
plot(xvals,fLinear(xvals, m, b),'r-', label="Fit")
legend()



