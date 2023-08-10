import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

get_ipython().magic('pinfo stats.norm')

stats.norm.cdf(0.5)

stats.norm.cdf(0.5,loc=0,scale=1)

get_ipython().magic('pinfo stats.poisson')

stats.poisson.pmf(1, mu=0.6)

x = np.linspace(-4, 4, 100)

cdfvals = stats.norm.cdf(x)

plt.plot(x, cdfvals)
plt.ylabel('$F_X(x)$')
plt.xlabel('$x$')
plt.grid(True)

x=np.linspace(0,10,100)
cdfvals=stats.poisson.cdf(x,mu=5)

plt.plot(x, cdfvals)
plt.ylabel('$F_X(x)$')
plt.xlabel('$x$')
plt.grid(True)

x=np.linspace(0,10,100)
cdfvals=stats.poisson.cdf(x,mu=5)

plt.step(x, cdfvals)
plt.ylabel('$F_X(x)$')
plt.xlabel('$x$')
plt.grid(True)

x=range(11)

pmfvals=stats.poisson.pmf(x,mu=5)
plt.stem(x, pmfvals)
plt.ylabel('$P_X(x)$')
plt.xlabel('$x$')
plt.grid(True)

x = np.linspace(-4, 4, 100)
pdfvals = stats.norm.pdf(x)

plt.plot(x, pdfvals)
plt.ylabel('$f_X(x)$')
plt.xlabel('$x$')
plt.grid(True)

p=stats.poisson.rvs(mu=5,size=100000)

np.mean(p)



np.count_nonzero(p==0)

mybins=range(12) #Note to get 11 included in mybins, we need to use 11+1=12 as the limit in the range function

phist=np.histogram(p,bins=mybins)
print(phist)

plt.stem(phist[1][:-1],phist[0])

phist2=phist[0]/np.sum(phist[0])

plt.stem(phist[1][:-1],phist2)

plt.stem(phist[1][:-1],phist2,markerfmt='bo')
#plt.hold(True)
x=range(11)
pmfvals=stats.poisson.pmf(x,mu=5)

# I am slightly offsetting the true PMF values, so that
# both sets of results are visible:
plt.stem(np.array(x)+0.2,pmfvals,'g',markerfmt='gx')

phist=np.histogram(p,bins=mybins,normed=True)

plt.stem(phist[1][:-1],phist[0],markerfmt='bo')
#plt.hold(True)
x=range(11)
pmfvals=stats.poisson.pmf(x,mu=5)

# I am slightly offsetting the true PMF values, so that
# both sets of results are visible:
plt.stem(np.array(x)+0.2,pmfvals,'g',markerfmt='gx')


plt.hist(p,bins=mybins,normed=True)

g = stats.norm.rvs(size=100000) # generate 100000 random numbers with normal distribution

mybins=np.linspace(-4,4,20)

ghist=np.histogram(g,bins=mybins)
print(ghist)

plt.step(ghist[1][:-1],ghist[0])

binwidth=mybins[1]-mybins[0]
totalvals=sum(ghist[0])
c=1/binwidth/totalvals

plt.subplot(211) 
plt.step(ghist[1][:-1],ghist[0]*c)

plt.subplot(212)
x = np.linspace(-4, 4, 100)
pdfvals = stats.norm.pdf(x)

plt.plot(x, pdfvals)
plt.ylabel('$f_X(x)$')
plt.xlabel('$x$')
plt.grid(True)

mybins=np.linspace(-4,4,50)
ghist=np.histogram(g,bins=mybins)


binwidth=mybins[1]-mybins[0]
totalvals=sum(ghist[0])
c=1/binwidth/totalvals

plt.subplot(211) 
plt.step(ghist[1][:-1],ghist[0]*c)

plt.subplot(212)
x = np.linspace(-4, 4, 100)
pdfvals = stats.norm.pdf(x)

plt.plot(x, pdfvals)
plt.ylabel('$f_X(x)$')
plt.xlabel('$x$')
plt.grid(True)

mybins=np.linspace(-4,4,1000)
ghist=np.histogram(g,bins=mybins)


binwidth=mybins[1]-mybins[0]
totalvals=sum(ghist[0])
c=1/binwidth/totalvals

plt.subplot(211) 
plt.step(ghist[1][:-1],ghist[0]*c)

plt.subplot(212)
x = np.linspace(-4, 4, 100)
pdfvals = stats.norm.pdf(x)

plt.plot(x, pdfvals)
plt.ylabel('$f_X(x)$')
plt.xlabel('$x$')
plt.grid(True)

plt.hist(g,bins=50,normed=True,alpha=0.3) # add some transparency
plt.plot(x, pdfvals,'g-',lw=2) # bump up the line width

plt.hist(g,bins=50,normed=True,cumulative=True,alpha=0.3) # add some transparency
cdfvals = stats.norm.cdf(x)


plt.plot(x, cdfvals,'g-',lw=2) # bump up the line width



