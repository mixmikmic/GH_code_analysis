import numpy as np
import scipy.integrate as integrate
def E(z,OmDE,OmM):
    """
    This function computes the integrand for the computation of the luminosity distance for a flat universe
    z -> float
    OmDE -> float
    OmM -> float
    gives
    E -> float
    """
    Omk=1-OmDE-OmM
    return 1/np.sqrt(OmM*(1+z)**3+OmDE+Omk*(1+z)**2)

def dl(z,OmDE,OmM,h=0.7):
    """
    This function computes the luminosity distance
    z -> float
    OmDE -> float
    h ->float
    returns
    dl -> float
    """
    inte=integrate.quad(E,0,z,args=(OmDE,OmM))
    # Velocidad del sonido en km/s
    c = 299792.458
    # Factor de Hubble
    Ho = 100*h
    Omk=1-OmDE-OmM
    distance_factor = c*(1+z)/Ho 
    if Omk>1e-10:
        omsqrt = np.sqrt(Omk)
        return distance_factor / omsqrt * np.sinh(omsqrt * inte[0])
    elif Omk<-1e-10:
        omsqrt = np.sqrt(-Omk)
        return distance_factor / omsqrt * np.sin(omsqrt * inte[0])
    else:
        return distance_factor * inte[0]
        
zandmu = np.loadtxt('../data/SCPUnion2.1_mu_vs_z.txt', skiprows=5,usecols=(1,2))
covariance = np.loadtxt('../data/SCPUnion2.1_covmat_sys.txt')
inv_cov = np.linalg.inv(covariance)
dl = np.vectorize(dl)
def loglike(params,h=0.7):
    """
    This function computes the logarithm of the likelihood. It recieves a vector
    params-> vector with one component (Omega Dark Energy, Omega Matter)
    """
    OmDE = params[0]
    OmM = params[1]
    
# Ahora quiero calcular la diferencia entre el valor reportado y el calculado
    muteo = 5.*np.log10(dl(zandmu[:,0],OmDE,OmM,h))+25
    print dl(zandmu[:,0],OmDE,OmM,h)
    delta = muteo-zandmu[:,1]
    chisquare=np.dot(delta,np.dot(inv_cov,delta))
    return -chisquare/2

loglike([0.6,0.3])

def markovchain(steps, step_width, pasoinicial):
    chain=[pasoinicial]
    likechain=[loglike(chain[0])]
    accepted = 0
    for i in range(steps):
        rand = np.random.normal(0.,1.,len(pasoinicial))
        newpoint = chain[i] + step_width*rand
        liketry = loglike(newpoint)
        if np.isnan(liketry) :
            print 'Paso algo raro'
            liketry = -1E50
            accept_prob = 0
        elif liketry > likechain[i]:
            accept_prob = 1
        else:
            accept_prob = np.exp(liketry - likechain[i])

        if accept_prob >= np.random.uniform(0.,1.):
            chain.append(newpoint)
            likechain.append(liketry)
            accepted += 1
        else:
            chain.append(chain[i])
            likechain.append(likechain[i])
    chain = np.array(chain)
    likechain = np.array(likechain)

    print "Razon de aceptacion =",float(accepted)/float(steps)

    return chain, likechain

chain1, likechain1 = markovchain(100,[0.1,0.1],[0.7,0.3])

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.plot(chain1[:,0],'o')
plt.plot(chain1[:,1],'o')

columna1=chain1[:,0]
np.sqrt(np.mean(columna1**2)-np.mean(columna1)**2)

import seaborn as sns

omegade=chain1[:,0]
omegadm=chain1[:,1]

sns.distplot(omegade)

sns.distplot(omegadm)

sns.jointplot(x=omegadm,y=omegade)

import corner

corner.corner(chain1)

mean_de=np.mean(omegade)
print(mean_de)

dimde=len(omegade)
points_outside = np.int((1-0.68)*dimde/2)
de_sorted=np.sort(omegade)
print de_sorted[points_outside], de_sorted[dimde-points_outside]

dimde=len(omegade)
points_outside = np.int((1-0.95)*dimde/2)
de_sorted=np.sort(omegade)
print de_sorted[points_outside], de_sorted[dimde-points_outside]

