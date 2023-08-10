import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.constants import G, m_p, k_B
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

small = 3*u.earthRad

NEXSCI_API = 'http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI'
df = pd.read_csv(NEXSCI_API + '?table=planets&select=pl_hostname,pl_letter,pl_disc,ra,dec,pl_trandep,pl_tranflag,pl_orbsmax,pl_radj,pl_bmassj,pl_bmassjerr1,pl_bmassjerr2,pl_eqt,pl_orbper,pl_k2flag,pl_kepflag,pl_facility,st_rad,st_teff,st_optmag,st_j,st_h', comment='#')
df.to_csv('planets.csv')

#We should only do this when there are nan values
df = df[df.pl_tranflag==1].reset_index(drop=True)

nan = ~np.isfinite(df.pl_eqt)
sep = np.asarray(df.pl_orbsmax)*u.AU
rstar = (np.asarray(df.st_rad)*u.solRad).to(u.AU)
temp = np.asarray(df.st_teff)*u.K
df.loc[nan, ['pl_eqt']] = (temp[nan]*np.sqrt(rstar[nan]/(2*sep[nan])))

nan = ~np.isfinite(df.pl_trandep)
df.loc[nan,['pl_trandep']] = ((np.asarray(df.pl_radj[nan])*u.jupiterRad).to(u.solRad)/(np.asarray(df.st_rad[nan])*u.solRad))**2

nan = ~np.isfinite(df.pl_bmassj)
higherrs = df.pl_bmassjerr1/df.pl_bmassj > 0.1
low = (np.asarray(df.pl_radj)*u.jupiterRad).to(u.earthRad)  > 1.5*u.earthRad
high = (np.asarray(df.pl_radj)*u.jupiterRad).to(u.earthRad)  < 4.*u.earthRad
recalculate = np.all([low,high, np.any([nan, higherrs], axis=0)], axis=0)

rade = (np.asarray(df.loc[recalculate,'pl_radj'])*u.jupiterRad).to(u.earthRad).value
df.loc[recalculate, 'pl_bmassj'] = (((2.69* rade)**0.93)*u.earthMass).to(u.jupiterMass).value

mu = 2
g = G * (np.asarray(df.pl_bmassj)*u.jupiterMass)/(np.asarray(df.pl_radj)*u.jupiterRad)**2
g = g.to(u.m/u.second**2)
H = ((k_B*np.asarray(df.pl_eqt)*u.K)/(mu * m_p*g)).to(u.km)

delta = ((H*5) + ((np.asarray(df.pl_radj)*u.jupiterRad).to(u.km)))**2/((np.asarray(df.st_rad)*u.solRad).to(u.km))**2
delta = delta.value - np.asarray(df.pl_trandep)

df['delta'] = delta

#One second scan and a 50 pixel spectrum
exptime = 1
scansize = 50

star_fl = (5.5/0.15)*10.**(-0.4*(df.st_h-15))
#star_fl[star_fl>33000] = 33000
fl = df.delta*star_fl
fl *= scansize * exptime


df['snr'] = fl**0.5


k2 = df[(df.pl_k2flag==1)&(df.pl_facility=='K2')].reset_index(drop=True)
kepler = df[(df.pl_kepflag==1)&(df.pl_facility=='Kepler')].reset_index(drop=True)

fig, ax = plt.subplots()
fl = kepler.snr
h=plt.hist((fl[np.isfinite(fl)]), np.arange(0,20,0.5), normed=True, alpha=0.7, label='Kepler')

fl = k2.snr
plt.hist((fl[np.isfinite(fl)]), h[1], normed=True, alpha=0.7, label='K2')

plt.xlabel('SNR for 5H opaque atmosphere (HST WFC3, 1s exposure)', fontsize=13)
plt.ylabel('Normalized Frequency', fontsize=15)
plt.title('Observability of Exoplanet Atmospheres', fontsize=15)
plt.legend(fontsize=12)
plt.savefig('charts/K2observability.png', dpi=300, bbox_inches='tight')

#Annotations
ok = (np.asarray(k2.pl_radj)*u.jupiterRad).to(u.earthRad) < small
i=0
df1 = k2[ok].sort_values('snr', ascending=False)
for i, n, l, x, y  in zip(range(len(df)), df1.pl_hostname, df1.pl_letter, df1.snr, df1.pl_orbper):
    ann = ax.annotate("{}{}".format(n, l),
                      xy=(x, 0.+i*0.05), xycoords='data',
                      xytext=(x, 0.1+i*0.1), textcoords='data',
                      size=10, va="center", ha="center",
                      bbox=dict(boxstyle="round4", fc="C1", alpha=0.5),
                      arrowprops=dict(arrowstyle="simple",
                                      connectionstyle="arc3, rad=-{}".format(0),
                                      fc="C1"), 
                      )
    i+=1
    if i>=5:
        break
        
plt.savefig('charts/K2observability_annotated.png', dpi=300, bbox_inches='tight')

fig, ax = plt.subplots()
ok = (np.asarray(kepler.pl_radj)*u.jupiterRad).to(u.earthRad) < small
plt.scatter((kepler[ok].snr),kepler[ok].pl_orbper, label='Kepler')

ok = (np.asarray(k2.pl_radj)*u.jupiterRad).to(u.earthRad) < small
plt.scatter((k2[ok].snr),k2[ok].pl_orbper, label='K2')
        
plt.ylabel('Planet Orbital Period', fontsize=15)
plt.ylim(1e-1, 1e3)
plt.xlabel('SNR for 5H opaque atmosphere (HST WFC3, 1s exposure)', fontsize=13)
plt.yscale('log')
plt.legend(fontsize=12, loc='lower left')

#Annotations
i=0
df1 = k2[ok].sort_values('snr', ascending=False)
for i, n, l, x, y  in zip(range(len(df)), df1.pl_hostname, df1.pl_letter, df1.snr, df1.pl_orbper):
    ann = ax.annotate("{}{}".format(n, l),
                      xy=(x, y), xycoords='data',
                      xytext=(x*(1-(i*0.1)), y+(10+(np.exp(i)*10))), textcoords='data',
                      size=10, va="center", ha="center",
                      bbox=dict(boxstyle="round4", fc="C1", alpha=0.5),
                      arrowprops=dict(arrowstyle="simple",
                                      connectionstyle="arc3, rad=-{}".format(0.1*i),
                                      fc="C1"), 
                      )
    i+=1
    if i>=5:
        break

both =kepler.append(k2).reset_index(drop=True)
ok = (np.asarray(both.pl_radj)*u.jupiterRad).to(u.earthRad) < small
top = both[ok][['pl_hostname','pl_letter','pl_eqt','pl_radj','pl_bmassj','pl_orbper','delta','snr','pl_facility','pl_disc']].sort_values('snr', ascending=False)[0:20].reset_index(drop=True)

top

