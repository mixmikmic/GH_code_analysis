from __future__ import print_function, absolute_import, division
import ionize
db = ionize.Database()
import numpy as np
from matplotlib import pyplot as plot
get_ipython().magic('matplotlib inline')
np.set_printoptions(precision=3)
import itertools

pKa = [4.75, 7, 9.25]
c = np.logspace(-6, 0)

for valence, extra_pKa, style in [(-1, -2, 'k'), (1, 16, '--k')]:
    for k in pKa + [extra_pKa]:
        ion = ionize.Ion('dummy', [valence], [k], [valence])
        pH = [ionize.Solution(ion, cp).pH for cp in c]
        plot.semilogy(pH, c, style)
        
plot.xlim(2, 12)
plot.xlabel('pH')
plot.ylabel('Concentration (M)')
plot.show()

pKa = [4.75, 7, 9.25]
c = np.logspace(-8, 0)
valence = -1

for k in pKa:
        ion = ionize.Ion('dummy', [valence], [k], [valence])
        pH = [ionize.Solution(ion, cp).pH for cp in c]
        plot.semilogy(pH, c, 'k')

plot.semilogy(-np.log10(c), c, 'k')
plot.semilogy(-np.log10(c), 10*c, '--k')
# plot.Rectangle((0, 1e-3), 3, 100, color='b') # fix area plot
plot.xlim(2, 7)
plot.ylim(1e-8, 1)
plot.xlabel('pH')
plot.ylabel('Concentration (M)')
plot.show()

ratio = np.linspace(0, 1.5)
pKa = 7
conc = 0.01

weak_acid = ionize.Ion('weak acid', [-1], [pKa], [-1])
weak_base = ionize.Ion('weak base', [1], [pKa], [1])
strong_acid = ionize.Ion('strong acid', [-1], [3], [-1])
strong_base = ionize.Ion('strong base', [1], [11], [1])

for weak_ion, strong_ion in ((weak_acid, strong_base), 
                             (weak_base, strong_acid)):
    pH = [ionize.Solution([weak_ion, strong_ion], 
                           [conc, r*conc]).pH for r in ratio]
    pH = np.array(pH)
    plot.plot(ratio, pH - pKa, 'k')
    
plot.xlim(0, 1.5)
plot.ylim(-3.5, 3.5)
plot.xlabel('C_titrant / C_weak')
plot.ylabel('pH - pKa')
plot.show()

tris = db['tris']
acetic = db['acetic acid']
conc = np.linspace(0, .1)
pH = [ionize.Solution([tris, acetic], [.05, c]).pH for c in conc]
plot.plot(conc*1000, pH, 'k')
plot.xlabel('C_a (mM)')
plot.ylabel('Buffer pH')
plot.show()

tris = db['tris']
acetic = db['acetic acid']
strong_acid = ionize.Ion('strong acid', [-1], [-2], [-1])
strong_base = ionize.Ion('strong base', [1], [16], [1])
pH = np.linspace(3, 11)
conc = [.01, .03, .05]

for buff, titrant, style in [(tris, strong_acid, 'k'), 
                             (acetic, strong_base, '--k')]:
    for c in conc:
        cap = []
        for p in pH:
            try:
                sol = ionize.Solution(buff, c).titrate(titrant, p)
                cap.append(sol.buffering_capacity())
            except:
                cap.append(None)
        plot.plot(pH, cap, style)
plot.xlabel('Buffer pH')
plot.ylabel('Buffer Capacity (M)')
plot.show()

H = ionize.Aqueous.henry_CO2(25)
c = .0004 * H
pH = np.linspace(3, 10)
CO2 = db['carbonic acid']

CT = []
C1 = []
C2 = []
I = []
for p in pH:
    f1, f2 = CO2.ionization_fraction(p)
    f0 = 1 - f1 - f2
    CT.append(c / f0)
    C1.append(CT[-1] * f1)
    C2.append(CT[-1] * f2)
    I.append(CT[-1] * CO2.charge(p, moment=2) / 2)

plot.semilogy(pH, CT, 'k')
plot.semilogy(pH, C1, 'k')
plot.semilogy(pH, C2, 'k')
plot.semilogy(pH, I, '--k')

plot.xlabel('pH')
plot.ylabel('c (M)')
plot.ylim(3e-7, 3e-2)
plot.show()

c = np.logspace(-5, 0)
pH = [7, 8, 9]

for p in pH:
    adj = []
    for cp in c:
        try:
            pa = ionize.Solution('tris', cp
                           ).titrate('chloride', p
                                    ).equilibrate_CO2().pH
            adj.append(pa)
        except:
            adj.append(None)
    plot.semilogy(adj, c, 'k')
    
plot.xlabel("pH")
plot.ylabel('c_buffer')
plot.xlim(5.5, 9.5)
plot.ylim(1e-5, 1)
plot.show()  

fl = db['fluorescein']
pH = np.linspace(4, 9)
I = [0, 1e-3, 1e-2, 1e-1, .5]

for Ip in I:
    mob = np.array([fl.mobility(pp, Ip) 
                    for pp in pH])
    plot.plot(pH, -mob, 'k')
    
plot.show()



concentrations = np.linspace(0, 0.14)
ref_mob = 50.e-9
z = [1, 2]
for zp, zm in itertools.product(z, repeat=2):
    positive_ion = ionize.Ion('positive', [zp], [14], [ref_mob])
    negative_ion = ionize.Ion('negative', [-zm], [0], [-ref_mob])
    mob = []
    i = []
    for c in concentrations:
        sol = ionize.Solution([positive_ion, negative_ion], [c/zp, c/zm])
        mob.append(sol.ions[0].actual_mobility() / ref_mob )
        i.append(sol.ionic_strength)
    plot.plot(i, mob, label='-{}:{}'.format(zm, zp))
plot.ylim(0, 1)
plot.xlim(0, .14)
plot.legend(loc='lower left')
plot.xlabel('Concentration (M)')
plot.ylabel('$\mu$/$\mu_o$')
plot.show()

fl = db['fluorescein']
fl= ionize.Ion(reference_mobility=[-3.6000000000000004e-08, -1.9e-08], valence=[-2, -1], alias=None, reference_pKa=[6.8, 4.4], reference_temperature=25.0, name='fluorescein')
I = np.linspace(0, .1)
pH = [9.35, 7.15]

for p in pH:
    mob = np.array([fl.mobility(p, Ip) for Ip in I])
    plot.plot(I, -mob, 'k')

plot.xlim(0, .1)
plot.ylim(1e-8, 4e-8)
plot.xlabel('Ionic Strength (M)')
plot.ylabel('mobility (m^2/V/s)')
plot.show()

analyte = db['alexa fluor 488']
weak = db['histidine']
print(repr(weak))
print(weak.pKa())
LE = ionize.Solution('adenosine diphosphate', .05)
print(LE)
LE.pH

cl = db['chloride']
print(cl.pKa())
c = np.linspace(0, 0.1)
pH = np.array([(LE + (cl, cp)).pH for cp in c])
plot.plot(c, pH)
plot.show()

