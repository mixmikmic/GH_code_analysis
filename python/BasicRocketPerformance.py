get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'svg'")
import numpy as np
import operator
import pprint
from matplotlib import pyplot as plt
from IPython.display import Math
import pypropep as ppp

plt.style.use(u'ggplot')
plt.rcParams['figure.figsize'] = (10,6)
ppp.init()

p = ppp.ShiftingPerformance()
o2 = ppp.PROPELLANTS['OXYGEN (GAS)']
ch4 = ppp.PROPELLANTS['METHANE']
p.add_propellants([(ch4, 1.0), (o2, 1.0)])
p.set_state(P=10, Pe=0.01)
print p

for k,v in p.composition.items():  
    print "{} : ".format(k)
    pprint.pprint(v[0:8], indent=4)

OF = np.linspace(1, 5)
m_CH4 = 1.0
cstar_fr = []
cstar_sh = []
Isp_fr = []
Isp_sh = []
for i in xrange(len(OF)):
    p = ppp.FrozenPerformance()
    psh = ppp.ShiftingPerformance()
    
    m_O2 = OF[i]
    
    p.add_propellants_by_mass([(ch4, m_CH4), (o2, m_O2)])
    psh.add_propellants_by_mass([(ch4, m_CH4), (o2, m_O2)])
    
    p.set_state(P=1000./14.7, Pe=1)
    psh.set_state(P=1000./14.7, Pe=1)
    
    cstar_fr.append(p.performance.cstar)
    Isp_fr.append(p.performance.Isp/9.8)
    
    cstar_sh.append(psh.performance.cstar)
    Isp_sh.append(psh.performance.Isp/9.8)

ax = plt.subplot(211)
ax.plot(OF, cstar_fr, label='Frozen')
ax.plot(OF, cstar_sh, label='Shifting')
ax.set_ylabel('C*')
ax1 = plt.subplot(212, sharex=ax)
ax1.plot(OF, Isp_fr, label='Frozen')
ax1.plot(OF, Isp_sh, label='Shifting')
ax1.set_ylabel('Isp (s)')
plt.xlabel('O/F')
plt.legend(loc='best')

kno3 = ppp.PROPELLANTS['POTASSIUM NITRATE']
sugar = ppp.PROPELLANTS['SUCROSE (TABLE SUGAR)']
p = ppp.ShiftingPerformance()
p.add_propellants_by_mass([(kno3, 0.65), (sugar, 0.35)])
p.set_state(P=30, Pe=1.)

for station in ['chamber', 'throat', 'exit']:
    print "{} : ".format(station)
    pprint.pprint(p.composition[station][0:8], indent=4)
    print "Condensed: "
    pprint.pprint(p.composition_condensed[station], indent=4)
    print '\n'

ap = ppp.PROPELLANTS['AMMONIUM PERCHLORATE (AP)']
pban = ppp.PROPELLANTS['POLYBUTADIENE/ACRYLONITRILE CO POLYMER']
al = ppp.PROPELLANTS['ALUMINUM (PURE CRYSTALINE)']
p = ppp.ShiftingPerformance()
p.add_propellants_by_mass([(ap, 0.70), (pban, 0.12), (al, 0.16)])
p.set_state(P=45, Ae_At=7.7)

for station in ['chamber', 'throat', 'exit']:
    print "{} : ".format(station)
    pprint.pprint(p.composition[station][0:8], indent=4)
    print "Condensed: "
    pprint.pprint(p.composition_condensed[station], indent=4)
    print '\n'
print p.performance.Ivac/9.8

p = ppp.ShiftingPerformance()
lh2 = ppp.PROPELLANTS['HYDROGEN (CRYOGENIC)']
lox = ppp.PROPELLANTS['OXYGEN (LIQUID)']

OF = 3
p.add_propellants_by_mass([(lh2, 1.0), (lox, OF)])
p.set_state(P=200, Pe=0.01)

print "Chamber Temperature: %.3f K, Exit temperature: %.3f K" % (p.properties[0].T, p.properties[2].T)
print "Gaseous exit products:"
pprint.pprint(p.composition['exit'][0:8])

print "Condensed exit products:"
pprint.pprint(p.composition_condensed['exit'])

