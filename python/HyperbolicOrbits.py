class ListTable(list):     # this is just to nicely display the table (credit: Caleb Madrigal)
    def _repr_html_(self):
        html = ["<table>"]
        for row in self:
            html.append("<tr>")
            
            for col in row:
                html.append("<td>{0}</td>".format(col))
            
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)

table = ListTable()
# this is what each column represents
table.append(['tracklet_id', 'e', 'q[AU]', 'i[deg]', 'Omega[deg]', 'argperi[deg]', 't_peri[MJD, days]', 'epoch_of_elements[MJD, days]', 'epoch_of_observation[MJD, days]'])

import numpy as np
comets = np.genfromtxt("comets.txt", skip_header=1, dtype=None) # load the table
for comet in comets:
    table.append(comet)
table

import rebound
sim = rebound.Simulation()
k = 0.01720209895 # Gaussian constant
sim.G = k**2

sim.add(m=1.) # Sun
sim.add(m=1.e-3, a=5.) # Jupiter
sim.add(m=3.e-4, a=10.) # Saturn

def getOrbit(comet_elem):
    o = rebound.Orbit()
    o.e = comet_elem[1]
    o.a = comet_elem[2]/(1.-o.e) # q = a(1-e)
    o.inc = comet_elem[3]*np.pi/180. # have to convert to radians
    o.Omega = comet_elem[4]*np.pi/180.
    o.omega = comet_elem[5]*np.pi/180.
    o.T = comet_elem[6] # time of pericenter passage
    return o

comet_elem = comets[0]
sim.t = comet_elem[-1] # last column is the time of observation
o = getOrbit(comet_elem)
sim.add(a=o.a, e=o.e, inc=o.inc, Omega=o.Omega, omega=o.omega, T=o.T)

comet = sim.particles[3]
print(comet.calculate_orbit(primary=sim.particles[0]))

get_ipython().magic('matplotlib inline')
fig = rebound.OrbitPlot(sim, trails=True)

tfinal = 60000 # Final time in days since J2000
sim.integrate(tfinal)
fig = rebound.OrbitPlot(sim, trails=True)

print(comet.orbit)

import rebound
sim = rebound.Simulation()
k = 0.01720209895 # Gaussian constant
sim.G = k**2
sim.add(m=1.) # Sun
sim.add(m=1.e-3, a=5.) # Jupiter
sim.add(m=3.e-4, a=10.) # Saturn
for comet_elem in comets:
    sim.t = comet_elem[-1] # last column is the time of observation
    o = getOrbit(comet_elem)
    sim.add(a=o.a, e=o.e, inc=o.inc, Omega=o.Omega, omega=o.omega, T=o.T)
fig = rebound.OrbitPlot(sim, trails=True)



