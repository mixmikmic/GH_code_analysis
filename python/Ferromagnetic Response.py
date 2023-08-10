import numpy as np
from joommf.drivers.evolver import LLG, Minimiser
from joommf.mesh import Mesh
from joommf.sim import Sim
from joommf.energies.demag import Demag
from joommf.energies.zeeman import FixedZeeman
from joommf.energies.exchange import Exchange
import joommf.vectorfield

mesh = Mesh([120e-9, 120e-9, 10e-9], [5e-09, 5e-09, 5e-09])
A = 1.3e-11
gamma = 2.210173e5
dm = 0.01

relax = Sim(mesh, 8.0e5, name='relax', debug=True)
relax.add_energy(Demag())
H_Field = np.array([0.81345856316858023, 0.58162287266553481, 0.0])*8e4
relax.add_energy(FixedZeeman(H_Field))
relax.add_energy(Exchange(A=1.3e-11))
relax.set_evolver(LLG(t=5e-9,
                      m_init = [0, 0, 1],
                      Ms = 8.0e5,
                      alpha = 1.0,
                      gamma = 2.210173e5,
                      dm=0.01,
                      name=relax))
relax.run()

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.plot(relax.df.time, relax.df.mx)
plt.plot(relax.df.time, relax.df.my)
plt.plot(relax.df.time, relax.df.mz)

filename = relax.final_mag

dynamic = Sim(mesh, 8.0e5, name='dynamic', debug=True)
dynamic.add_energy(Demag())
H_Field = np.array([0.81345856316858023, 0.57346234436332832, 0.0])*8e4
dynamic.add_energy(FixedZeeman(H_Field))
dynamic.add_energy(Exchange(A=1.3e-11))
dynamic.set_evolver(LLG(t=5e-12,
                      m_init = filename,
                      Ms = 8.0e5,
                      alpha = 0.008,
                      gamma = 2.210173e5,
                      name=relax))
dynamic.run(stages=4000)

plt.plot(dynamic.df.time, dynamic.df.mx)
plt.xlim([0,.8e-8])

dynamic.final_mag

field = joommf.vectorfield.VectorField(dynamic.final_mag)

plt.quiver(field.z_slice(0.5)[0], field.z_slice(0.5)[1])



