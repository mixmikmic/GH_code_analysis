#NAME: Integration Methods
#DESCRIPTION: Investigation of different numerical integration techniques and their stability in dynamics simulations.

from pycav.mechanics import *
from vpython import *
import numpy as np

class EulerParticle(Particle):
    def update(self, dt):
        self.pos += (self.v * dt)
        self.v += (dt * self.total_force) * self.inv_mass

class MeasuredSystem(System):
    @property
    def kinetic_energy(self):
        _ke = 0.
        for particle in self.particles:
            _ke += (0.5 * np.inner(particle.v, particle.v))/(particle.inv_mass)
        return _ke
    
    @property
    def gravitational_potential_energy(self):
        _gpe = 0.
        for particle1 in self.particles:
            for particle2 in self.particles:
                dist = particle1.pos - particle2.pos
                if particle1 is not particle2:
                    r = np.sqrt(np.inner(dist, dist))
                    _gpe = -1. / (r * particle1.inv_mass * particle2.inv_mass)
        _gpe /= 2.
        return _gpe
    
    @property
    def total_energy(self):
        return (self.gravitational_potential_energy + self.kinetic_energy)

planet_verlet = Particle(pos = np.array([200.,0.,0.]),
                        v = np.array([0., 31.62, 0.]),
                        inv_mass = 1.,
                        make_trail = True,
                        radius = 10.)
star_verlet = Particle(pos = np.array([0., 0., 0.]),
                      v = np.array([0., 0., 0.]),
                      inv_mass = 1./200000.,
                      radius = 20.)

planet_euler = EulerParticle(pos = np.array([200.,0.,0.]),
                        v = np.array([0., 31.62, 0.]),
                        inv_mass = 1.,
                        make_trail = True,
                        color = [1., 1., 1.],
                        radius = 10.)
star_euler = EulerParticle(pos = np.array([0., 0., 0.]),
                      v = np.array([0., 0., 0.]),
                      inv_mass = 1./200000.,
                      radius = 20.)


verlet = [planet_verlet, star_verlet]
euler = [planet_euler, star_euler]

scene1 =  canvas(title='Comparison of integrating methods')
verlet_sys = MeasuredSystem(collides = False, interacts = True, visualize = True, particles = verlet, canvas = scene1)
euler_sys = MeasuredSystem(collides = False, interacts = True, visualize = True, particles = euler, canvas = scene1)

graph1 = graph(x=0, y=0, 
      xtitle='Steps', ytitle='Star-Planet distance', 
      foreground=color.black, background=color.black)
f1 = gcurve(color=color.white)
f2 = gcurve(color=color.red)
dt = 0.1

while True:
    rate(150)
    verlet_sys.simulate(dt)
    euler_sys.simulate(dt)
    dis_verlet = planet_verlet.pos - star_verlet.pos
    dis_euler = planet_euler.pos - star_euler.pos
    f2.plot(verlet_sys.steps, np.sqrt(np.inner(dis_verlet, dis_verlet)))
    f1.plot(verlet_sys.steps, np.sqrt(np.inner(dis_euler, dis_euler)))



