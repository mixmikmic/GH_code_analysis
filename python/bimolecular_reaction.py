import os
import numpy as np
import matplotlib.pyplot as plt
import readdy

system = readdy.ReactionDiffusionSystem([20.,20.,20.], temperature=300.*readdy.units.kelvin)

system.add_species("A", diffusion_constant=1.0)
system.add_species("B", diffusion_constant=1.0)
system.add_species("C", diffusion_constant=1.0)

lambda_on = 1.
system.reactions.add("myfusion: A +(1) B -> C", rate=lambda_on/readdy.units.nanosecond)

simulation = system.simulation(kernel="CPU")
simulation.output_file = "out.h5"
simulation.reaction_handler = "Gillespie"

n_particles = 2000
initial_positions_a = np.random.random(size=(n_particles, 3)) * 20. - 10.
initial_positions_b = np.random.random(size=(n_particles, 3)) * 20. - 10.
simulation.add_particles("A", initial_positions_a)
simulation.add_particles("B", initial_positions_b)

simulation.observe.number_of_particles(stride=1, types=["A"])

if os.path.exists(simulation.output_file):
    os.remove(simulation.output_file)

simulation.run(n_steps=5000, timestep=1e-3*readdy.units.nanosecond)

traj = readdy.Trajectory(simulation.output_file)
time, counts = traj.read_observable_number_of_particles()

kappa = np.sqrt(lambda_on / 2.)
k_on = 4. * np.pi * 2. * 1. * (1. - np.tanh(kappa * 1.) / (kappa * 1.) )

def a(t): 
    return 1. / ((system.box_volume.magnitude / n_particles) + k_on * t)

t_range = np.linspace(0., 5000 * 1e-3, 10000)

plt.plot(time[::200]*1e-3, counts[::200] / system.box_volume.magnitude, "x", label="ReaDDy")
plt.plot(t_range, a(t_range), label=r"analytical $a(t)$")
plt.legend(loc="best")
plt.xlabel("time in nanoseconds")
plt.ylabel(r"concentration in nm$^{-3}$")
plt.show()

