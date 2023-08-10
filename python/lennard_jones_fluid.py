import os
import numpy as np

import readdy
print(readdy.__version__)

def average_across_first_axis(values):
    n_values = len(values)
    mean = np.sum(values, axis=0) / n_values  # shape = n_bins
    difference = values - mean  # broadcasting starts with last axis
    std_dev = np.sqrt(np.sum(difference * difference, axis=0) / n_values)
    std_err = np.sqrt(np.sum(difference * difference, axis=0) / n_values ** 2)
    return mean, std_dev, std_err


def lj_system(edge_length, temperature=1.):
    system = readdy.ReactionDiffusionSystem(
        box_size=[edge_length, edge_length, edge_length],
        unit_system=None
    )
    system.kbt = temperature
    system.add_species("A", diffusion_constant=1.)
    system.potentials.add_lennard_jones("A", "A", m=12, n=6, epsilon=1., sigma=1., cutoff=4., shift=True)
    return system

def equilibrate_and_measure(density=0.3):
    n_per_axis = 12
    n_particles = n_per_axis ** 3
    edge_length = (n_particles / density) ** (1. / 3.)
    pos_x = np.linspace(-edge_length/2., edge_length/2.-1., n_per_axis)
    pos = []
    for x in pos_x:
        for y in pos_x:
            for z in pos_x:
                pos.append([x,y,z])
    pos = np.array(pos)
    print("n_particles", len(pos), "box edge_length", edge_length)
    assert len(pos)==n_particles
    
    def pos_callback(x):
        nonlocal pos
        n = len(x)
        pos = np.zeros((n,3))
        for i in range(n):
            pos[i][0] = x[i][0]
            pos[i][1] = x[i][1]
            pos[i][2] = x[i][2]
        print("saved positions")
    
    # create system
    system = lj_system(edge_length, temperature=3.)
    
    # equilibrate
    sim = system.simulation(kernel="CPU")
    sim.add_particles("A", pos)

    sim.observe.particle_positions(2000, callback=pos_callback, save=None)
    sim.observe.energy(500, callback=lambda x: print(x), save=None)

    sim.record_trajectory(stride=1)
    sim.output_file = "lj_eq.h5"
    if os.path.exists(sim.output_file):
        os.remove(sim.output_file)

    sim.run(n_steps=10000, timestep=1e-4)

    traj = readdy.Trajectory(sim.output_file)
    traj.convert_to_xyz(particle_radii={"A": 0.5})
    
    # measure
    sim = system.simulation(kernel="CPU")
    sim.add_particles("A", pos)
    sim.observe.energy(200)
    sim.observe.pressure(200)
    sim.observe.rdf(
        200, bin_borders=np.linspace(0.5, 4., 50),
        types_count_from="A", types_count_to="A", particle_to_density=density)

    sim.output_file = "lj_measure.h5"
    if os.path.exists(sim.output_file):
        os.remove(sim.output_file)

    sim.run(n_steps=10000, timestep=1e-4)
    
    # obtain results
    traj = readdy.Trajectory(sim.output_file)
    _, energy = traj.read_observable_energy()
    _, bin_centers, rdf = traj.read_observable_rdf()
    _, pressure = traj.read_observable_pressure()
    
    energy_mean, _, energy_err = average_across_first_axis(energy) # time average
    energy_mean /= n_particles
    energy_err /= n_particles

    pressure_mean, _, pressure_err = average_across_first_axis(pressure) # time average

    rdf_mean, _, rdf_err = average_across_first_axis(rdf) # time average

    return {
        "energy_mean": energy_mean, "energy_err": energy_err,
        "pressure_mean": pressure_mean, "pressure_err": pressure_err,
        "rdf_mean": rdf_mean, "rdf_err": rdf_err, "rdf_bin_centers": bin_centers
    }

result_low_dens = equilibrate_and_measure(density=0.3)

result_low_dens

result_hi_dens = equilibrate_and_measure(density=0.6)

result_hi_dens

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

print("density 0.3:")
print("mean energy per particle {}\nerr energy per particle {}".format(
    result_low_dens["energy_mean"], result_low_dens["energy_err"]))
print("pressure {}\nerr pressure {}".format(
    result_low_dens["pressure_mean"], result_low_dens["pressure_err"]))
print("density 0.6:")
print("mean energy per particle {}\nerr energy per particle {}".format(
    result_hi_dens["energy_mean"], result_hi_dens["energy_err"]))
print("pressure {}\nerr pressure {}".format(
    result_hi_dens["pressure_mean"], result_hi_dens["pressure_err"]))

plt.plot(result_low_dens["rdf_bin_centers"], result_low_dens["rdf_mean"], label=r"density $\rho=0.3$")
plt.plot(result_hi_dens["rdf_bin_centers"], result_hi_dens["rdf_mean"], label=r"density $\rho=0.6$")
plt.xlabel(r"distance $r/\sigma$")
plt.ylabel(r"radial distribution $g(r)$")
plt.legend(loc="best")
plt.show()

