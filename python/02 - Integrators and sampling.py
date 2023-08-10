# Preliminary imports
from simtk import openmm, unit
from simtk.openmm import app
import numpy as np

# Create an alanine dipeptide in vacuum
from openmmtools import testsystems
t = testsystems.AlanineDipeptideVacuum()
system, positions, topology = t.system, t.positions, t.topology

# Create a new integrator since the previously-created integrator was irrevocably bound to the previous Context
temperature = 298.0 * unit.kelvin
collision_rate = 91.0 / unit.picosecond
timestep = 2.0 * unit.femtoseconds
integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
# Create a Context for this integrator
context = openmm.Context(system, integrator)
# Set the positions
context.setPositions(positions)
# Minimize the potential energy
openmm.LocalEnergyMinimizer.minimize(context)

# Set velocities from Maxwell-Boltzmann distribution
context.setVelocitiesToTemperature(temperature)

# Integrate some dynamics
nsteps = 100 # number of integrator steps
integrator.step(nsteps)

# Run a few iterations of a few steps each, reporting potential energy
for iteration in range(10):
    integrator.step(100)
    state = context.getState(getEnergy=True)
    print('%8.3f ps : potential %12.6f kJ/mol' % (state.getTime() / unit.picoseconds, state.getPotentialEnergy() / unit.kilojoules_per_mole))

from sys import stdout
integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
simulation = app.Simulation(topology, system, integrator)
simulation.context.setPositions(positions)
simulation.minimizeEnergy()
simulation.reporters.append(app.StateDataReporter(stdout, 100, step=True, potentialEnergy=True, temperature=True))
simulation.step(1000)

import mdtraj
integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
simulation = app.Simulation(topology, system, integrator)
simulation.context.setPositions(positions)
simulation.minimizeEnergy()
reportInterval = 100
simulation.reporters.append(mdtraj.reporters.HDF5Reporter('output.h5', reportInterval, coordinates=True, time=True, cell=True, potentialEnergy=True, temperature=True))
simulation.reporters.append(mdtraj.reporters.DCDReporter('output.dcd', reportInterval))
simulation.reporters.append(mdtraj.reporters.NetCDFReporter('output.nc', reportInterval))
simulation.step(2000)
del simulation # Make sure to close all files

traj = mdtraj.load('output.h5', 'r')
import nglview
view = nglview.show_mdtraj(traj)
view.add_ball_and_stick('all')
view.center_view(zoom=True)
view

# Define a CompoundIntegrator with high- and low-friction integrators
integrator = openmm.CompoundIntegrator()
integrator.addIntegrator(openmm.LangevinIntegrator(temperature, 91/unit.picosecond, timestep)) # high friction
integrator.addIntegrator(openmm.LangevinIntegrator(temperature, 1/unit.picosecond, timestep)) # low friction
simulation = app.Simulation(topology, system, integrator)
simulation.reporters.append(app.StateDataReporter(stdout, 100, step=True, potentialEnergy=True, temperature=True))
simulation.context.setPositions(positions)
simulation.minimizeEnergy()
# Simulate with high friction
integrator.setCurrentIntegrator(0)
simulation.step(500)
# Simulate with low friction
integrator.setCurrentIntegrator(1)
simulation.step(500)

# Implement velocity Verlet (without constraints) as a CustomIntegrator
integrator = openmm.CustomIntegrator(1.0 * unit.femtoseconds)
integrator.addUpdateContextState() # allow barostat and other updates
integrator.addComputePerDof("v", "v+0.5*dt*f/m")
integrator.addComputePerDof("x", "x+dt*v")
integrator.addComputePerDof("v", "v+0.5*dt*f/m");

# Implement velocity Verlet (without constraints) as a subclass of CustomIntegrator
class UnconstrainedVelocityVerlet(openmm.CustomIntegrator):
    def __init__(self, timestep):
        super(UnconstrainedVelocityVerlet, self).__init__(timestep)
        self.addUpdateContextState()
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("v", "v+0.5*dt*f/m")

# Now use this subclass in a simulation!
integrator = UnconstrainedVelocityVerlet(1.0 * unit.femtoseconds)
simulation = app.Simulation(topology, system, integrator)
simulation.context.setPositions(positions)
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(temperature)
reportInterval = 100
simulation.reporters.append(app.StateDataReporter(stdout, 100, step=True, potentialEnergy=True, temperature=True))
simulation.step(1000)

# Construct an integrator with the VRORV (BAOAB) splitting
from openmmtools import integrators
integrator = integrators.LangevinIntegrator(temperature=temperature, collision_rate=collision_rate, timestep=timestep,
                                           splitting='V R O R V')
simulation = app.Simulation(topology, system, integrator)
simulation.context.setPositions(positions)
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(temperature)
reportInterval = 100
simulation.reporters.append(app.StateDataReporter(stdout, 100, step=True, potentialEnergy=True, temperature=True))
simulation.step(1000)

# Create GHMC integrator based on OVRVO with inner velocity Verlet step Metropolized
integrator = integrators.LangevinIntegrator(temperature=temperature, collision_rate=collision_rate, timestep=timestep,
                                           splitting='O { V R V } O')

simulation = app.Simulation(topology, system, integrator)
simulation.context.setPositions(positions)
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(temperature)
reportInterval = 100
simulation.reporters.append(app.StateDataReporter(stdout, 100, step=True, potentialEnergy=True, temperature=True))
simulation.step(1000)

# We can even pretty-print LangevinIntegrator to inspect what it does
integrator.pretty_print()

for force in system.getForces():
    if force.__class__.__name__ in ['NonbondedForce']:
        force.setForceGroup(1)
    else:
        force.setForceGroup(0)
        
# Create multiple-timestep integrator version of OVRVO
integrator = integrators.LangevinIntegrator(temperature=temperature, collision_rate=collision_rate, timestep=timestep,
                                           splitting='O V1 V0 V0 R V0 V0 V1 O')        

simulation = app.Simulation(topology, system, integrator)
simulation.context.setPositions(positions)
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(temperature)
reportInterval = 100
simulation.reporters.append(app.StateDataReporter(stdout, 100, step=True, potentialEnergy=True, temperature=True))
simulation.step(1000)



