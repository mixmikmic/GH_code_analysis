from simtk import openmm, unit
from simtk.openmm import app
import numpy as np

# Define the parameters of the Lennard-Jones periodic system
nparticles = 512
reduced_density = 0.05
mass = 39.9 * unit.amu
charge = 0.0 * unit.elementary_charge
sigma = 3.4 * unit.angstroms
epsilon = 0.238 * unit.kilocalories_per_mole

# Create a system and add particles to it
system = openmm.System()
for index in range(nparticles):
    # Particles are added one at a time
    # Their indices in the System will correspond with their indices in the Force objects we will add later
    system.addParticle(mass)
    
# Set the periodic box vectors
number_density = reduced_density / sigma**3
volume = nparticles * (number_density ** -1)
box_edge = volume ** (1. / 3.)
box_vectors = np.diag([box_edge/unit.angstrom for i in range(3)]) * unit.angstroms
system.setDefaultPeriodicBoxVectors(*box_vectors)

# Add Lennard-Jones interactions using a NonbondedForce
force = openmm.NonbondedForce()
force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
for index in range(nparticles): # all particles must have parameters assigned for the NonbondedForce
    # Particles are assigned properties in the same order as they appear in the System object
    force.addParticle(charge, sigma, epsilon)
force.setCutoffDistance(3.0 * sigma) # set cutoff (truncation) distance at 3*sigma
force.setUseSwitchingFunction(True) # use a smooth switching function to avoid force discontinuities at cutoff
force.setSwitchingDistance(2.5 * sigma) # turn on switch at 2.5*sigma
force.setUseDispersionCorrection(True) # use long-range isotropic dispersion correction
force_index = system.addForce(force) # system takes ownership of the NonbondedForce object

# construct a Quantity using the multiply operator
bond_length = 1.53 * unit.nanometer
# equivalently using the explicit Quantity constructor
bond_length = unit.Quantity(1.53, unit.nanometer)
# or more verbosely
bond_length = unit.Quantity(value=1.53, unit=unit.nanometer)
# print a unit in human-readable form
print(bond_length)
# or display it in machine-readable form
repr(bond_length)

# Get the number of particles in the System
print('The system has %d particles' % system.getNumParticles())
# Print a few particle masses
for index in range(5):
    print('Particle %5d has mass %12.3f amu' % (index, system.getParticleMass(index) / unit.amu))
# Print number of constraints
print('The system has %d constraints' % system.getNumConstraints())
# Get the number of forces and iterate through them
print('There are %d forces' % system.getNumForces())
for (index, force) in enumerate(system.getForces()):
    print('Force %5d : %s' % (index, force.__class__.__name__))

positions = box_edge * np.random.rand(nparticles,3) 
print(positions)

# Create an integrator
timestep = 1.0 * unit.femtoseconds
integrator = openmm.VerletIntegrator(timestep)
# Create a Context using the default platform (the fastest abailable one is picked automatically)
# NOTE: The integrator is bound irrevocably to the context, so we can't reuse it
context = openmm.Context(system, integrator)

# We have to create a new integrator for every Context since it takes ownership of the integrator we pass it
integrator = openmm.VerletIntegrator(timestep)
# Create a Context using the multithreaded mixed-precision CPU platform
platform = openmm.Platform.getPlatformByName('CPU')
context = openmm.Context(system, integrator, platform)

from copy import deepcopy
# Create a new integrator we can throw away
throwaway_integrator = openmm.VerletIntegrator(timestep)
# Copy the System object for this example
copy_system = deepcopy(system)
throwaway_context = openmm.Context(copy_system, throwaway_integrator)
# Change something in the Python System, e.g. double the box size
copy_system.setDefaultPeriodicBoxVectors(*box_vectors*2)
# Observe how the box size in the Context and Python System are different
print([vector for i, vector in enumerate(copy_system.getDefaultPeriodicBoxVectors())])
print([vector for i, vector in enumerate(throwaway_context.getState().getPeriodicBoxVectors())])
# Cleanup
del throwaway_integrator, throwaway_context

# Set the positions
context.setPositions(positions)
# Retrieve the energy and forces
state = context.getState(getEnergy=True, getForces=True)
potential_energy = state.getPotentialEnergy()
print('Potential energy: %s' % potential_energy)
forces = state.getForces()
print('Forces: %s' % forces[0:5])

# Get forces as numpy array
forces = state.getForces(asNumpy=True)
print('Forces: %s' % forces)

# Compute the energy
state = context.getState(getEnergy=True, getForces=True)
potential_energy = state.getPotentialEnergy()
print('Potential energy: %s' % potential_energy)
# Compute the force magnitude
forces = state.getForces(asNumpy=True)
force_magnitude = (forces**2).sum().sqrt()
print('Force magnitude: %s' % force_magnitude)

# Minimize the potential energy
openmm.LocalEnergyMinimizer.minimize(context)

# Compute the energy and force magnitude after minimization
state = context.getState(getEnergy=True, getForces=True)
potential_energy = state.getPotentialEnergy()
print('Potential energy: %s' % potential_energy)
forces = state.getForces(asNumpy=True)
force_magnitude = (forces**2).sum().sqrt()
print('Force magnitude: %s' % force_magnitude)

# Serialize to a string
system_xml = openmm.XmlSerializer.serialize(system)
# Restore from a string
restored_system = openmm.XmlSerializer.deserialize(system_xml)
assert(system.getNumParticles() == restored_system.getNumParticles())

# Peek at the first 15 lines of the XML file
system_xml.split('\n')[:15]

# Load in an AMBER system
from simtk.openmm import app
prmtop = app.AmberPrmtopFile('resources/alanine-dipeptide-implicit.prmtop')
inpcrd = app.AmberInpcrdFile('resources/alanine-dipeptide-implicit.inpcrd')
system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, implicitSolvent=app.OBC2, constraints=app.HBonds)
positions = inpcrd.getPositions(asNumpy=True)

# Compute the potential energy
def compute_potential(system, positions):
    """Print the potential energy given a System and positions."""
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = openmm.Context(system, integrator)
    context.setPositions(positions)
    print('Potential energy: %s' % context.getState(getEnergy=True).getPotentialEnergy())
    # Clean up
    del context, integrator
    
compute_potential(system, positions)

# Load in a CHARMM system
from simtk.openmm import app
psf = app.CharmmPsfFile('resources/ala_ala_ala.psf')
pdbfile = app.PDBFile('resources/ala_ala_ala.pdb')
params = app.CharmmParameterSet('resources/charmm22.rtf', 'resources/charmm22.par')
system = psf.createSystem(params, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
positions = pdbfile.getPositions(asNumpy=True)
compute_potential(system, positions)

# Load in a gromacs system
from simtk.openmm import app
gro = app.GromacsGroFile('resources/input.gro')
# Make sure you change this to your appropriate gromacs include directory!
gromacs_include_filepath = '/Users/choderaj/miniconda3/share/gromacs/top/'
top = app.GromacsTopFile('resources/input.top', periodicBoxVectors=gro.getPeriodicBoxVectors(), includeDir=gromacs_include_filepath)
system = top.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1*unit.nanometer, constraints=app.HBonds)
positions = gro.getPositions(asNumpy=True)
compute_potential(system, positions)

# Create an OpenMM system using the ForceField class
pdb = app.PDBFile('resources/input.pdb')
forcefield = app.ForceField('amber99sb.xml', 'tip3p.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=1*unit.nanometer, constraints=app.HBonds)
positions = pdb.positions
compute_potential(system, positions)

# Extract the topology from a PDB file
pdbfile = app.PDBFile('resources/alanine-dipeptide-implicit.pdb')
positions = pdbfile.getPositions()
topology = pdbfile.getTopology()

# Write out a PDB file
with open('output.pdb', 'w') as outfile:
    app.PDBFile.writeFile(topology, positions, outfile)

# Create an MDTraj Trajectory object
import mdtraj
mdtraj_topology = mdtraj.Topology.from_openmm(topology)
traj = mdtraj.Trajectory(positions/unit.nanometers, mdtraj_topology)

# View it in nglview
import nglview
view = nglview.show_mdtraj(traj)
view.add_ball_and_stick('all')
view.center_view(zoom=True)
view

from openmmtools import testsystems

t = testsystems.HarmonicOscillator()
system, positions, topology = t.system, t.positions, t.topology

print(positions)
help(t)

# Create a Lennard-Jones cluster with a harmonic spherical restraint:
t = testsystems.LennardJonesCluster()

def visualize(t):
    import mdtraj
    mdtraj_topology = mdtraj.Topology.from_openmm(t.topology)
    traj = mdtraj.Trajectory([t.positions/unit.nanometers], mdtraj_topology)

    # View it in nglview
    import nglview
    view = nglview.show_mdtraj(traj)
    if t.system.getNumParticles() < 2000:
        view.add_ball_and_stick('all')
    view.center_view(zoom=True)
    return view

visualize(t)

t = testsystems.LennardJonesFluid(reduced_density=0.90)
visualize(t)

t = testsystems.WaterBox()
visualize(t)

t = testsystems.AlanineDipeptideVacuum()
visualize(t)

# The Joint AMBER-CHARMM (JAC) DHFR in explicit solvent benchmark system
t = testsystems.DHFRExplicit()
visualize(t)

# Get all subclasses of type `TestSystem`
from openmmtools.tests.test_testsystems import get_all_subclasses
for t in get_all_subclasses(testsystems.TestSystem):
    print(t.__name__)



