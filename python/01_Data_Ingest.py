from pathlib import Path

import gsd.hoomd

# Create a Path object pointing to the directory holding
# all the input files we will be using
directory = Path('data/')
# Search for all files in the above directory which end in .gsd
input_files = directory.glob('*.gsd')

# input files is a generator, essentially a list of files,
# so we can iterate through each individually
for fname in input_files:
    # This is a context manager. We open the file and assign
    # it to the variable traj.
    with gsd.hoomd.open(str(fname)) as trj:
        # This extracts the first (and only) frame of the trajectory
        snap = trj[0]

from collections import namedtuple

sim_params = namedtuple('sim_params', ['temperature', 'pressure', 'moment_inertia', 'crystal'])

def get_sim_params(fname: Path):
    """Extract the simulation parameters from a filename string."""
    # This gives just the filename as a string without directory or extension
    fname = fname.stem
    params = fname.split('-')
    return sim_params(temperature=float(params[2][1:]),
                      pressure=float(params[3][1:]),
                      moment_inertia=float(params[4][1:]),
                      crystal=params[5])

# Check our function works
get_sim_params(fname)

from statdyn.figures.configuration import plot
from bokeh.plotting import show, output_notebook
output_notebook()

show(plot(snap))

import numpy as np

def classify_mols(snapshot, crystal, boundary_buffer=3.5):
    """Classify molecules as crystalline, amorphous or boundary."""
    # This is the method of extracting the positions from a gsd snapshot
    position = snapshot.particles.position
    # This gets the details of the box from the simulation
    box = snapshot.configuration.box
    
    # All axes have to be True, True == 1, use product for logical and operation
    is_crystal = np.product(np.abs(position) < box[:3]/3, axis=1).astype(bool)
    boundary = np.logical_and(np.product(np.abs(position) < box[:3]/3+boundary_buffer, axis=1),
                              np.product(np.abs(position) > box[:3]/3-boundary_buffer, axis=1))
    
    # Create classification array
    classification = np.full(snapshot.particles.N, 'liq', dtype='<U4')
    classification[is_crystal] = crystal
    classification[boundary] = None
    return classification

from statdyn.analysis.order import relative_orientations

def compute_orientations(snapshot):
    """Compute the orientation of 6 nearest neighbours from a gsd snapshot."""
    # I am assuming an orthorhombic simulation cell
    box = snapshot.configuration.box[:3]
    return relative_orientations(box=box,
                                 position=snapshot.particles.position,
                                 orientation=snapshot.particles.orientation,
                                 max_radius=3.5,
                                 max_neighbours=6)

# Check our function works 
compute_orientations(snap)

import pandas

directory = Path('data/')
input_files = directory.glob('*.gsd')
all_dataframes = []

for fname in input_files:
    with gsd.hoomd.open(str(fname)) as trj:
        snap = trj[0]
        # Get simulation parameters
        params = get_sim_params(fname)
        # Classify all molecules
        classes = classify_mols(snap, params.crystal)
        # Compute the orientations
        orientations = compute_orientations(snap)
        
        # Create dataframe
        df = pandas.DataFrame({
            'temperature': params.temperature,
            'pressure': params.pressure,
            'crystal': params.crystal,
            'class': classes,
            'orient0': orientations[:, 0],
            'orient1': orientations[:, 1],
            'orient2': orientations[:, 2],
            'orient3': orientations[:, 3],
            'orient4': orientations[:, 4],
            'orient5': orientations[:, 5],
        })
        
        # Remove molecules close to interface
        df = df[df['class'] != 'None']
        
        all_dataframes.append(df)

# Collate list of dataframes into single large dataframe
training_dataset = pandas.concat(all_dataframes)

# Save dataset to file
training_dataset.to_hdf('data/training_data.h5', key='trimer')

# Check the dataframe contains reasonable data
training_dataset.head()

