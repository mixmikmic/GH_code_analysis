import holoviews as hv
hv.notebook_extension()

import numpy as np

def create_vectorfield(freq=1, phase=0):

    x,y  = np.mgrid[-10:10,-10:10] * 0.25
    sine_rings  = np.sin(freq * (x**2+y**2 + phase))*np.pi+np.pi
    exp_falloff = 1/np.exp((x**2+y**2)/8)

    vector_data = np.array([x.flatten()/5.,           # X positions
                            y.flatten()/5.,           # Y positions
                            sine_rings.flatten(),     # Arrow angles
                            exp_falloff.flatten()])   # Arrow sizes
    scalar_data = sine_rings 
    return vector_data, scalar_data


matrices = {(phase, freq): hv.VectorField(create_vectorfield(freq, phase)[0].T, 
                                   label='my_label', group='my_group')
          for freq in [0.05, 0.1, 0.25, 0.5, 1.0,  1.5,  2.0]    # Frequencies
          for phase in [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]}  # Phases

get_ipython().run_cell_magic('opts', "VectorField (color='blue')", "\nvector_data, scalar_data = create_vectorfield(freq=0.1)\nhv.VectorField(vector_data.T, label='label', group='group')\n")

# polar angle

hv.Image(scalar_data)

hv.HoloMap(matrices, kdims=['phase', 'frequency'])

matrices2 = {(phase, freq): hv.Image(create_vectorfield(freq, phase)[1]) * 
                                     hv.VectorField(create_vectorfield(freq, phase)[0].T,
                                     label='my_label', group='my_group')                                       
          for freq in [0.1, 0.25, 0.5, 1.0,  1.25, 1.5]    # Frequencies
          for phase in [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]}  # Phases

get_ipython().run_cell_magic('opts', "VectorField (color='r') Image (cmap='gray')", "\n\nhv.HoloMap(matrices2, kdims=['phase', 'frequency'])")

