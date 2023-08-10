# Import the required packages
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

# Make the display easy to read
get_ipython().magic('precision 2')

# Enter the information for the canalas into a Python dictionary.
# The data are taken from Blanks RH, Curthoys IS, and Markham CH. Planar relationships of
# the semicircular canals in man. Acta Otolaryngol  80:185-196, 1975.
Canals = {'info': 'The matrix rows describe horizontal, anterior, and posterior canal orientation',
 'right': np.array(
        [[0.365,  0.158, -0.905], 
         [0.652,  0.753, -0.017],
         [0.757, -0.561,  0.320]]),
 'left': np.array(
        [[-0.365,  0.158,  0.905],
         [-0.652,  0.753,  0.017],
         [-0.757, -0.561, -0.320]])}

# Normalize these vectors (just a small correction):
for side in ['right', 'left']:
    Canals[side] = (Canals[side].T / np.sqrt(np.sum(Canals[side]**2, axis=1))).T

# Show the results for the right SCCs:
print(Canals['info'])
print(Canals['right'])

omega = np.r_[0, 0, -100]
stim = omega @ Canals['right'][0]

# Before Python 3.5:
# stim = np.dot(omega, Canals['right'][0])

print('The angular velocity sensed by the right horizontal canal is {0:3.1f} deg/s.'.format(stim))



