import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

cwd = os.getcwd()
data = pd.read_csv(os.path.join(cwd, 'output_example.txt'), header=0)

data.head()

plt.plot(data['throttle'], 'b')
plt.plot(data['brake'], 'g')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.1))
plt.title('Ground-Truth Throttle and Braking')

plt.plot(np.zeros_like(data['brake']), 'b')
plt.plot(data['brake'], 'g')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.1))
plt.title('Panic!!')

plt.figure(figsize=(10, 6))
plt.title('Black: Vt | Blue: Vt+1 | Red: At | Purple: Projection direction of At onto Vt+1')

plt.arrow(0.0, 0.0, 0.2, 0.4, head_width=0.02, head_length=0.02, fc='k', ec='k')
plt.arrow(0.2, 0.4, 0.6, 0.2, head_width=0.02, head_length=0.02, fc='r', ec='r')
plt.arrow(0.0, 0.0, 0.8, 0.6, head_width=0.02, head_length=0.02, fc='b', ec='b')
plt.arrow(0.2, 0.4, 0.06, -0.2, linestyle='--', fc='r', ec='purple')

def accel_from_velocity(velocities):
    """
    Estimates the forward/backward acceleration applied to the car from the velocity vectors.
    
    Takes in a numpy array with shape [n_obs, 2], with the first column being the X position of the velocity and the 
    second column being it's Y position. Returns a 1D numpy array with the estimated forward/backward acceleration 
    values.
    """
    n_obs = velocities.shape[0]
    
    # Calculate all of the raw accelerations at once using array slices.
    acceleration_vectors = velocities[1:] - velocities[:(n_obs-1)]
    
    # Calculate the magnitudes of the velocity vectors
    velocity_magnitudes = []
    for v1, v2 in zip(velocities[:, 0], velocities[:, 1]):
        # Make the magnitude of the velocity vector Infinity if it is close to zero.
        # This is to resolve division by zero errors, and will make the velocity vector have length
        # zero rather than one.
        if abs(v1) < 1e-5 and abs(v2) < 1e-5:
            velocity_magnitudes.append(np.inf)
        else:
            velocity_magnitudes.append(np.sqrt(v1**2 + v2**2))

    velocity_magnitudes = np.array(velocity_magnitudes)

    # Divide the original velocity vectors by their lengths to make them unit length.
    normalized_velocities = velocities[1:] / velocity_magnitudes[1:].reshape(-1, 1)

    # Get the scalar projection of the raw acceleration vectors onto the unit-length velocity vectors. 
    normalized_acceleration = np.array([normalized_velocities[k, :] @ acceleration_vectors[k, :] for k in range(n_obs-1)])
    
    # Zero out any accelerations that are too large.
    normalized_acceleration[normalized_acceleration > 1] = 0
    
    # Append a zero to the start of the array to make it the same length as the dataset. We lost an observation when
    # we differenced the velocities.
    return np.insert(normalized_acceleration, 0, [0], axis=0)


def scale_between_zero_and_one(array):
    """
    Utility function to scale an array's values between zero and one.
    """
    return (array - np.min(array)) / (np.max(array) - np.min(array))

velocities = data[['vel0', 'vel1']].as_matrix()
projected_accelerations = accel_from_velocity(velocities)

plt.figure(figsize=(10, 6))
plt.plot(projected_accelerations, 'r')
plt.plot(data['throttle'], 'b')
plt.plot(data['brake'], 'g')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.1))

interp_throttle = np.copy(projected_accelerations)
interp_throttle[interp_throttle < 0] = 0
interp_throttle = scale_between_zero_and_one(interp_throttle)

interp_brake = np.copy(projected_accelerations)
interp_brake[interp_brake > 0] = 0
interp_brake *= -1  # Make the braking positive
interp_brake = scale_between_zero_and_one(interp_brake)

plt.plot(interp_throttle, 'r')
plt.plot(data['throttle'], 'b')

plt.plot(interp_brake, 'r')
plt.plot(data['brake'], 'b')



