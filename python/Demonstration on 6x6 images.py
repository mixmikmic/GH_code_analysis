import numpy as np

p_X_given_Z = [0.03, 0.02]  # pretend Gaussian samples, not normalized
p_zik_given_Z_minus = [0.8, 0.2]  # normalized, sum to 1.0

#Then
p_zik_given_X_Z = [0] * 2
for zik in [0, 1]:
    p_zik_given_X_Z[zik] = p_X_given_Z[zik] * p_zik_given_Z_minus[zik]

print(p_zik_given_X_Z)

# normalize
p_zik_given_X_Z /= np.sum(p_zik_given_X_Z)
print('normalized p_zik_given_X_Z', p_zik_given_X_Z)

