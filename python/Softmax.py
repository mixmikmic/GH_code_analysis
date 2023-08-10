import numpy as np
import math
z = np.array([0.9, 0.8, 0.2, 0.1, 0.5])

sum(z)


def softmax(z):
    z_exp = [math.exp(x) for x in z]
    sum_z_exp = sum(z_exp)
    softmax = [round(i / sum_z_exp, 3) for i in z_exp]
    return softmax

softmax(z)

sum(softmax(z))



