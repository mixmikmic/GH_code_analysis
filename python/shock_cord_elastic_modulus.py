import numpy as np
import matplotlib.pyplot as plt

data = [
    (3866, 1.5),
    (3891, 20.5),
    (3918, 39.5),
    (3900, 20.5),
    (3875, 1.5),
    (3896, 20.5),
    (3919, 39.5),
    (3902, 20.5),
    (3876, 1.5)
]
x, y = zip(*data)
x, y = np.array(x), np.array(y)

# linear fit with np.polyfit
m, b = np.polyfit(x, y, 1)

plt.plot(x, y, '.')
x = np.array([max(x), min(x)])
plt.plot(x, m*x+ b, 'g')
plt.show()

# 1/2 Inch shock cord, 3.866m long
k = 9.81*m*1000
print('{:.3f} [kg/mm]'.format(m))
print('k = {:.0f} [N/m], (1/2Inch, 7m)'.format(k))

# projected k for a shock cord of 1 Inch shock cord
k = k*2*3.866
print('k/l = {:.0f} [N], (1Inch)'.format(k))



