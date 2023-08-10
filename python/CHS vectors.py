get_ipython().magic('matplotlib notebook')

import matplotlib.pyplot as plt
import numpy as np

from dcprogs.likelihood import QMatrix

tau = 1e-4
qmatrix = QMatrix([[ -3050,        50,  3000,      0,    0 ], 
                   [ 2./3., -1502./3.,     0,    500,    0 ],  
                   [    15,         0, -2065,     50, 2000 ],  
                   [     0,     15000,  4000, -19000,    0 ],  
                   [     0,         0,    10,      0,  -10 ] ], 2)

from dcprogs.likelihood import MissedEventsG

eG = MissedEventsG(qmatrix, tau)
assert np.all(abs(eG.initial_CHS_occupancies(4e-3) - [0.220418, 0.779582]) < 1e-5)
assert np.all(abs(eG.final_CHS_occupancies(4e-3) - [0.974852, 0.21346, 0.999179]) < 1e-5)
np.set_printoptions(precision=15)

fig = plt.figure()
x = np.arange(0, 5*tau, tau/10)

ax = fig.add_subplot(1, 2, 1) 
ax.plot(x*1e3, [eG.initial_CHS_occupancies(u)[0] for u in x])
ax.set_xlabel('$t_{\mathrm{crit}}$ (ms)')
ax.set_ylabel('Components of the initial CHS vector $\phi_A$')

ax = fig.add_subplot(1, 2, 2) 
ax.plot(x*1e3, [eG.final_CHS_occupancies(u)[0] for u in x])
ax.set_xlabel('$t_{\mathrm{crit}}$ (ms)')
ax.set_ylabel('Components of the final CHS vector $e_F$')
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

fig.tight_layout()

qmatrix = QMatrix([[ -3050,        50,  3000,      0,    0 ], 
                   [ 2./3., -1502./3.,     0,    500,    0 ],  
                   [    15,         0, -2065,     50, 2000 ],  
                   [     0,     15000,  4000, -19000,    0 ],  
                   [     0,         0,    10,      0,  -10 ] ], 2)
qmatrix.matrix /= 1e3
eG = MissedEventsG(qmatrix, 0.2)
print(eG.initial_CHS_occupancies(4))
print(eG.final_CHS_occupancies(4))

qmatrix = QMatrix([[-1, 1, 0], [19, -29, 10], [0, 0.026, -0.026]], 1)
eG = MissedEventsG(qmatrix, 0.2)
print(eG.initial_CHS_occupancies(0.2))
print(eG.final_CHS_occupancies(4))

qmatrix = QMatrix([ [-2,    1,   1,    0], 
                    [ 1, -101,   0,  100], 
                    [50,    0, -50,    0],
                    [ 0,  5.6,   0, -5.6]], 1)
eG = MissedEventsG(qmatrix, 0.2)
print(eG.initial_CHS_occupancies(4))
print(eG.final_CHS_occupancies(4))



