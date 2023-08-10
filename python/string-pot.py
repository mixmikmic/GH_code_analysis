get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

mpl.style.use('mitch-exp')

data = pd.read_csv('pot_calibrate_05_11.csv')

bm_cal = data.set_index('bm_measure')['bm_read'].dropna()
sk_cal = data.set_index('sk_measure')['sk_read'].dropna()
bk_cal = data.set_index('bk_measure')['bk_read'].dropna()

bm_cal, sk_cal, bk_cal



bm_cal.name = 'Boom Pot'
sk_cal.name = 'Stick Pot'
bk_cal.name = 'Bucket Pot'

(bm_cal / 4096.0).plot(linestyle='--', marker='o')
(sk_cal / 4096.0).plot(linestyle=':', marker='D')
(bk_cal / 4096.0).plot(marker='v', markersize=7)
plt.xlabel('Measured Actuator Displacement (mm)')
plt.ylabel('Potentiometer Voltage')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_xbound(0, 150)
plt.tight_layout()
plt.legend()
plt.savefig('figs/pot_calibration.pdf')

', '.join([a for a in str(np.array([bk_cal.values, bk_cal.index.values])).split(' ') if a is not ''])



