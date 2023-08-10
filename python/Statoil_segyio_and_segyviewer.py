import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().system('ls ../data/*.sgy')

import segyio

fname = "../data/Penobscot_0-1000ms.sgy"
with segyio.open(fname) as f:
    data = np.stack(np.copy(i) for i in f.iline)

data.shape

tslice = 122
inline = 250
xline = 300

f = plt.figure(figsize=(12,4), facecolor='white')

params = dict(aspect='auto', cmap='viridis', vmin=-8000, vmax=8000)

# Plot timeslice
ax0 = f.add_axes([0.05, 0.05, 0.35, 0.90])
ax0.imshow(data[:, :, tslice], **params)
ax0.set_title('timeslice at index {}'.format(tslice))
ax0.axhline(xline, c='w', alpha=0.4)
ax0.axvline(inline, c='w', alpha=0.4)

# Plot inline
ax1 = f.add_axes([0.45, 0.05, 0.2, 0.90])
ax1.imshow(data[:, inline, :].T, **params)
ax1.set_title('inline {}'.format(inline))
ax1.axhline(tslice, c='w', alpha=0.4)

# Plot xline 
ax2 = f.add_axes([0.70, 0.05, 0.3, 0.90])
ax2.imshow(data[xline, :, :].T, **params)
ax2.set_title('xline {}'.format(xline))
ax2.axhline(tslice, c='w', alpha=0.4)

plt.show()

from PyQt4.QtGui import QApplication
import segyviewlib
qapp = QApplication([])
l = segyviewlib.segyviewwidget.SegyViewWidget(fname)
l.show()



