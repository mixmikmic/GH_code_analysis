get_ipython().magic('pylab inline')
import pysd
model = pysd.read_vensim('../../models/Roessler_Chaos/roessler_chaos.mdl')
res = model.run()

import seaborn
plt.plot(res['x'], res['y']);

import mpld3
plt.plot(res['x'], res['y']);
mpld3.display()

