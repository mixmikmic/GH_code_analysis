import numpy as np
from wave_1d_fwi_tf import fwi, test_wave_1d_fwi_tf

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

model=test_wave_1d_fwi_tf.model_one()

y = fwi.TFFWI(np.ones(len(model['model']), np.float32)*1500,                                
              model['sources'], model['sx'],                             
              model['data'], model['rx'], model['nsteps'],
              model['dx'], model['dt'])

y.sess.run(y.init)

grad = y.sess.run(y.y_grad)

plt.plot(grad)

pred_y, pred_model=y.invert(nsteps=2000)

plt.plot(pred_y.reshape(-1))
plt.plot(model['data'].reshape(-1))

plt.plot(pred_model.reshape(-1))

