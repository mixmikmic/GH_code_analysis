import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from lsrtm_1d import lsrtm

from wave_1d_fd_pml import test_wave_1d_fd_pml

N=50
dt=0.0005
dx=5
model=test_wave_1d_fd_pml.model_one(N=N)
model['sx']=[1]
v0=2500
model['model']=np.float32(np.ones(N)*v0)
model['model'][-int(N/2):] += np.float32(1000*(np.random.rand(int(N/2))-0.5))
h=3
model['model']=np.float32(np.convolve(np.ones(h)/h, np.pad(model['model'], int((h-1)/2), 'edge'), 'valid'))
h=11
migmodel=1/np.float32(np.convolve(np.ones(h)/h, np.pad(1/model['model'], int((h-1)/2), 'edge'), 'valid'))

plt.plot(model['model'], label='True model')
plt.plot(migmodel, label='Migration model')
plt.ylabel('Wave speed (m/s)');
plt.xlabel('Depth (cells)');
plt.legend(loc=2);

r=lsrtm.Lsrtm(dx, dt, pml_width=20)

maxt = 2*N*dx/v0
nmute = int(maxt/dt/2)
print(maxt, nmute)

d=r.model_shot(model['model'], model['sources'][0], model['sx'][0], model['sx'], maxt)
d[0, :nmute] = 0

plt.plot(d.reshape(-1))
plt.xlabel('Time (time steps)')
plt.ylabel('Amplitude')

costa, jaca = r.migrate_shot(migmodel, model['sources'][0], model['sx'][0], d, model['sx'], maxt, manual_check_grad=True)

plt.plot(costa, label='Finite difference')
plt.plot(jaca, label='Adjoint')
plt.xlabel('Depth (cells)');
plt.ylabel('Gradient amplitude');
plt.legend();

r.adjoint_test(migmodel, model['sources'][0], int(maxt/dt))

-2*15000/2500**3

res=r.migrate_shot(migmodel, model['sources'][0], model['sx'][0], d, model['sx'], 200)

res

plt.plot(res.x)
plt.xlabel('Depth (cells)')
plt.ylabel('Inverted Delta c amplitude')

plt.plot(migmodel, label='Migration model')
plt.plot(migmodel-res.x*migmodel**3/2, label='Inverted model')
plt.plot(model['model'], label='True model')
plt.xlabel('Depth (cells)');
plt.ylabel('Wave speed (m/s)');
plt.legend(loc=2);

