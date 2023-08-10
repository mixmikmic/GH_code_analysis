import pypick.pypick
import pypick.predict_picks
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

data = xr.open_dataset('allshots.nc')

num_frames, frame_len, trace_len = data.Samples.shape
frame_params = np.zeros([num_frames, 2], dtype=np.float32)
frame_params[:, 0] = data.FieldRecord
trace_params = np.concatenate([data.offset.values[:,:,np.newaxis],
                               np.abs(data.offset.values[:,:,np.newaxis]),
                               data.GroupX.values[:,:,np.newaxis],
                               data.ReceiverGroupElevation.values[:,:,np.newaxis]],
                              axis=2)

print(frame_params.shape, trace_params.shape)

data.Samples.shape

data['normalised_Samples'] = data.Samples/np.max(np.abs(data.Samples), axis=2)

picks = pypick.pypick.Pypicks(data.normalised_Samples.values, frame_params, trace_params,
                              perform_prediction=False)

picks.pypick()

picks.pypick()

picks.picks[0] # picks for the first frame

ppicks = picks.picks
import pickle
pickle.dump(ppicks, open('picks.pickle', 'wb'))

# I am finished with interactive plotting now
get_ipython().run_line_magic('matplotlib', 'inline')

picks = pypick.pypick.Pypicks(data.normalised_Samples.values, frame_params, trace_params,
                              approx_reg=GridSearchCV(SVR(), {'C': 10**np.arange(0, 4)}),
                              fine_reg=GridSearchCV(pypick.predict_picks.Fine_reg_tensorflow(batch_size=25,
                                                                                             num_steps=5000),
                                                    [{'box_size': [10], 'layer_len': [3, 5]},
                                                     {'box_size': [15], 'layer_len': [3, 5, 10]},
                                                     {'box_size': [20], 'layer_len': [3, 5, 10, 15]}]),
                              picks=ppicks)

plt.figure()
plt.plot(ppicks[0], label='true')
plt.plot(picks.predict(0), label='predicted')
plt.legend();

print(picks.approx_reg.best_params_, picks.fine_reg.best_params_)

all_predict = picks.predict()

np.max(np.abs(data.offset.values)) * 0.305e-3 # maximum offset converted from ft to km

plt.figure(figsize=(12,12))
plt.scatter(np.arange(55).repeat(96), data.GroupX.values.reshape(-1), s=0.1, label='receivers')
plt.scatter(np.arange(55), data.SourceX[:,0].values.reshape(-1), s=5, label='shots')
plt.xlabel('shot index');
plt.ylabel('x');
plt.legend();

data['SourceX_km'] = data.SourceX * 0.305e-3
data['GroupX_km'] = data.GroupX * 0.305e-3
data['SourceSurfaceElevation_km'] = data.SourceSurfaceElevation * 0.305e-3
data['ReceiverGroupElevation_km'] = data.ReceiverGroupElevation * 0.305e-3

picks_file = open('picks.txt', 'w')

num_src = 55
num_rcv = 96 * np.ones(55, np.int)
# for shots after 45, only use the first 75 receivers as picks
num_rcv[45:] = 75
minx = np.min(data.GroupX_km.values)
srcrecdepth_km = 0.0001
picks_file.write('%d\n' % num_src)
for src in range(num_src):
    picks_file.write('s %f %f %d\n' % (data.SourceX_km[src, 0] - minx,
                                       -(data.SourceSurfaceElevation_km[src, 0] - srcrecdepth_km),
                                       num_rcv[src]))
    for rcv in range(num_rcv[src]):
        pick_time = (all_predict[src*96 + rcv] - 10) * 0.002
        picks_file.write('r %f %f 0 %f 0.02\n' % (data.GroupX_km[src, rcv] - minx,
                                                  -(data.ReceiverGroupElevation_km[src, rcv] - srcrecdepth_km),
                                                  pick_time))
picks_file.close()

rcv_group = data.groupby(data.GroupX_km).first()
shot_group = data.groupby(data.SourceX_km).first()

elevations = np.concatenate([rcv_group.ReceiverGroupElevation_km.values,
                             shot_group.SourceSurfaceElevation_km.values])
x_pos = np.concatenate([rcv_group.GroupX_km.values, shot_group.SourceX_km.values])

import scipy.interpolate

topo_func = scipy.interpolate.interp1d(x_pos, elevations, fill_value='extrapolate')

print(np.min(x_pos), np.max(x_pos), np.max(x_pos) - np.min(x_pos))

x_new = np.arange(0, 10.6, 0.02) + np.min(x_pos)
topo_interp = topo_func(x_new)

plt.plot(x_new, topo_interp)
plt.xlabel('x (km)');
plt.ylabel('elevation (km)');

vfile_x = x_new - np.min(x_pos)
vfile_z = np.arange(0, 0.3, 0.005)
vfile_t = -topo_interp

vfile_x.tofile('x.txt', sep='\n')
vfile_z.tofile('z.txt', sep='\n')
vfile_t.tofile('t.txt', sep='\n')

get_ipython().system('./gen_smesh -A4.0 -B0.5 -Xx.txt -Zz.txt -Tt.txt > v.in')

get_ipython().system('./tt_inverse -Mv.in -Gpicks.txt -I30 -SV1000 -CVvcorrt.in')

v1=np.fromfile('inv_out.smesh.30.1', sep=" ")
nx=530
nz=60
v1=v1[4+2*nx+nz:].reshape(nx,nz)

plt.figure(figsize=(12,6))
plt.imshow(v1.T, aspect='auto')
plt.colorbar();
plt.xlabel('x (cells)')
plt.ylabel('z (cells)')

