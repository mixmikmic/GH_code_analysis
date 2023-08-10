from wave_1d_fd_pml import propagators, test_wave_1d_fd_pml, find_profile
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
np.random.seed(0)

v = propagators.Pml1(np.ones(100), 5)
plt.plot(v.sigma)
plt.xlabel('x')
plt.ylabel('sigma')

profile_len = 10
maxiter = 1024
vs = np.linspace(1500, 5000, 3)

profile = lambda x: x[0] + x[1] * np.arange(profile_len)

models = find_profile._get_models(vs, 5)
prop = find_profile._get_prop(2)
cost = np.zeros([50, 50])
for i0, x0 in enumerate(np.linspace(0, 500, 50)):
    for i1, x1 in enumerate(np.linspace(0, 500, 50)):
        cost[i0, i1] = find_profile.evaluate([x0, x1], profile, models, prop)

ax = plt.subplot(111)
ax.imshow(np.log10(cost), vmax=np.median(np.log10(cost)))
plt.xlabel('x[1]')
plt.ylabel('x[0]')
ticks = ax.get_xticks()*(int(np.linspace(0, 500, 50)[1]))
ax.set_xticklabels(ticks);
ticks = ax.get_yticks()*(int(np.linspace(0, 500, 50)[1]))
ax.set_yticklabels(ticks);
plt.title('Cost function of linear profile')

bounds = ((0, 5000), (0, 500))
x_linear1, _, _ = find_profile.find_profile(profile, bounds, optimizer='brute', pml_version=1, vs=vs, maxiter=maxiter)
x_linear1, profile_linear1, cost_linear1 = find_profile.find_profile(profile, bounds, optimizer='hygsa', init=x_linear1, pml_version=1, vs=vs, maxiter=maxiter)

x_linear2, _, _ = find_profile.find_profile(profile, bounds, optimizer='brute', pml_version=2, vs=vs, maxiter=maxiter)
x_linear2, profile_linear2, cost_linear2 = find_profile.find_profile(profile, bounds, optimizer='hygsa', init=x_linear2, pml_version=2, vs=vs, maxiter=maxiter)

profile = lambda x: x[0] + x[1] * np.arange(profile_len) ** x[2]
bounds = ((0, 5000), (0, 500), (0.1, 10))
x_power1, _, cost_power1 = find_profile.find_profile(profile, bounds, optimizer='brute', pml_version=1, vs=vs, maxiter=maxiter)
if cost_linear1 < cost_power1:
    init = np.append(x_linear1, 1)
else:
    init = x_power1
x_power1, profile_power1, cost_power1 = find_profile.find_profile(profile, bounds, optimizer='hygsa', init=init, pml_version=1, vs=vs, maxiter=maxiter)

x_power2, profile_power2, cost_power2 = find_profile.find_profile(profile, bounds, optimizer='brute', pml_version=2, vs=vs, maxiter=maxiter)
if cost_linear2 < cost_power2:
    init = np.append(x_linear2, 1)
else:
    init = x_power2
x_power2, profile_power2, cost_power2 = find_profile.find_profile(profile, bounds, optimizer='hygsa', init=init, pml_version=2, vs=vs, maxiter=maxiter)

profile = lambda x: x[0] + x[1] * np.cos(np.linspace(x[2], x[2] + x[3], profile_len))
bounds = ((0, 5000), (-500, 0), (0, np.pi/2), (0, np.pi/2))
x_cosine1, _, _ = find_profile.find_profile(profile, bounds, optimizer='brute', pml_version=1, vs=vs, maxiter=maxiter)
x_cosine1, profile_cosine1, cost_cosine1 = find_profile.find_profile(profile, bounds, optimizer='hygsa', init=x_cosine1, pml_version=1, vs=vs, maxiter=maxiter)

x_cosine2, _, _ = find_profile.find_profile(profile, bounds, optimizer='brute', pml_version=2, vs=vs, maxiter=maxiter)
x_cosine2, profile_cosine2, cost_cosine2 = find_profile.find_profile(profile, bounds, optimizer='hygsa', init=x_cosine2, pml_version=2, vs=vs, maxiter=maxiter)

profile = lambda x: x
bounds = [(0, 5000)] * profile_len
x_freeform1, _, cost_freeform1 = find_profile.find_profile(profile, bounds, optimizer='brute', pml_version=1, vs=vs, maxiter=maxiter)
if cost_linear1 < cost_freeform1:
    init = profile_linear1
else:
    init = x_freeform1
if cost_power1 < cost_linear1:
    init = profile_power1
if cost_cosine1 < cost_power1:
    init = profile_cosine1
x_freeform1, profile_freeform1, cost_freeform1 = find_profile.find_profile(profile, bounds, optimizer='hygsa', init=init, pml_version=1, vs=vs, maxiter=maxiter)

x_freeform2, _, cost_freeform2 = find_profile.find_profile(profile, bounds, optimizer='brute', pml_version=2, vs=vs, maxiter=maxiter)
if cost_linear2 < cost_freeform2:
    init = profile_linear2
else:
    init = x_freeform2
if cost_power2 < cost_linear2:
    init = profile_power2
if cost_cosine2 < cost_power2:
    init = profile_cosine2
x_freeform2, profile_freeform2, cost_freeform2 = find_profile.find_profile(profile, bounds, optimizer='hygsa', init=init, pml_version=2, vs=vs, maxiter=maxiter)

profiles=[[profile_linear1, profile_power1, profile_cosine1, profile_freeform1],
         [profile_linear2, profile_power2, profile_cosine2, profile_freeform2]]
for pml_version, profiles in enumerate(profiles):
    plt.figure()
    for profile in profiles:
        plt.plot(profile)

nv = 10
e = np.zeros([nv**2, 8])
vs2 = np.linspace(1500, 5000, nv)
for v0i, v0 in enumerate(vs2):
    for v1i, v1 in enumerate(vs2):
        model = test_wave_1d_fd_pml.model_one(500, v0=v0, v1=v1, freq=25)
        profiles=[[profile_linear1, profile_power1, profile_cosine1, profile_freeform1],
                 [profile_linear2, profile_power2, profile_cosine2, profile_freeform2]]
        for pml_version, profiles in enumerate(profiles):
            if pml_version == 0:
                prop = propagators.Pml1
            else:
                prop = propagators.Pml2
            for pi, profile in enumerate(profiles):
                v = prop(model['model'], model['dx'], model['dt'], len(profile), profile=profile)
                y = v.steps(model['nsteps'], model['sources'], model['sx'])
                e[v0i*nv+v1i, pml_version*4+pi] = (np.sum(np.abs(v.current_wavefield)))

plt.boxplot(e);

profile = profile_freeform2
prop = propagators.Pml2
model = test_wave_1d_fd_pml.model_one(500, v0=1500, v1=5000, freq=25)
v = prop(model['model'], model['dx'], model['dt'], len(profile), profile=profile)
y = v.steps(model['nsteps'], model['sources'], model['sx'])
y[:, v.total_pad] = y[:, -v.total_pad] = np.nan
plt.imshow(y[:,8:-8], aspect='auto')
plt.xlabel('x')
plt.ylabel('time step')

dxs = np.arange(1,5)
x_dx = np.zeros([5, 2])
x_dx[-1, :] = x_linear2
bounds = ((0, 5000), (0, 500))
for i, dx in enumerate(dxs):
    profile_len_dx = int(profile_len * 5 / dx)
    profile = lambda x: x[0] + x[1] * np.arange(profile_len_dx)
    x_dx[i, :], _, _ = find_profile.find_profile(profile, bounds, optimizer='brute', pml_version=2, vs=vs, maxiter=maxiter, dx=dx)
    x_dx[i, :], _, _ = find_profile.find_profile(profile, bounds, optimizer='hygsa', init=x_dx[i, :], pml_version=2, vs=vs, maxiter=maxiter, dx=dx) 

ax=plt.subplot(111)
ax.plot(np.arange(1,6), x_dx[:,0], label='x[0]')
ax.plot(np.arange(1,6), x_dx[:,1], label='x[1]')
ax.legend()
plt.xlabel('cell size (m)')
plt.ylabel('parameter value')
ax.set_ylim(ymin=0);

