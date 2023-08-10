get_ipython().magic('matplotlib notebook')
import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import mat4py
import math

mpl.style.use('mitch-exp')

with open('save_old.pkl', 'rb') as savefile:
    demo_sgs = np.array(pickle.load(savefile))

demo_sgs

demo_shift = np.insert(demo_sgs[1:], -1, [0, 0, 0, 0], axis=0)
demo_shift.shape

demo_collapse = demo_sgs[(demo_shift != demo_sgs).any(axis=-1)]
demo_collapse.shape

np.random.rand(1, 3)

noise = np.hstack((np.zeros((31, 3)), np.random.randn(31, 1)/20))
demo_collapse += noise

def forward_kin_v(exc, sw, bm, sk, bk, bias=0):
    '''This func is the same as 'forward_kin' in this module but is easily vectorized.

    Note: ported to Python from MATLAB "fwd_kin.m", assumed options = [0, 0]

    Args:
        exc (dict): a dict of the excavator physical parameters
        sw (float): the swing angle
        bm (float): boom displacement in cm
        sk      ^^
        bk      ^^
        bias (float): positive z bias on output, to adjust weird base frame

    Returns:
        eef (list: float): the position of the end-effector (EEF) in (x, y, z - base frame) and the angle of the bucket (axis x4 w.r.t. x1(0?) ground axis)
    '''
    # Assign the base swing angle
    t1 = sw

    # Define lengths
    a1 = exc['a1']
    a2 = exc['a2']
    a3 = exc['a3']
    a4 = exc['a4']

    # Compute or Get joint angles
    # Boom angle
    r_c1 = bm + exc['r_cyl1']
    a_a1b = np.arccos((exc['r_o1b']**2 + exc['r_o1a']**2 - r_c1**2)/(2 * exc['r_o1b']*exc['r_o1a']))
    t2 = a_a1b - exc['a_b12'] - exc['a_a1x1']

    # Stick angle
    r_c2 = sk + exc['r_cyl2']
    a_c2d = np.arccos((exc['r_o2c']**2 + exc['r_o2d']**2 - r_c2**2)/(2 * exc['r_o2c'] * exc['r_o2d']))
    t3 = 3 * np.pi - exc['a_12c'] - a_c2d - exc['a_d23']

    # Bucket angle
    r_c3 = bk + exc['r_cyl3']
    a_efh = np.arccos((exc['r_ef']**2 + exc['r_fh']**2 - r_c3**2)/(2 * exc['r_ef'] * exc['r_fh']))
    a_hf3 = np.pi - exc['a_dfe'] - a_efh
    r_o3h = math.sqrt(exc['r_o3f']**2 + exc['r_fh']**2 - 2 * exc['r_o3f'] * exc['r_fh'] * np.cos(a_hf3))
    a_f3h = np.arccos((r_o3h**2 + exc['r_o3f']**2 - exc['r_fh']**2)/(2 * r_o3h * exc['r_o3f']))
    a_h3g = np.arccos((r_o3h**2 + exc['r_o3g']**2 - exc['r_gh']**2)/(2 * r_o3h * exc['r_o3g']))
    t4 = 3 * np.pi - a_f3h - a_h3g - exc['a_g34'] - exc['a_23d']

    c1 = np.cos(t1)
    c2 = np.cos(t2)
    c234 = np.cos(t2 + t3 + t4)
    c23 = np.cos(t2 + t3)
    s1 = np.sin(t1)
    s2 = np.sin(t2)
    s234 = np.sin(t2 + t3 + t4)
    s23 = np.sin(t2 + t3)

    P04 = np.array([[c1*(a4*c234+a3*c23+a2*c2+a1)],
                    [s1*(a4*c234+a3*c23+a2*c2+a1)],
                    [(a4*s234+a3*s23+a2*s2)],
                    [1]])

    # Bucket angle; angle between x4 and x0-y0 plane
    tb = t2 + t3 + t4 - 3 * np.pi

    # Position and orientation of the end effector
    eef = [axis.pop() for axis in P04[0:3].tolist()]
    assert eef
    eef.append(tb)

    return eef[0], eef[1], eef[2] + bias

forward_kin = np.vectorize(forward_kin_v)
exc = mat4py.loadmat('exc.mat')['exc']

demo_xyz = np.array(forward_kin(exc, demo_collapse[:, 3], demo_collapse[:, 0],
                      demo_collapse[:, 1], demo_collapse[:, 2], bias=17.1))
demo_xyz.shape

demo_xyz[2]

from mpl_toolkits.mplot3d import Axes3D
import itertools
from sklearn import mixture

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(demo_xyz[0], demo_xyz[1], demo_xyz[2], zdir='z')

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange', 'r'])

def plot_results(X, Y_, title):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(6):
#         v, w = linalg.eigh(covar)
#         v = 2. * np.sqrt(2.) * np.sqrt(v)
#         u = w[0] / linalg.norm(w[0])
#         # as the DP will not use every component it has access to
#         # unless it needs it, we shouldn't plot the redundant
#         # components.
        if not np.any(Y_ == i):
            continue
        ax.scatter(X[Y_ == i, 0], X[Y_ == i, 1], X[Y_ == i, 2], zdir='z', c=color_iter.next())

#     plt.xlabel('X')
#     plt.ylabel('Y')
    plt.title(title)

demo_xyz.T.shape

gmm = mixture.GaussianMixture(n_components=6, covariance_type='full').fit(demo_collapse)

plot_results(demo_xyz.T, gmm.predict(demo_collapse), 'Test Cases')

for i in range(6):
    print('The %i\'th subgoal distribution is located at %s with covariance %s.') % (i, gmm.means_[i], gmm.covariances_[i])

ind = range(6)
ind.pop(3)
gmm_model = {'means': gmm.means_[ind],
             'covs': gmm.covariances_[ind]}

with open('gmm_model.pkl', 'wb') as savefile:
    pickle.dump(gmm_model, savefile)

gmm_model.values()



