import numpy as np
from menpo.transform import PiecewiseAffine

from menpo.shape import TriMesh, PointCloud
a = np.array([[0, 0], [1, 0], [0, 1], [1, 1],
              [-0.5, -0.7], [0.8, -0.4], [0.9, -2.1]])
b = np.array([[0,0], [2, 0], [-1, 3], [2, 6],
              [-1.0, -0.01], [1.0, -0.4], [0.8, -1.6]])
tl = np.array([[0,2,1], [1,3,2]])

src = TriMesh(a, tl)
src_points = PointCloud(a)
tgt = PointCloud(b)

pwa = PiecewiseAffine(src_points, tgt)

get_ipython().magic('matplotlib inline')
# points_s = PointCloud(np.random.rand(10000).reshape([-1,2]))
points_f = PointCloud(np.random.rand(10000).reshape([-1,2]))
points_f.view()

t_points_f = pwa.apply(points_f);
t_points_f.view()

test = np.array([[0.1,0.1], [0.7, 0.9], 
                 [0.2,0.3], [0.5, 0.6]])

pwa.index_alpha_beta(test)

