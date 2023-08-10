import numpy as np
from menpo.transform import ThinPlateSplines
from menpo.shape import PointCloud

# landmarks used in Principal Warps paper
# http://user.engineering.uiowa.edu/~aip/papers/bookstein-89.pdf
src_landmarks = np.array([[3.6929, 10.3819],
                          [6.5827,  8.8386],
                          [6.7756, 12.0866],
                          [4.8189, 11.2047],
                          [5.6969, 10.0748]])

tgt_landmarks = np.array([[3.9724, 6.5354],
                          [6.6969, 4.1181],
                          [6.5394, 7.2362],
                          [5.4016, 6.4528],
                          [5.7756, 5.1142]])

src = PointCloud(src_landmarks)
tgt = PointCloud(tgt_landmarks)
tps = ThinPlateSplines(src, tgt)

get_ipython().magic('matplotlib inline')
tps.view();

np.allclose(tps.apply(src_landmarks), tgt_landmarks)

# deformed diamond
src_landmarks = np.array([[ 0, 1.0],
                          [-1, 0.0],
                          [ 0,-1.0],
                          [ 1, 0.0]])

tgt_landmarks = np.array([[ 0, 0.75],
                          [-1, 0.25],
                          [ 0,-1.25],
                          [ 1, 0.25]])

src = PointCloud(src_landmarks)
tgt = PointCloud(tgt_landmarks)
tps = ThinPlateSplines(src, tgt)

get_ipython().magic('matplotlib inline')
tps.view();

np.allclose(tps.apply(src_landmarks), tgt_landmarks)

