import enthought.mayavi.mlab as mylab
import numpy as np
x, y, z, value = np.random.random((4, 40))
mylab.points3d(x, y, z, value)
mylab.show()



