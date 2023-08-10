get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from menpo.shape import PointCloud
from menpo.image import Image

# A unit square centered at (0.5, 0.5)
points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
plt.plot(points[:, 0], points[:, 1], 'bo')

# A PointCloud containing a copy of the unit square
pcloud = PointCloud(points)
pcloud.view(new_figure=True)

# A small black image
image = Image.init_blank([100, 100], fill=1)
image.view(new_figure=True);

get_ipython().magic('matplotlib inline')
from menpo.transform import Translation

small_2d_translation = Translation([-5, 5])

translated_pcloud = small_2d_translation.apply(pcloud)
# Notice it is now centered around (-4.5, 5.5)
translated_pcloud.view();

get_ipython().magic('matplotlib inline')
from menpo.image import BooleanImage

small_2d_translation = Translation([-50, 50])

translated_image = image.warp_to_mask(BooleanImage.init_blank([100, 100]), small_2d_translation)
# Notice it is now centered around (-4.5, 5.5)
translated_image.view();

