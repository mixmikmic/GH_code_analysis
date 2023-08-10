# Click the Blue Plane to preview this notebook as a CrossCompute Tool
source_image_path = 'smiley-20170620-2300.png'
color_select = """
    yellow
    magenta
    cyan
    red
    green
    blue"""
target_folder = '/tmp'

MULTIPLIER_BY_COLOR = {
    'yellow': (1, 1, 0),
    'magenta': (1, 0, 1),
    'cyan': (0, 1, 1),
    'red': (1, 0, 0),
    'green': (0, 1, 0),
    'blue': (0, 0, 1),
}
color = color_select.strip().splitlines()[0].strip('*')
color

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
image = plt.imread(source_image_path)
plt.imshow(image);

tinted_image = MULTIPLIER_BY_COLOR[color] * image[:, :, :3]
plt.imshow(tinted_image);

from os.path import join
target_path = join(target_folder, 'image.png')
plt.imsave(target_path, tinted_image)
print('target_image_path = ' + target_path)

