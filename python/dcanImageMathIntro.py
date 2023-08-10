get_ipython().run_line_magic('matplotlib', 'inline')
# load some python visualization and mathematics packages
from matplotlib import pyplot as plt
import numpy as np
from pprint import pprint

# intialize a grid, we will call it "grid1"
m = 6; n = 6
grid1 = np.zeros((m, n), dtype=int)
grid1[:, :n//2] = 1

# initialize a second grid, "grid2"
m = 3; n = 3
even = np.array(m * [1, 0])
odd = even[::-1]
grid2 = np.row_stack(n * (even, odd))

# print the grids out as data/matrices
pprint(grid1.tolist())
print('')
pprint(grid2.tolist())

plt.subplot(1, 2, 1)
plt.imshow(grid1, cmap='gray')
plt.title('grid1')
plt.subplot(1, 2, 2)
plt.imshow(grid2, cmap='gray')
plt.title('grid2')

# define image I and number a.
I = grid1; a = 2
# add 2 to the image.
J = a + I
pprint(I.tolist())
print('')
pprint(J.tolist())

plt.imshow(J, cmap='gray')

plt.imshow(J, cmap='gray', vmin=0, vmax=4)

I = grid1; J = grid2
K = I + J
pprint(K.tolist())
plt.imshow(K, cmap='gray')

from IPython.display import Image
Image("img/all_4_affine.jpg")

from ipywidgets import interact
import ipywidgets as widgets
from skimage.transform import warp


N = 64
square = np.zeros((N, N))
square[N//2-5:N//2+6, N//2-5:N//2+6] = 1


def f(tx, ty, θ, interp=None):
    if interp:
        interp = 0
    else:
        interp = 1
    θ =  θ / 180 * np.pi
    global square
    mat = np.array([
        [np.cos(θ), -np.sin(θ), tx],
        [np.sin(θ), np.cos(θ), ty],
        [0, 0, 1] # rigid body
    ])
    img = warp(square, mat, output_shape=square.shape, order=interp)
    plt.imshow(img, cmap='gray')
    plt.grid()

x = widgets.IntSlider(min=-32, max=32, step=2)
y = widgets.IntSlider(min=-32, max=32, step=2)
t = widgets.IntSlider(min=-50, max=50, step=10)
interpolation = widgets.Checkbox(
    value=False,
    description='Nearest Neighbor',
    disabled=False
)

plot = interact(f, tx=x, ty=y, θ=t, interp=interpolation)

N = 128
square2 = np.zeros((N, N))
square2[0:11, 0:11] = 1

shift = np.array([
        [1, 0, -60],
        [0, 1, -60],
        [0, 0, 1] # rigid body
    ])

def f(tx, ty, θ, kx, ky, sx, sy, interp=None):
    if interp:
        interp = 0
    else:
        interp = 1
    θ =  θ / 180 * np.pi
    global square2
    mat = np.array([
        [sx * np.cos(θ), ky * -np.sin(θ), tx],
        [kx * np.sin(θ), sy * np.cos(θ), ty],
        [0, 0, 1] # rigid body
    ])
    mat = mat @ shift
    
    img = warp(square2, mat, output_shape=square2.shape, order=interp)
    plt.imshow(img, cmap='gray')
    plt.grid()

x2 = widgets.IntSlider(min=-64, max=64, step=4)
y2 = widgets.IntSlider(min=-64, max=64, step=4)
t2 = widgets.IntSlider(min=-180, max=180, step=10)
kx = widgets.FloatSlider(min=-.5, max=2.0, value=1)
ky = widgets.FloatSlider(min=-.5, max=2.0, value=1)
sx = widgets.FloatSlider(min=0, max=2.0, value=1)
sy = widgets.FloatSlider(min=0, max=2.0, value=1)
interpolation = widgets.Checkbox(
    value=False,
    description='Nearest Neighbor',
    disabled=False
)

plot = interact(f, tx=x2, ty=y2, θ=t2, kx=kx, ky=ky, sx=sx, sy=sy, interp=interpolation)

