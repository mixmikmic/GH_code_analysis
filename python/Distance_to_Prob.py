get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

data = np.random.uniform(0, 234, size=20).reshape(5,4)
prob = 1. / data**2.
prob = prob / prob.sum()

fig, (ax0, ax1) = plt.subplots(1,2)

im = ax0.imshow(data)
divider = make_axes_locatable(ax0)
cax0 = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax0, label='Distance (Lower is better)')

im = ax1.imshow(prob)
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax1, label='Probability (Higher is Better)')

fig.tight_layout()
fig.show()
# print(prob)

x, y = np.meshgrid(range(prob.shape[0]),range(prob.shape[1]) )
xy = np.array([x,y])

from collections import defaultdict

a = defaultdict(int)

for _ in range(1000):
    rand_pos = np.random.choice(range(np.prod(prob.shape)), p=prob.flatten())
    x = rand_pos // prob.shape[0]
    y = rand_pos % prob.shape[1]
    a[(x,y)] += 1

loc = np.unravel_index(np.argmin(data, axis=None), prob.shape)

loc

a

from collections import defaultdict

func = lambda: {'correct':0, 'wrong':0, 'accuracy':0}
book = defaultdict(func)

book['1_1']['correct'] +=1
book['1_2']['correct'] +=1
book['1_3']['correct'] +=1
book

{f'{k}_{k2}': v2 for k, v in book.items() for k2, v2 in v.items()}



