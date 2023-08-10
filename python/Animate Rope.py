# Click the Blue Plane to preview this notebook as a CrossCompute Tool
base_intercept = 1
base_slope = 0.3
sine_amplitude = 0.2
sine_period = 3.14
target_folder = '/tmp'

import numpy as np

x = np.arange(0, 10, 0.01)
sine_a = sine_amplitude
sine_b = 2 * np.pi / sine_period

def f(t):
    return sum([
        base_slope * x,
        sine_a * np.sin(sine_b * (x + t / 25)),
        base_intercept,
    ])

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

figure, axes = plt.subplots()
line, = axes.plot(x, f(0))

def initialize():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

def animate(t):
    line.set_ydata(f(t))
    return line,

animation = FuncAnimation(
    figure, animate, np.arange(1, 360), init_func=initialize,
    interval=20, blit=True)

from os.path import join
target_path = join(target_folder, 'animation.mp4')
animation.save(target_path)
print('animation_video_path = ' + target_path)

