import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
initial_snippet = np.fromfile('/home/mwootten/school/research/ATS1.raw32', dtype=np.dtype('i4'), count=20000)

pd.DataFrame(initial_snippet).describe()

def lowpass(x, dt, RC):
    y = []
    α = dt / (RC + dt)
    y.append(α * x[0])
    for i in range(1, len(x)):
        y.append(α * x[i] + (1-α) * y[i-1])
    return y

def plot_array(y):
    x = np.arange(0, len(y), 1)
    plt.plot(x, y)
    plt.show()

def plot_smoothing(dt, RC):
    plot_array(lowpass(initial_snippet, dt, RC))

plot_array(initial_snippet)

plot_smoothing(1, 50)

plot_smoothing(1, 150)

plot_smoothing(1, 500)

plot_smoothing(1, 20000)



