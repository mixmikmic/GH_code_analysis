from bqplot import pyplot as plt
import numpy as np
# import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets.widgets import IntSlider, HBox, Button
get_ipython().run_line_magic('matplotlib', 'inline')

def circ(x, radius=2):
    y = np.sqrt(radius**2 - x**2)
    return y

def calculate_ratios(samples):

    samples_y_curve = circ(samples[:, 0])
    inside_dots = samples_y_curve > samples[:, 1]

    ratios = []
    for n_plt in range(N_SAMPLES):
        if n_plt == 0:
            ratios.append(0)
            continue
        ratios.append(4 * (sum(inside_dots[:n_plt]) / len(inside_dots[:n_plt])))

    return samples_y_curve, inside_dots, ratios

def _update_plot(change=None):
    n_plt = slider.value
    plt_samples = plot_data['data'][:n_plt]
    inside_dots = plot_data['inside']
    ratios = plot_data['ratios']

    colors = np.where(inside_dots[:n_plt], '#41f465', "#f44b42").tolist()
    scat.x = plt_samples[:, 0]
    scat.y = plt_samples[:, 1]
    scat.colors = colors
    fig.title = 'Ratio of in-to-out dots: {}'.format(ratios[n_plt])
    
    fig_line.x = range(n_plt)
    line.y = ratios[:n_plt]
    line.x = range(n_plt)
    
def _reset_data(change=None):
    samples = np.random.random(N_SAMPLES * 2) * 2
    samples = samples.reshape([-1, 2])
    
    samples_y_curve, inside_dots, ratios = calculate_ratios(samples)
    
    plot_data['data'] = samples
    plot_data['inside'] = inside_dots
    plot_data['ratios'] = ratios
    _update_plot()

N_SAMPLES = 700
samples = np.random.random(N_SAMPLES * 2) * 2
samples = samples.reshape([-1, 2])
samples_y_curve, inside_dots, ratios = calculate_ratios(samples)

width = '350px'

plt.clear()

plot_data = {'data': samples, 'ratios': ratios, 'inside': inside_dots}

fig = plt.figure()
x = np.linspace(0, 5, 1000)
y = circ(x)
myplt = plt.plot(x=x, y=y, s=[100] * len(y))
scat = plt.scatter(samples[:10, 0], samples[:10, 1], options={'s': 1})
plt.xlim(0, 2)
fig.layout.height = width
fig.layout.width = width

fig_line = plt.figure()
fig_line.layout.height = width
fig_line.layout.width = width
fig_line.title = "Estimation of Pi"
line = plt.plot(range(10), ratios[:10])
line2 = plt.hline(np.pi, ls='--', c='k')
plt.xlim(0, N_SAMPLES)
plt.ylim(2, 4)

slider = IntSlider(value=10, min=1, max=N_SAMPLES-1, description="$N_{samples}$")
slider.observe(_update_plot)

button = Button(description="Reset data")
button.on_click(_reset_data)

_update_plot()

box = HBox([fig, fig_line])
box2 = HBox([slider, button])
display(box, box2)





