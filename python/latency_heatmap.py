from __future__ import print_function

import csv
import sys
from collections import Counter
from collections import defaultdict
from collections import namedtuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from IPython.display import display
from ipywidgets import widgets


# Supported converters for different data formats
# ab = Apache Benchmark -g output
# tl = csv of time,latency pairs

def read_ab_data(input_file):
    with open(input_file, 'r') as f:
        next(f)
        reader = csv.reader(f, delimiter='\t')
        for starttime, seconds, ctime, dtime, ttime, wait in reader:
            yield int(seconds), int(ttime)


def read_tl_data(input_file):
    with open(input_file, 'r') as f:
        next(f)
        reader = csv.reader(f)
        for time, latency in reader:
            yield int(time), int(latency)


converters = {
    'ab': read_ab_data,
    'tl': read_tl_data
}

# Common functionality for bucketing latency and drawing plots


def read_data(input_file, data_format='ab'):
    for time, latency in converters[data_format](input_file):
        yield time, latency


PlotData = namedtuple(
    'PlotData',
    ['data',
     'min_latency', 'max_latency',
     'min_time', 'max_time',
     'num_values']
)


def calculate_data(data_generator, min_num_values=40):
    buckets = defaultdict(list)
    min_time, max_time = sys.maxint, 0
    min_latency, max_latency = sys.maxint, 0
    for time, latency in data_generator:
        buckets[time].append(latency)
        min_time, max_time = min(min_time, time), max(max_time, time)
        min_latency, max_latency = min(
            min_latency, latency), max(max_latency, latency)

    num_values = min(min_num_values, max_latency + 1)

    # If we want N buckets on the vertical, we have to divide the
    # latency data into those buckets
    v_bucket_interval = (max_latency / float(num_values))

    data = np.zeros(
        shape=(max_time - min_time + 1, num_values),
        dtype=np.float32
    )

    for bucket in buckets:
        x_index = bucket - min_time
        count = Counter(buckets[bucket])
        total = float(sum(count.values()))
        for key in count:
            y_index = min(int((key) / v_bucket_interval), num_values - 1)
            value = count[key] / total
            data[x_index][y_index] += value
    return PlotData(
        data=data, min_time=min_time, max_time=max_time,
        min_latency=min_latency, max_latency=max_latency,
        num_values=num_values
    )


def draw_figure(dataset, plt_data):
    fig = plt.figure(figsize=(12, 6))

    v_bucket_interval = (plt_data.max_latency / float(plt_data.num_values))

    plt.pcolormesh(
        plt_data.data.T, cmap='RdYlBu_r',
        vmin=0, vmax=1.0,
        edgecolor='k', linewidth=0.01
    )
    plt.title('Latency Heatmap of {0}'.format(dataset))
    plt.ylabel('Response Time (ms)')
    plt.xlabel('Time (s)')
    plt.colorbar()
    plt.grid(False)
    plt.xlim(0, plt_data.max_time - plt_data.min_time + 1)
    plt.ylim(0, plt_data.num_values)
    # This way we see the actual latency values of the buckets
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=2.0)
    plt.gca().yaxis.set_major_locator(loc)
    plt.gca().yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, pos: ('{0:2.0f}'.format(x * v_bucket_interval))
        )
    )

    fig.canvas.draw()
    plt.show()

# UI setup
input_file = widgets.Text(
    description='Data File:', value='demo.dat'
)

input_num_values = widgets.IntSlider(
    value=40, min=1, max=100, step=1,
    description='Minimum number of vertical buckets',
    tooltip=(
        'Make this smaller to group more data together, '
        'and smaller to separate it out'
    )
)

input_type = widgets.Dropdown(
    options=[
        ('Apache Benchmark', 'ab'),
        ('CSV time vs latency', 'tl'),
    ],
    description='Type of datafile (e.g. ab):',
)

input_submit = widgets.Button(
    description='Draw Latency Plot!', button_style='success',
    tooltip='Click me to draw the data file as a latency histogram',
    icon='paint-brush'
)


def handle_submit(sender):
    print('Plotting: (data_file, type): ',
          (input_file.value, input_type.value))
    print('min_buckets: ', input_num_values.value)
    data_generator = read_data(input_file.value, input_type.value)
    data = calculate_data(
        data_generator, min_num_values=input_num_values.value)
    draw_figure(input_file.value, data)


input_submit.on_click(handle_submit)

input_layout = widgets.HBox([input_file, input_type, input_num_values])
submit_layout = widgets.HBox([input_submit])
layout = widgets.VBox([input_layout, submit_layout])

display(layout)



