import numpy
import toyplot

def grouped_bars(axes, data, group_names, group_width=None):
    if group_width is None:
        group_width = 0.5
    group_left_edges = numpy.arange(data.shape[0], dtype="float") - (group_width / 2.0)
    bar_width = group_width / data.shape[1]
    
    marks = []
    axes.x.ticks.locator = toyplot.locator.Explicit(labels=group_names)
    for index, series in enumerate(data.T):
        left_edges = group_left_edges + (index * bar_width)
        right_edges = group_left_edges + ((index + 1) * bar_width)
        marks.append(axes.bars(left_edges, right_edges, series, opacity=0.5))
        
    return marks

numpy.random.seed(1234)

group_names = [250, 500, 1000, 2000, 3000]

canvas = toyplot.Canvas(width=1000, height=1000)
for index, bar_count in enumerate(numpy.arange(1, 10)):
    axes = canvas.cartesian(grid=(3, 3, index))
    axes.x.label.text = "Number of Runs"
    axes.y.label.text = "Wallclock Time (min)"

    series = numpy.random.uniform(low=0.2, high=1.2, size=(len(group_names), bar_count))
    
    grouped_bars(axes, series, group_names)

