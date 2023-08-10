import conx as cx
import random

count = 500

positives = [(i/count, i/(count * 2) + random.random()/6) for i in range(count)]
negatives = [(i/count, 0.3 + i/(count * 2) + random.random()/6) for i in range(count)]

cx.scatter([
         ["Positive", positives], 
         ["Negative", negatives],
        ], 
    symbols={"Positive": "bo", "Negative": "ro"})

ds = cx.Dataset()

ds.load([(p, [ 1.0], "Positive") for p in positives] +
        [(n, [ 0.0], "Negative") for n in negatives])

ds.shuffle()

ds.split(.1)

ds.summary()

net = cx.Network("Linearly Separable", 2, 1, activation="sigmoid")
net.compile(error="mae", optimizer="adam") 

net.set_dataset(ds)

net.dashboard()

net.summary()

net.to_array()

net.test(tolerance=0.4)

net.plot_activation_map(title="Before Training")

# net.test(tolerance=0.4, interactive=False)

symbols = {
    "Positive (correct)": "w+",
    "Positive (wrong)": "k+",
    "Negative (correct)": "w_",
    "Negative (wrong)": "k_",
}

net.plot_activation_map(scatter=net.test(tolerance=0.4, interactive=False), 
                        symbols=symbols, title="Before Training")

#net.delete()
#net.reset()

if net.saved():
    net.load()
    net.plot_results()
else:
    net.train(epochs=10000, accuracy=1.0, report_rate=50, 
             tolerance=0.4, batch_size=len(net.dataset.train_inputs), 
             plot=True, record=100, save=True)

net.plot_activation_map(scatter=net.test(tolerance=0.4, interactive=False), 
                        symbols=symbols, title="After Training")

net.to_array()

from conx.activations import sigmoid

def output(x, y):
    wts = net.get_weights("output")
    return sigmoid(x * wts[0][1][0] + y * wts[0][0][0] + wts[1][0])

def ascii(f):
    return "%4.1f" % f

for y in cx.frange(0, 1.1, .1):
    for x in cx.frange(1.0, 0.1, -0.1):
        print(ascii(output(x, y)), end=" ")
    print()

net.playback(lambda net, epoch: net.plot_activation_map(title="Epoch %s" % epoch, 
                                                        scatter=net.test(tolerance=0.4, interactive=False), 
                                                        symbols=symbols, 
                                                        format="svg"))

net.set_weights_from_history(-1)

# net.movie(lambda net, epoch: net.plot_activation_map(title="Epoch %s" % epoch, 
#                                                      scatter=net.test(tolerance=0.4, interactive=False), 
#                                                      symbols=symbols, 
#                                                      format="image"))

import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

negatives = []
while len(negatives) < 500:
    x = random.random()
    y = random.random()
    d = distance(x, y, 0.5, 0.5)
    if d > 0.375 and d < 0.5:
        negatives.append([x, y])
positives = []
while len(positives) < 500:
    x = random.random()
    y = random.random()
    d = distance(x, y, 0.5, 0.5)
    if d < 0.25:
        positives.append([x, y])

cx.scatter([
         ["Positive", positives], 
         ["Negative", negatives],
        ], 
    symbols={"Positive": "bo", "Negative": "ro"})

net = cx.Network("Non-Linearly Separable", 2, 5, 1, activation="sigmoid")
net.compile(error="mae", optimizer="adam") 

net.picture()

ds = cx.Dataset()

ds.load([(p, [ 1.0], "Positive") for p in positives] +
        [(n, [ 0.0], "Negative") for n in negatives])

ds.shuffle()

ds.split(.1)

net.set_dataset(ds)

net.test(tolerance=0.4)

net.dashboard()

net.plot_activation_map(scatter=net.test(interactive=False), symbols=symbols, title="Before Training")

# net.delete()
# net.reset()

if net.saved():
    net.load()
    net.plot_results()
else:
    net.train(epochs=10000, accuracy=1.0, report_rate=50, 
              tolerance=0.4, batch_size=256, 
              plot=True, record=100, save=True)

net.plot_activation_map(scatter=net.test(interactive=False), symbols=symbols, title="After Training")

for y in cx.frange(0, 1.1, .1):
    for x in cx.frange(1.0, 0.1, -0.1):
        print(ascii(net.propagate([x, y])[0]), end=" ")
    print()

net.playback(lambda net, epoch: net.plot_activation_map(title="Epoch: %s" % epoch, 
                                                        scatter=net.test(interactive=False),
                                                        symbols=symbols,
                                                        format="svg"))

# net.movie(lambda net, epoch: net.plot_activation_map(title="Epoch %s" % epoch, 
#                                                      scatter=net.test(tolerance=0.4, interactive=False), 
#                                                      symbols=symbols, 
#                                                      format="image"))

