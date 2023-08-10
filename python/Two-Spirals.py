import conx as cx
import math

def spiral_xy(i, spiral_num):
    """
    Create the data for a spiral. 
    
    Arguments:
        i runs from 0 to 96
        spiral_num is 1 or -1
    """
    φ = i/16 * math.pi
    r = 6.5 * ((104 - i)/104)
    x = (r * math.cos(φ) * spiral_num)/13 + 0.5
    y = (r * math.sin(φ) * spiral_num)/13 + 0.5
    return (x, y)

def spiral(spiral_num):
    return [spiral_xy(i, spiral_num) for i in range(97)]

a = ["A", spiral(1)]
b = ["B", spiral(-1)]

cx.scatter([a,b])

net = cx.Network("Two-Spirals")
net.add(
    cx.Layer("input", 2),
    cx.Layer("hidden3", 5, activation="sigmoid"),
    cx.Layer("hidden2", 5, activation="sigmoid"),
    cx.Layer("hidden1", 5, activation="sigmoid"),
    cx.Layer("output", 2, activation="softmax")
)
net.connect("input", "hidden1")
net.connect("hidden2", "hidden3")
net.connect("hidden3", "output")
net.connect("hidden1", "output")
net.connect("hidden1", "hidden2")
net.connect("hidden1", "hidden3")
net.connect("hidden2", "output")
net.compile(error="categorical_crossentropy", optimizer="rmsprop")

net.summary()

net.dashboard()

net.dataset.load([(xy, [0, 1]) for xy in spiral(1)] +
                 [(xy, [1, 0]) for xy in spiral(-1)])

net.reset()

net.train(10000, report_rate=100, batch_size=32, accuracy=1.0, tolerance=0.1)

import conx as cx
import copy

RESOLUTION = 50

def make_picture(res):
    matrix = [[0.0 for i in range(res)]
              for j in range(res)]
    for x,y in spiral(1):
        x = min(int(round(x * res)), res - 1)
        y = min(int(round(y * res)), res - 1)
        matrix[1 - y][x] = 0.5
    for x,y in spiral(-1):
        x = min(int(round(x * res)), res - 1)
        y = min(int(round(y * res)), res - 1)
        matrix[1 - y][x] = 0.5
    return matrix

matrix = make_picture(RESOLUTION)

cx.array_to_image(matrix, shape=(RESOLUTION,RESOLUTION,1)).resize((400,400))

def make_data(res):
    data = []
    for x,y in spiral(1):
        x = min(int(round(x * res)), res - 1)
        y = min(int(round(y * res)), res - 1)
        inputs = copy.deepcopy(matrix)
        inputs[1 - y][x] = 1.0
        inputs = cx.reshape(inputs,(50,50,1))
        data.append([inputs, [0, 1]])
    for x,y in spiral(-1):
        x = min(int(round(x * res)), res - 1)
        y = min(int(round(y * res)), res - 1)
        inputs = copy.deepcopy(matrix)
        inputs[1 - y][x] = 1.0
        inputs = cx.reshape(inputs,(50,50,1))
        data.append([inputs, [1, 0]])
    return data

data = make_data(RESOLUTION)

net = cx.Network("Two-Spirals using Pictures")
net.add(
    cx.ImageLayer("input", (RESOLUTION, RESOLUTION), 1),
    cx.Conv2DLayer("conv2d", 2, 4),
    cx.FlattenLayer("flatten"),
    cx.Layer("output", 2, activation="softmax")
)
net.connect()
net.compile(error="categorical_crossentropy", optimizer="rmsprop")

net.dataset.load(data)

net.dashboard()

net.reset()

net.train(1000, accuracy=1.0, report_rate=10)

def test0(x, y, res=RESOLUTION):
    x = min(int(round(x * res)), res - 1)
    y = min(int(round(y * res)), res - 1)
    inputs = copy.deepcopy(matrix)
    inputs[1 - y][x] = 1.0
    inputs = cx.reshape(inputs,(50,50,1))
    return net.propagate(inputs)[0]

def test1(x, y, res=RESOLUTION):
    x = min(int(round(x * res)), res - 1)
    y = min(int(round(y * res)), res - 1)
    inputs = copy.deepcopy(matrix)
    inputs[1 - y][x] = 1.0
    inputs = cx.reshape(inputs,(50,50,1))
    return net.propagate(inputs)[1]

cx.view([cx.heatmap(test0, format="image"), cx.heatmap(test1, format="image")], 
        labels=["output[0]","output[1]"], scale=7.0)

