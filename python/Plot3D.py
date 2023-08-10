import conx as cx

net = cx.Network("XOR", 2, 5, 1, activation="tanh")

net.picture()

net.compile(error="mse", optimizer="sgd")

net.propagate([-.5, .5])

net.summary()

net.reset()

net.dataset.append([-1, -1], [-1])
net.dataset.append([-1, +1], [+1])
net.dataset.append([+1, -1], [+1])
net.dataset.append([+1, +1], [-1])

dash = net.dashboard()
dash

net.train(10000, accuracy=1.0, tolerance=.1, plot=True, report_rate=200)

cx.plot3D(lambda x,y: x ** 2 + y ** 2, (-1,1,.1), (-1,1,.1), label="Label",
          zlabel="activation", linewidth=0, colormap="RdGy", mode="surface")

cx.plot3D(lambda x,y: x ** 2 + y ** 2, (-1,1,.1), (-1,1,.1), label="Label",
          zlabel="activation", linewidth=1, colormap="RdGy", mode="wireframe")

import random
points1 = []
for i in range(100):
    points1.append([random.random(), random.random(), random.random()])
points2 = []
for i in range(100):
    points2.append([random.random(), random.random(), random.random()])
    
cx.plot3D([["Test1", points1], ["Test2", points2]], zlabel="activation", mode="scatter")

cx.plot3D(lambda x,y: net.propagate([x,y])[0], (-1, 1, .1), (-1, 1, .1), 
          zlabel="activation", 
          mode="surface")

cx.plot3D(lambda x,y: net.propagate([x,y])[0], (-1, 1, .1), (-1, 1, .1), 
          zlabel="activation", 
          mode="wireframe", linewidth=1)

