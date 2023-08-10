import conx as cx
import random

cx.Dataset.datasets()

fingers = cx.Dataset.get("fingers")

fingers.info()

net = cx.Network("Finger Math")
net.add(
    cx.ImageLayer("input1", (30, 40), 3),
    cx.ImageLayer("input2", (30, 40), 3),
    cx.FlattenLayer("flatten1"),
    cx.FlattenLayer("flatten2"),
    cx.Layer("hidden", 100, activation="relu"),
    cx.Layer("output", 11, activation="softmax"),
)
net.connect("input1", "flatten1")
net.connect("input2", "flatten2")
net.connect("flatten1", "hidden")
net.connect("flatten2", "hidden")
net.connect("hidden", "output")
net.compile(error="categorical_crossentropy", optimizer="sgd")

net.config["hspace"] = 300
net.picture()

def get_dataset():
    for i in range(len(fingers)):
        r1 = random.randint(0, len(fingers) - 1)
        v1 = int(fingers.labels[r1])
        r2 = random.randint(0, len(fingers) - 1)
        v2 = int(fingers.labels[r2])
        target = v1 + v2
        yield [[fingers.inputs[r1], fingers.inputs[r2]], cx.onehot(target, 11)]

net.dataset.load(get_dataset(), 1000)

net.dashboard()

net.train(20, accuracy=1.0, batch_size=256, tolerance=0.5)



