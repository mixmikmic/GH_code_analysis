import conx as cx

net = cx.Network("Auto-Encoding with Conv")
net.add(cx.Layer("input", (28,28,1)),
        cx.Conv2DLayer("Conv2D-1", 16, (5,5), colormap="gray", activation="relu"),
        cx.MaxPool2DLayer("maxpool1", (2,2)),
        cx.Conv2DLayer("Conv2D-2", 132, (5,5), activation="relu"),
        cx.MaxPool2DLayer("maxpool2", (2,2)),
        cx.FlattenLayer("flatten"))
net.add(cx.Layer("output", 28 * 28, vshape=(28,28), activation='sigmoid'))
net.connect()

net.compile(error="mse", optimizer="adam")

net.dataset.get("mnist")

net.dataset.info()

net.dataset.set_targets_from_inputs()

net.dataset.targets.reshape(28 * 28)

net.dataset.targets.shape

net.dashboard()

net.dataset.chop(69900)

net.dataset.split(0.1)

net.reset()
net.train(50)



