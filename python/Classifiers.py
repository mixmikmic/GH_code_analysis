import conx as cx

blue   = [( 90, 404), (143, 396), (231, 377), 
          (310, 342), (367, 276), (402, 188), 
          (391,  70)]
red    = [( 94, 196), (126,  78), (233, 269), 
          (232, 126), (307,  75), (633, 271)]
green  = [(230, 522), (330, 419), (476, 464), 
          (607, 165), (684, 509), (627,  45), 
          (720, 175)]
purple = [(542, 296), (504, 167), (500, 60), 
          (620, 400), (720, 437)]

size = (800, 600)

def scale(data, size):
    return [(x/size[0],1 - y/size[1]) for (x,y) in data]

scale(blue, size)

## "r" is red, "g" is green, "b" is blue, 
## "m" is magenta/purple, and "o" means plot a dot
symbols = {"Red": "ro", "Green": "go", "Blue": "bo", "Purple": "mo"}

cx.scatter([["Red", scale(red, size)], 
            ["Green", scale(green, size)],
            ["Purple", scale(purple, size)],
            ["Blue", scale(blue, size)]], 
            symbols=symbols, 
            xmin=0, xmax=1,
            ymin=0, ymax=1)

net = cx.Network("RGB Classifier")
net.add(cx.Layer("input", 2),
        cx.Layer("hidden", 10, activation="relu"),
        cx.Layer("output", 4, activation="softmax"))
net.connect()
net.compile(error="categorical_crossentropy", 
            optimizer="sgd", lr=0.1, momentum=0.3)

net.picture()

ds = ([(inputs, cx.onehot(0, 4), "Red") for inputs in scale(red, size)] +
      [(inputs, cx.onehot(1, 4), "Green") for inputs in scale(green, size)] +
      [(inputs, cx.onehot(3, 4), "Purple") for inputs in scale(purple, size)] +
      [(inputs, cx.onehot(2, 4), "Blue") for inputs in scale(blue, size)])

ds

net.dataset.load(ds)

net.dataset.inputs[0], net.dataset.targets[0], net.dataset.labels[0]

net.dataset.split("all")

net.reset()
net.train(15000, accuracy=1.0, batch_size=16, 
          report_rate=200, use_validation_to_stop=True,
          save=True, record=50)

net.test(show=True, tolerance=0.5)

net.dashboard()

cx.view([net.plot_activation_map(to_unit=i, format="image") for i in range(4)], scale=3)

def function(x, y):
    output = net.propagate([x, y])
    if cx.argmax(output) == 0: ## red
        return 0.5
    elif cx.argmax(output) == 1: ## green
        return 1.0
    elif cx.argmax(output) == 2: ## blue
        return 0.0
    else:
        return 0.2
heatmap_image = cx.heatmap(function, colormap="brg", format="image")

scatter_image = cx.scatter([["Red", scale(red, size)], 
            ["Green", scale(green, size)],
            ["Purple", scale(purple, size)],
            ["Blue", scale(blue, size)]], 
            symbols=symbols, 
            xmin=0, xmax=1,
            ymin=0, ymax=1, format="image")

scatter_image.size

display(heatmap_image)
display(scatter_image)

net.playback(lambda net, epoch: cx.heatmap(function, colormap="brg", title="Epoch %s" % epoch, format="image"))

net.movie(lambda net, epoch: cx.heatmap(function, colormap="brg", title="Epoch %s" % epoch, format="image"))



