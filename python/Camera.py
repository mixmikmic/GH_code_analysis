from conx.widgets import CameraWidget
import conx as cx

camera = CameraWidget()

camera

image = camera.get_image()
image

image.save("camera.jpg")

data = camera.get_data()

data.shape

net = cx.Network("Camera Network")
net.add(cx.ImageLayer("camera", (240, 320), 3),
        cx.Conv2DLayer("conv2d", 32, (3,3)),
        cx.MaxPool2DLayer("maxpool", (2,2)),
        cx.FlattenLayer("flatten"),
        cx.Layer("output", 10))
net.connect()
net.compile(error="mse", optimizer="adam")

net.dataset.append(data, cx.to_categorical(1, 10))

net.dashboard()

net.picture(camera.get_data())

