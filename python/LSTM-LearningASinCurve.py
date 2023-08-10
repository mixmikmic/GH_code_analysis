from conx import Network, Layer, LSTMLayer, plot, frange
import math

scaled_data = [round(math.sin(x),1) for x in frange(0, 2 * math.pi + .1, .1)]

[x for x in enumerate(scaled_data) if -0.1 < x[1] < 0.1]

plot(["Scaled Data", scaled_data], default_symbol="o")

scaled_data[0], scaled_data[-1]

sequence = [[datum] for datum in scaled_data]

time_steps = 10  # history
batch_size = 1  # how many to load at once
features = 1    # features (length of input vector)

def create_dataset(sequence, time_steps):
    dataset = []
    for i in range(len(sequence)-time_steps-1):
        dataset.append([sequence[i:(i+time_steps)], 
                       sequence[i + time_steps]])
    return dataset

dataset = create_dataset(sequence, time_steps)

print(dataset[0])
print(dataset[1])

net = Network("LSTM - sin")
net.add(Layer("input", features, batch_shape=(batch_size, time_steps, features)))
net.add(LSTMLayer("lstm", 4)) 
net.add(Layer("output", 1))
net.connect()
net.compile(error="mse", optimizer="adam")

net.dataset.clear()
net.dataset.load(dataset)

net.dashboard()

net["lstm"].get_output_shape()

#net.dataset.split(.33)

net.propagate([[.02]] * time_steps)

net.reset()
outputs = [net.propagate(i) for i in net.dataset.inputs]
plot([["Network", outputs], ["Training data", net.dataset.targets]])

#net.reset(); 
net.delete()

if net.saved():
    net.load()
    net.plot_results()
else:
    net.train(500, batch_size=batch_size, accuracy=1.0, tolerance=0.1, 
              shuffle=False, plot=True, save=True)

outputs = [net.propagate(i)  for i in net.dataset.inputs]
plot([["Network", outputs], ["Training data", net.dataset.targets]])

net.propagate(net.dataset.inputs[0]), net.dataset.targets[0]



