import conx as cx
import jyro.simulator as jy
import math
import time

def make_world(physics):
    physics.addBox(0, 0, 8, 2, fill="backgroundgreen", wallcolor="gray")

MAX_SENSOR_DISTANCE = 6 # meters
    
def make_robot():
    robot = jy.Pioneer("Pioneer", 7.5, 1, math.pi/2)  #parameters are x, y, heading (in radians)
    robot.addDevice(jy.PioneerFrontSonar(MAX_SENSOR_DISTANCE))
    robot.addDevice(jy.Camera())
    return robot

robot = make_robot()
robot

def collect_data(simulator):
    data = []
    simulator.reset() # put robot back to where it is defined
    while True:
        scaled_dist = simulator.robot["sonar"].getData()[0]/MAX_SENSOR_DISTANCE
        # The power is a function of distance:
        power = 1.0 - ((1 - scaled_dist)/0.33 * 0.9) 
        robot.move(power, 0)
        data.append([[scaled_dist], [power]])
        simulator.step()
        time.sleep(.1) # don't overwhelm the network
        if power < 0.05:
            break
    return data

sim = jy.VSimulator(robot, make_world, size=(700, 180))

data = collect_data(sim)

len(data)

train = ["Training Data", [pair[1] for pair in data]]

cx.shape(train[0])

cx.plot(train, 
        title="Speed as a Function of Distance",
        xlabel="distance from target",
        ylabel="speed",
        xs=[pair[0] for pair in data], default_symbol="o")

net = cx.Network("Go To Target")
net.add(cx.Layer("input", 1))
net.add(cx.Layer("hidden", 10, activation="sigmoid"))
net.add(cx.Layer("output", 1, activation = "linear"))
net.connect()
net.compile(loss="mse", optimizer="sgd", lr=.1, momentum=.5)
net.model.summary()

net.dataset.load(data)
net.dataset.info()
net.dataset.summary()

net.dashboard()

if net.saved():
    net.load()
    net.plot_results()
else:
    net.train(400, accuracy=1.0, tolerance=0.05, batch_size=1, save=True, plot=True)

test = ["Network", [net.propagate(pair[0])[0] for pair in data]]

cx.plot([train, test], 
        title="Speed as a Function of Distance",
        xlabel="distance from target",
        ylabel="speed",
        default_symbol="o",
        xs=[pair[0] for pair in data])

def net_brain(robot):
    scaled_distance = robot["sonar"].getData()[0]/6
    output = net.propagate([scaled_distance])[0]
    robot.move(output, 0)
    outputs.append([[scaled_distance], output])
    history.append(robot.getPose())
        
robot.brain = net_brain

outputs = []
history = []
sim.reset()
sim.display()

trained_range = ["Network interpolation", outputs]

cx.scatter(trained_range,
           title="Network Generalization", 
           xlabel="input", ylabel="output", default_symbol="o")

len(history)

def replay_history(index):
    pose = history[index]
    robot.setPose(*pose)
    sim.update()
    return sim.canvas.render(format="pil")

cx.movie(replay_history, "generalize-in-range.gif", (0, len(history)))

robot.setPose(.5, 1)
sim.update_gui()

outputs = []
history = []
sim.display()

cx.movie(replay_history, "generalize-out-range.gif", (0, len(history)))

untrained_range = ["Network extrapolation", outputs]

cx.scatter([trained_range, untrained_range],
           title="Network Generalization", 
           xlabel="input", ylabel="output", default_symbol="o")

