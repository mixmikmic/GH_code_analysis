import pandas as pd
from dateutil.parser import parse 

from athena.equations import *
from athena.framework import Framework
from athena.dataset import Dataset
from athena.model import AdditiveModel
from athena.helpers import *

np.random.seed(seed = 4)

data_frame = pd.read_csv('test_data.csv')
data_frame = data_frame[["time", "temp", "humidity"]]

def get_hour(x):
    y = parse(x)
    return y.hour + y.minute/60.0

data_frame["time"] = [get_hour(x)/24.0 for x in data_frame["time"].values]

training_df, testing_df = split_dataframe(data_frame, 0.9)

parameters_map = {
    "normalized": {
        
    },
    "not_normalized": {
        "time": "t",
        "humidity": "h",     
    },
    "target": "temp"
}

max_iterations = int(1e4)
starter_learning_rate = 0.0005
momentum = 0.95

framework_parameters = {
    "starting_lr": starter_learning_rate,
    "max_iterations": max_iterations,
    "momentum": momentum,
}

fw = Framework(framework_parameters)

fw.add_dataset(Dataset(training_df, testing_df, parameters_map))

model = AdditiveModel(fw)

training_targets = fw.dataset.training_targets
testing_targets = fw.dataset.testing_targets

model.add(Bias)

for i in range(4):
    model.add(SimpleSinusoidal, "time")

for i in range(2):
    model.add(FlexiblePower, "humidity")


fw.initialize(model, training_targets)

for step in range(int(fw.max_iters + 1)):
    fw.run_learning_step()
    if step % int(fw.max_iters / 10) == 0:
        print("\n", "=" * 40, "\n", round(step / fw.max_iters * 100), "% \n", "=" * 40, sep="", end="\n")
        training_t = training_targets, fw.get_training_predictions()
        testing_t = testing_targets, fw.get_testing_predictions()

        try:
            for j, k in list(zip(["Training", "Testing "], [training_t, testing_t])):
                print(j, end = "\t")
                print_statistics(*k)

        except Exception as e:
            print("Error! {}".format(e))

equation = fw.produce_equation()
fw.session.close()

from sympy import N, nsimplify, init_printing

init_printing()
N(nsimplify(equation, tolerance=1e-4), 2)

get_ipython().magic('pylab inline')
import seaborn as sns
sns.set_style('whitegrid')

from sympy import lambdify
from sympy.abc import t, h

y_axis = lambdify((t, h), equation, "numpy")

plt.figure(figsize=(8, 8))
x_axis = np.linspace(0.0, 1.0, 100)

for humidity in range(5, 100, 20):
    plt.plot(x_axis * 24.0, y_axis(x_axis, np.array([humidity])), label='{}%'.format(humidity))

plt.legend()
plt.xlabel('Time of Day (hrs)', fontsize=16)
plt.ylabel('Temperature (°C)', fontsize=16)
plt.show()

