from pycap import PropertyTree, CyclicVoltammetry
import sys
sys.path.append('/notebooks')
from battery import Dualfoil
from matplotlib import pyplot

file_path = '/notebooks/docker/dualfoil5-1/'
df1 = Dualfoil(path=file_path)
init_v1 = df1.get_voltage()
df2 = Dualfoil(path=file_path)
init_v2 = df2.get_voltage()
df3 = Dualfoil(path=file_path)
init_v3 = df3.get_voltage()
assert(init_v1 == init_v2 == init_v3)

# larger voltage interval with larger step
# BIG currents
v1 = 2.5
v2 = 4.5
step_size = .10
ptree1 = PropertyTree()
ptree1.put_double('cycles', 6)
ptree1.put_double('scan_limit_1', v1)
ptree1.put_double('scan_limit_2', v2)
ptree1.put_double('initial_voltage', v1)
ptree1.put_double('final_voltage', v1)
ptree1.put_double('scan_rate', 25e-3)
ptree1.put_double('step_size', step_size)

get_ipython().magic('matplotlib inline')
from pycap import initialize_data, plot_data
data1 = initialize_data()

df1.reset()
exp = CyclicVoltammetry(ptree1)
get_ipython().magic('time exp.run(df1, data1)')

plot_data(data1)

# variable to count the number of steps per cycle
# used to print only one cycle
cycle = int(2*((v2-(v1-step_size)) / step_size))
print(cycle)
# plot last 3 cycles 
y = data1['current'][-3*cycle:]
x = data1['voltage'][-3*cycle:]
pyplot.plot(x, y)

# shorter voltage interval with smaller step
# more reasonable current range
v1 = 4.0
v2 = 3.6
step_size = .0075
ptree2 = PropertyTree()
ptree2.put_double('cycles', 3)
ptree2.put_double('scan_limit_1', v1)
ptree2.put_double('scan_limit_2', v2)
ptree2.put_double('initial_voltage', v1)
ptree2.put_double('final_voltage', v1)
ptree2.put_double('scan_rate', 0.05e-3)
ptree2.put_double('step_size', step_size)

get_ipython().magic('matplotlib inline')
from pycap import initialize_data, plot_data
data2 = initialize_data()

df2.reset()
exp = CyclicVoltammetry(ptree2)
get_ipython().magic('time exp.run(df2, data2)')

plot_data(data2)

# variable to count the number of steps per cycle
# used to print only one cycle
cycle = int(2*((v1-(v2-step_size)) / step_size))
print(cycle)
# don't print the first cycle
y = data2['current'][cycle:]
x = data2['voltage'][cycle:]
pyplot.plot(x, y)

# shorter voltage interval with smaller step
# more reasonable current range
v1 = 4.0
v2 = 3.6
step_size = .0075
ptree3 = PropertyTree()
ptree3.put_double('cycles', 3)
ptree3.put_double('scan_limit_1', v1)
ptree3.put_double('scan_limit_2', v2)
ptree3.put_double('initial_voltage', v1)
ptree3.put_double('final_voltage', v1)
ptree3.put_double('scan_rate', 0.5e-3)
ptree3.put_double('step_size', step_size)

get_ipython().magic('matplotlib inline')
from pycap import initialize_data, plot_data
data3 = initialize_data()

df3.reset()
exp = CyclicVoltammetry(ptree3)
get_ipython().magic('time exp.run(df3, data3)')

plot_data(data3)

# variable to count the number of steps per cycle
# used to print only one cycle
cycle = int(2*((v1-(v2-step_size)) / step_size))
print(cycle)
# don't print the first cycle
y = data3['current'][cycle:]
x = data3['voltage'][cycle:]
pyplot.plot(x, y)



