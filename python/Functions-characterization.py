import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os
import collections
import matplotlib as mp
mp.style.use('classic')

# Read simulation files
def datafileread(measurename,skipfirstrows):
    # Reading Datafiles
    path = measurename
    data = np.genfromtxt(path,
                        skip_header=skipfirstrows,
                        delimiter=',',
                        dtype=(float,float),
                        unpack=True)
    return data

# measurement
# unp : unpowered
# pow : powered
unp_time, unp_voltage, unp_current, dummy = datafileread('tlp_output_unpowered.csv',14)
pow_time, pow_voltage, pow_current, dummy = datafileread('tlp_output_powered.csv',14)
unp_in_time, unp_in_voltage, unp_in_current, dummy = datafileread('tlp_input_unpowered.csv',14)
pow_in_time, pow_in_voltage, pow_in_current, dummy = datafileread('tlp_input_powered.csv',14)

#
f, ax2 = plt.subplots(1,1,figsize=(10,4))
ax2.plot(pow_voltage,pow_current, label="powered")
ax2.plot(unp_voltage, unp_current, label="unpowered")
#ax2.set_xlim([44, 50])
#ax2.set_ylim([-4, 13])
#ax2.set_title('TLP output characterization')
ax2.set_xlabel('voltage (V)')
ax2.set_ylabel('current (A)')

#
plt.tight_layout()
plt.legend(loc="best")
plt.savefig("../../src/4/figures/tlp_input_characterization.png", pad_inches=0.3)
plt.show()

#
f, ax2 = plt.subplots(1,1,figsize=(5,4))
ax2.plot(pow_voltage,pow_current, label="alimenté")
ax2.plot(unp_voltage, unp_current, label="non-alimenté")
#ax2.set_xlim([44, 50])
#ax2.set_ylim([-4, 13])
#ax2.set_title('TLP output characterization')
ax2.set_xlabel('voltage (V)')
ax2.set_ylabel('current (A)')

#
plt.tight_layout()
plt.legend(loc="best")
plt.savefig("./tlp_input_characterization.png", pad_inches=0.3)
plt.show()

# Model configuration for Verilog-a cell iv_curve_4pts

v0 = -32
i0 = -3
v1 = -1.5
i1 = -0.05
v2 = 2.5
i2 = 0
v3 = 10
i3 = 0.05

#
step = 0.1

#
x1 = np.arange(v0, v1, step)
x2 = np.arange(v1, v2, step)
x3 = np.arange(v2, v3, step)

#
a1 = (i1-i0)/(v1-v0)
a2 = (i2-i1)/(v2-v1)
a3 = (i3-i2)/(v3-v2)

b1 = i1 - a1 * v1
b2 = i2 - a2 * v2
b3 = i3 - a3 * v3

#
y1 = x1 * a1 + b1
y2 = x2 * a2 + b2
y3 = x3 * a3 + b3

#
f, ax2 = plt.subplots(1,1,figsize=(10,4))
ax2.plot(pow_voltage,pow_current, label="powered")
#ax2.plot(unp_voltage, unp_current, label="unpowered")
ax2.plot(x1, y1, label="powered model")
ax2.plot(x2, y2, label="powered model")
ax2.plot(x3, y3, label="powered model")
ax2.set_xlim([-10, 18])
ax2.set_ylim([-1, 1])
#ax2.set_title('TLP output characterization')
ax2.set_xlabel('voltage (V)')
ax2.set_ylabel('current (A)')

#
plt.tight_layout()
plt.legend(loc="best")
plt.show()

#
f, ax2 = plt.subplots(1,1,figsize=(10,4))
ax2.plot(pow_in_voltage, pow_in_current, label="powered")
ax2.plot(unp_in_voltage, unp_in_current, label="unpowered")
#ax2.set_xlim([44, 50])
#ax2.set_ylim([-4, 13])
#ax2.set_title('TLP input characterization')
ax2.set_xlabel('voltage (V)')
ax2.set_ylabel('current (A)')

#
plt.tight_layout()
plt.legend(loc="best")
plt.savefig("../../src/4/figures/tlp_output_characterization.png", pad_inches=0.3)
plt.show()

#
f, ax2 = plt.subplots(1,1,figsize=(5,4))
ax2.plot(pow_in_voltage, pow_in_current, label="alimenté")
ax2.plot(unp_in_voltage, unp_in_current, label="non-alimenté")
ax2.set_xlim([-6, 100])
#ax2.set_ylim([-4, 13])
#ax2.set_title('TLP input characterization')
ax2.set_xlabel('voltage (V)')
ax2.set_ylabel('current (A)')

#
plt.tight_layout()
plt.legend(loc="lower right")
plt.savefig("./tlp_output_characterization.png", pad_inches=0.3)
plt.show()

# Model configuration for Verilog-a cell iv_curve_4pts

v0 = -6.5
i0 = -9
v1 = 1
i1 = 0
v2 = 21
i2 = 0.01
v3 = 35
i3 = 0.45

#
step = 0.01

#
x1 = np.arange(v0, v1, step)
x2 = np.arange(v1, v2, step)
x3 = np.arange(v2, v3, step)

#
a1 = (i1-i0)/(v1-v0)
a2 = (i2-i1)/(v2-v1)
a3 = (i3-i2)/(v3-v2)

b1 = i1 - a1 * v1
b2 = i2 - a2 * v2
b3 = i3 - a3 * v3

#
y1 = x1 * a1 + b1
y2 = x2 * a2 + b2
y3 = x3 * a3 + b3

#
f, ax2 = plt.subplots(1,1,figsize=(10,4))
ax2.plot(pow_in_voltage, pow_in_current, label="powered")
ax2.plot(unp_in_voltage, unp_in_current, label="unpowered")
ax2.plot(x1, y1, label="powered model")
ax2.plot(x2, y2, label="powered model")
ax2.plot(x3, y3, label="powered model")
#ax2.set_xlim([0, 50])
ax2.set_ylim([-7.5, 0.5])
#ax2.set_title('TLP input characterization')
ax2.set_xlabel('voltage (V)')
ax2.set_ylabel('current (A)')

#
plt.tight_layout()
plt.legend(loc="best")
plt.show()

# Model configuration for Verilog-a cell iv_curve_4pts

v0 = -6.5
i0 = -9
v1 = 1
i1 = 0
v2 = 21
i2 = 0.01
v3 = 35
i3 = 0.45

#
step = 0.01

#
x1 = np.arange(v0, v1, step)
x2 = np.arange(v1, v2, step)
x3 = np.arange(v2, v3, step)

#
a1 = (i1-i0)/(v1-v0)
a2 = (i2-i1)/(v2-v1)
a3 = (i3-i2)/(v3-v2)

b1 = i1 - a1 * v1
b2 = i2 - a2 * v2
b3 = i3 - a3 * v3

#
y1 = x1 * a1 + b1
y2 = x2 * a2 + b2
y3 = x3 * a3 + b3

#
f, ax2 = plt.subplots(1,1,figsize=(10,4))
ax2.plot(pow_in_voltage - 0.5, pow_in_current, 'r', label="Characterization")
ax2.plot(x2 - 0.5, y2, label="I(V) curve model")
ax2.plot([2, 2], [-0.01, 0.00075], 'k-', lw=1)
ax2.plot(2, -0.0002, 'o')
ax2.plot(2, 0.00075, 'o')
ax2.set_xlim([1, 4])
ax2.set_ylim([-0.005, 0.005])
ax2.set_xlabel('voltage (V)')
ax2.set_ylabel('current (A)')
ax2.annotate("I = 0.75 mA",[2.1,0.0012])
ax2.annotate("I = 0.2 mA",[2.1,-0.001])
ax2.annotate("2.0 V",[2.1,-0.0045])

#
plt.tight_layout()
plt.legend(loc="best")
plt.savefig("../../src/4/figures/tlp_output_characterization_magnified.png", pad_inches=0.3)

plt.show()



