import numpy
import matplotlib.pyplot as plt
import usbtmc
get_ipython().magic('matplotlib inline')

# this connects to the instrument directly using it's USB ID:
handle = usbtmc.Instrument(int("1AB1",16),int("0641",16))

# This will probably timeout the first time. Just run it again
handle.ask('*IDN?')

# set voltage high
handle.write(":VOLT:HIGH 2.00")

handle.write(":SOUR1:APPL:USER 100,10,5,0")

N = 40
x = numpy.array(range(40))
data = 0.99*numpy.exp(-((x-20)/4)**2)
plt.plot(data,"o-")

handle.write(":DATA:POINTS VOLATILE,{}".format(N))

handle.ask(":DATA:POINTS? VOLATILE")

def VfromI(Intensity):
    """Implement the inverted response function. See data fit in google drive AOM folder."""
    V = (.0039757327 + (.0039757327 ** 2 + 4 *.0078826605 * Intensity) ** (1/2))/(2*.0078826605)
    return V

voltages = VfromI(data)
floats = voltages/voltages.max() # values scaled to 0-1.0
floats

handle.write(":DATA:POINTS VOLATILE,40")

numPoints = int(handle.ask(":DATA:POINTS? VOLATILE"))
numPoints

for i in range(len(floats)):
    command_string = ":DATA:VAL VOLATILE," + str(i+1) + "," + str(int(0.9*16383*floats[i]))
    check_string = ":DATA:VAL? VOLATILE," + str(i+1)
    #print(command_string)
    handle.write(command_string)
    #print(handle.ask(check_string))

# Check what the instrument memory holds
# For some reason, it can only pull 38 values.
wave = []
# add 1 to numPoints to account for range function
for i in range(1,numPoints+1):
    #sleep(0.2)
    wave.append( handle.ask(":DATA:VALUE? VOLATILE,{}".format(i)) )
    
print(wave)
print(len(wave))
plt.plot(wave,"o-")

handle.close()



