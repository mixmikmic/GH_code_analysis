get_ipython().system('pip3 install --user git+https://bitbucket.org/subinitial/subinitial.git')

import subinitial.stacks as stacks

core = stacks.Core(host="192.168.1.49")  # Default host IP
analogdeck = stacks.AnalogDeck(core, bus_address=2)  # Default Analog Deck bus address

""" BASIC USAGE """

analogdeck.sourcemeter.set_sourcevoltage(2) # Set the sourcemeter output voltage

#Demo sourcemeter internal measurements
print("This should show 2V: {0}".format(analogdeck.sourcemeter.get_sourcevoltage()))
print("Show the measured telemetry:")
print("Voltage (V): {0}".format(analogdeck.sourcemeter.get_metervoltage()))
print("Current (A): {0}".format(analogdeck.sourcemeter.get_metercurrent()))

print("Show the measured power:")
print("Power(W): {0}".format(analogdeck.sourcemeter.get_meterpower()));

