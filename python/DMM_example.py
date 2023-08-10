get_ipython().system('pip3 install --user git+https://bitbucket.org/subinitial/subinitial.git')

import subinitial.stacks as stacks

#connect to Stacks and analog deck
core = stacks.Core(host="192.168.1.49")
analogdeck = stacks.AnalogDeck(core_deck=core, bus_address=2)

#read voltage
voltage = analogdeck.dmm.measure(channel=0)

