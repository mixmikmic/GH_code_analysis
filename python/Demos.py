# -*- coding: UTF-8 -*-
# Analog Deck RgbLed Example
# © 2012-2015 Subinitial LLC. All Rights Reserved
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met: Redistributions of source code and binary forms must retain the above copyright notice,
# these conditions, and the disclaimer below all in plain text. The name Subinitial may not be used to endorse or
# promote products derived from this software without prior written consent from Subinitial LLC. This software may only
# be redistributed and used with a Subinitial Stacks product.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. Refer to the included LICENSE.txt for more details.
import time
import subinitial.stacks as stacks

core = stacks.Core(host="192.168.1.49")  # Default host IP
analogdeck = stacks.AnalogDeck(core, bus_address=2)  # Default Analog Deck bus address

analogdeck.dio.set_config(dio0_3_innotout=False, dio4_7_innotout=True)  # Set 0-3 as outputs, 4-7 as inputs

state = True
# set DIO3 output to follow the input of DIO5
analogdeck.dio.clear(0,1,2,3)
while(state):
    
    time.sleep(2)
    
    analogdeck.dio.set(5)

    if analogdeck.dio.get_pin_status(5):
        analogdeck.dio.set(0,1,2,3)
    else:
        analogdeck.dio.clear(0,1,2,3)



