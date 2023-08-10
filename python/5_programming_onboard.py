from pynq import Overlay
from pynq.board import LED
from pynq.board import RGBLED
from pynq.board import Switch
from pynq.board import Button

Overlay("base.bit").download()

led0 = LED(0)

led0.on()

led0.off()

import time
from pynq.board import LED
from pynq.board import Button

led0 = LED(0)
for i in range(20):
    led0.toggle()
    time.sleep(.1)

MAX_LEDS = 4
MAX_SWITCHES = 2
MAX_BUTTONS = 4

leds = [0] * MAX_LEDS
switches = [0] * MAX_SWITCHES
buttons = [0] * MAX_BUTTONS

for i in range(MAX_LEDS):
    leds[i] = LED(i)              
for i in range(MAX_SWITCHES):
    switches[i] = Switch(i)      
for i in range(MAX_BUTTONS):
    buttons[i] = Button(i) 

# Helper function to clear LEDs
def clear_LEDs(LED_nos=list(range(MAX_LEDS))):
    """Clear LEDS LD3-0 or the LEDs whose numbers appear in the list"""
    for i in LED_nos:
        leds[i].off()
        
clear_LEDs()

clear_LEDs()

for i in range(MAX_LEDS):                  
    if switches[i%2].read():                                    
        leds[i].on()
    else:
        leds[i].off()

import time

clear_LEDs()

while switches[0].read():
    for i in range(MAX_LEDS):
        if buttons[i].read():
            leds[i].toggle()
            time.sleep(.1)
            
clear_LEDs()

