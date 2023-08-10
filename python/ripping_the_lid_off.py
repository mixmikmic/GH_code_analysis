import pynq

pynq.__path__

base = pynq.Overlay('base.bit')

dir(base)

base.bitfile_name

base.gpio_dict

base.ip_dict

import pynq.board.button
pynq.board.button.__file__

from pynq import Overlay, PL, MMIO

base = Overlay('base.bit')
base.download()             # Load the PL of the ZYNQ with the bitstream for buttons & LEDs.

# Create MMIO objects for reading the buttons and turning the LEDs on and off.
button_addr  = base.ip_dict['SEG_btns_gpio_Reg'][0]
button_range = base.ip_dict['SEG_btns_gpio_Reg'][1]
button_mmio  = MMIO(button_addr, button_range)
led_addr     = base.ip_dict['SEG_swsleds_gpio_Reg'][0]
led_range    = base.ip_dict['SEG_swsleds_gpio_Reg'][1]
led_mmio     = MMIO(led_addr, led_range)

# For a ten-second interval, read the values of all four buttons and
# display it on all four of the LEDs.
from time import time
end = time() + 10.0
while time() < end:
    buttons = button_mmio.read(0)  # Read memory word containing all four button values.
    led_mmio.write(0x8, buttons)   # Write button values to memory word driving all four LEDs.

from pynq import Overlay
from pynq.board.button import Button
from pynq.board.led import LED

# Create lists of the buttons and LEDs.
buttons = [Button(i) for i in range(4)]
leds = [LED(i) for i in range(4)]

# For a ten-second interval, execute a loop to read the values of each button and
# display it on the associated LED.
from time import time
end = time() + 10.0
while time() < end:
    for i in range(4):
        leds[i].write( buttons[i].read() )
        

