# Make sure the base overlay is loaded
from pynq import Overlay
Overlay("base.bit").download()

from pynq.iop import Grove_LEDbar
from pynq.iop import ARDUINO
from pynq.iop import ARDUINO_GROVE_G4

# Instantiate Grove LED Bar on Arduino shield G4
ledbar = Grove_LEDbar(ARDUINO,ARDUINO_GROVE_G4)
ledbar.reset()

from time import sleep

# Light up different bars in a loop
for i in range(2):
    ledbar.write_binary(0b1010100000)
    sleep(0.5)
    ledbar.write_binary(0b0000100100)
    sleep(0.5)
    ledbar.write_binary(0b1010101110)
    sleep(0.5)
    ledbar.write_binary(0b1111111110)
    sleep(0.5)

# Brightness 0-255
HIGH = 0xFF
MED  = 0xAA
LOW  = 0x01
OFF  = 0X00

brightness = [OFF, OFF, OFF, LOW, LOW, MED, MED, HIGH, HIGH, HIGH]

ledbar.write_brightness(0b1111111111,brightness)

for i in range (1,11):
    ledbar.write_level(i,3,0)
    sleep(0.3)
for i in range (1,10):
    ledbar.write_level(i,3,1)
    sleep(0.3)    

from pynq.board import Button

btns = [Button(index) for index in range(4)] 
i = 1
ledbar.reset()

done = False
while not done:
    if (btns[0].read()==1):
        sleep(0.2)
        ledbar.write_level(i,2,1)
        i = min(i+1,9)
    elif (btns[1].read()==1):
        sleep(0.2)
        i = max(i-1,0)
        ledbar.write_level(i,2,1)
    elif (btns[3].read()==1):
        ledbar.reset()
        done = True

