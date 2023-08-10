from time import sleep
from pynq.board import LED
from pynq.board import RGBLED
from pynq.board import Button

Delay1 = 0.3
Delay2 = 0.1
color = 0
leds = [LED(index) for index in range(4)]
rgbleds = [RGBLED(index) for index in [4,5]] 
btns = [Button(index) for index in range(4)]
        
for led in leds:
    led.on()    
while (btns[3].read()==0):
    if (btns[0].read()==1):
        color = (color+1) % 8
        for rgbled in rgbleds:
            rgbled.write(color)
        sleep(Delay1)
        
    elif (btns[1].read()==1):
        for led in leds:
            led.off()
        sleep(Delay2)
        for led in leds:
            led.toggle()
            sleep(Delay2)
            
    elif (btns[2].read()==1):
        for led in leds[::-1]:
            led.off()
        sleep(Delay2)
        for led in leds[::-1]:
            led.toggle()
            sleep(Delay2)                  
    
print('End of this demo ...')
for led in leds:
    led.off()
for rgbled in rgbleds:
    rgbled.off()
    
del leds,btns

