from pynq import Overlay
Overlay("base.bit").download()

from pynq.iop import Arduino_LCD18
from pynq.iop import ARDUINO

lcd = Arduino_LCD18(ARDUINO)

lcd.clear()

lcd.display('data/board_small.jpg',x_pos=0,y_pos=127,
            orientation=3,background=[255,255,255])

lcd.display('data/logo_small.png',x_pos=0,y_pos=127,
            orientation=1,background=[255,255,255],frames=100)

lcd.clear()
lcd.draw_line(x_start_pos=151,y_start_pos=98,x_end_pos=19,y_end_pos=13)

lcd.draw_line(50,50,150,50,color=[255,0,0],background=[255,255,0])

lcd.draw_line(50,20,50,120,[0,0,255],[255,255,0])

text = 'Hello, PYNQ!'
lcd.print_string(1,1,text,[255,255,255],[0,0,255])

import time
text = time.strftime("%d/%m/%Y")
lcd.print_string(5,10,text,[255,255,0],[0,0,255])

lcd.draw_filled_rectangle(x_start_pos=10,y_start_pos=10,
                          width=60,height=80,color=[64,255,0])

lcd.draw_filled_rectangle(x_start_pos=20,y_start_pos=30,
                          width=80,height=30,color=[255,128,0])

lcd.draw_filled_rectangle(x_start_pos=90,y_start_pos=40,
                          width=70,height=120,color=[64,0,255])

button=lcd.read_joystick()
if button == 1:
    print('Left')
elif button == 2:
    print('Down')
elif button==3:
    print('Center')
elif button==4:
    print('Right')
elif button==5:
    print('Up')
else:
    print('Not pressed')    



