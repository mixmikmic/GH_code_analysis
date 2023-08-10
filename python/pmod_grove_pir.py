from time import sleep
from pynq import Overlay
from pynq.board import LED
from pynq.iop import Grove_PIR
from pynq.iop import PMODA
from pynq.iop import PMOD_GROVE_G1

ol1 = Overlay("base.bit")
ol1.download()

pir = Grove_PIR(PMODA,PMOD_GROVE_G1)

led = LED(0)
led.on()

if pir.read()==0:
    print("Starting detection...")
    while True:
        led.on()
        sleep(0.1)
        led.off()
        sleep(0.1)
        if pir.read()==1:
            print("Detected a motion.")
            break
print("Ending detection...")

del pir
del led
del ol1

