from pynq import Overlay
Overlay("base.bit").download()

from pynq.iop import Grove_Buzzer
from pynq.iop import PMODB
from pynq.iop import PMOD_GROVE_G1 

grove_buzzer = Grove_Buzzer(PMODB, PMOD_GROVE_G1)

grove_buzzer.play_melody()

# Play a tone
tone_period = 1200
num_cycles = 500
grove_buzzer.play_tone(tone_period,num_cycles)

from pynq.iop import ARDUINO
from pynq.iop import Arduino_Analog
from pynq.iop import ARDUINO_GROVE_A1

analog1 = Arduino_Analog(ARDUINO, ARDUINO_GROVE_A1)

rounds = 200

for i in range(rounds):
    tone_period = int(analog1.read_raw()[0]/5)
    num_cycles = 500
    grove_buzzer.play_tone(tone_period,50)

