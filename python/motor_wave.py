import time
import subinitial.stacks as stacks
import numpy as np

core = stacks.Core(host="192.168.1.49")  # Default host IP
analogdeck = stacks.AnalogDeck(core, bus_address=2)  # Default Analog Deck bus address

# Set the Wavegen to apply a continuously-looping waveform instead of a DC voltage
analogdeck.wavegen.set_control(analogdeck.wavegen.MODE_DC)
analogdeck.dio.set_config(dio0_3_innotout=False, dio4_7_innotout=True)  # Set 0-3 as outputs, 4-7 as inputs

while True:
    for set_voltage in range(0,4):
        analogdeck.wavegen.set_dc(volts=set_voltage)
        
        measured_voltage = analogdeck.dmm.measure_channel(channel=0)
        print("DMM measured voltage: ", int(round(measured_voltage)))

        analogdeck.dio.set(int(round(measured_voltage)))
        time.sleep(0.5)
        
    for set_voltage in range(4, -1, -1):
        analogdeck.wavegen.set_dc(volts=set_voltage)
        
        measured_voltage = analogdeck.dmm.measure_channel(channel=0)
        print("DMM measured voltage: ", int(round(measured_voltage)))

        analogdeck.dio.clear(int(round(measured_voltage)))
        time.sleep(0.5)
    

analogdeck.dio.clear(0,1,2,3)



