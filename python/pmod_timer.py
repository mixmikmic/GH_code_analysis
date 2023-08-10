from pynq import Overlay
Overlay("base.bit").download()

from time import sleep
from pynq.iop import Pmod_Timer
from pynq.iop import PMODA

pt = Pmod_Timer(PMODA,0)
pt.stop()

# Generate a 10 ns pulse every period*10 ns
period=100
pt.generate_pulse(period)

# Sleep for 4 seconds and stop the timer
sleep(4)
pt.stop()

# Generate 3 pulses at every 1 us
count=3
period=100
pt.generate_pulse(period, count)

# Generate pulses per 1 us forever
count=0
period=100
pt.generate_pulse(period, count)

pt.stop()

# Detect any event within 10 us
period=1000
pt.event_detected(period)

# Detect any event within 20 ms
period=200000
pt.event_detected(period)

# Count number of events within 10 us
period=1000
pt.event_count(period)

period = pt.get_period_ns()
print("The measured waveform frequency: {} Hz".format(1e9/period))

