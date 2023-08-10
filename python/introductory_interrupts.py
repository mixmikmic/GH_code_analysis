from pynq import Overlay
from pynq.board import Switch

# Make sure the base overlay is installed in the ZYNQ PL.
Overlay('base.bit').download()

sw0 = Switch(0)        # Create Switch object for SW0.
sw0.wait_for_value(1)  # Push SW0 up to terminate this cell.
print('SW0 is 1!')

import asyncio
from psutil import cpu_percent
from pynq import Overlay
from pynq.board import Switch

# Make sure the base overlay is installed in the ZYNQ PL.
Overlay('base.bit').download()

# Create objects for both slide switches.
switches = [Switch(i) for i in range(2)]

# Coroutine that waits for a switch to change state.
async def show_switch(sw):
    while True:

        # Wait for the switch to change and then print its state.
        await sw.interrupt.wait()  # Wait for the interrupt to happen.
        print('Switch[{num}] = {val}'.format(num=sw.index, val=sw.read()))

        # Clear the interrupt.
        if Switch._mmio.read(0x120) & 0x1:
            Switch._mmio.write(0x120, 0x00000001)

# Create a task for each switch using the coroutine and place them on the event loop.
tasks = [asyncio.ensure_future(show_switch(sw)) for sw in switches]
    
# Create a simple coroutine that just waits for a time interval to expire.
async def just_wait(interval):
    await asyncio.sleep(interval)

# Run the event loop until the time interval expires,
# printing the switch values as they change.
time_interval = 10  # time in seconds
loop = asyncio.get_event_loop()
wait_task = asyncio.ensure_future(just_wait(time_interval))

# Surround the event loop with functions to record CPU utilization.
cpu_percent(percpu=True)  # Initialize the CPU monitoring.
loop.run_until_complete(wait_task)
cpu_used = cpu_percent(percpu=True)

# Print the CPU utilization % for the interval.
print('CPU Utilization = {cpu_used}'.format(**locals()))

# Remove all the tasks from the event loop.
for t in tasks:
    t.cancel()

def scan_switch(sw):
    try:
        sw_val = sw.read()  # Get the switch state.
        
        # Print the switch state if it has changed.
        if sw.prev != sw_val:
            print('Switch[{num}] = {val}'.format(num=sw.index, val=sw_val))
            
    except AttributeError:
        # An exception occurs the 1st time thru because the switch state
        # hasn't yet been stored in the object as an attribute.
        pass
    
    # Save the current state of the switch inside the switch object.
    sw.prev = sw_val

# Compute the end time for the polling.
from time import time
end = time() + 10.0

cpu_percent(percpu=True)  # Initialize the CPU monitoring.

# Now poll the switches for the given time interval.
while time() < end:
    for sw in switches:
        scan_switch(sw)
        
# Print the CPU utilization during the polling.
cpu_used = cpu_percent(percpu=True)
print('CPU Utilization = {cpu_used}'.format(**locals()))

