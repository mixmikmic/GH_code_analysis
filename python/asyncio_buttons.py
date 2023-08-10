from pynq import Overlay, PL
from pynq.board import LED, Switch, Button

Overlay('base.bit').download()

buttons = [Button(i) for i in range(4)]
leds = [LED(i) for i in range(4)]
switches = [Switch(i) for i in range(2)]

import asyncio

@asyncio.coroutine
def flash_led(num):
    while True:
        yield from buttons[num].wait_for_value_async(1)
        while buttons[num].read():
            leds[num].toggle()
            yield from asyncio.sleep(0.1)
        leds[num].off()

tasks = [asyncio.ensure_future(flash_led(i)) for i in range(4)]

import psutil

@asyncio.coroutine
def print_cpu_usage():
    # Calculate the CPU utilisation by the amount of idle time
    # each CPU has had in three second intervals
    last_idle = [c.idle for c in psutil.cpu_times(percpu=True)]
    while True:
        yield from asyncio.sleep(3)
        next_idle = [c.idle for c in psutil.cpu_times(percpu=True)]
        usage = [(1-(c2-c1)/3) * 100 for c1,c2 in zip(last_idle, next_idle)]
        print("CPU Usage: {0:3.2f}%, {1:3.2f}%".format(*usage))
        last_idle = next_idle

tasks.append(asyncio.ensure_future(print_cpu_usage()))

if switches[0].read():
    print("Please set switch 0 low before running")
else:
    switches[0].wait_for_value(1)

[t.cancel() for t in tasks]

switches[0].wait_for_value(0)

