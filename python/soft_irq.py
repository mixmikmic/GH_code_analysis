from pynq import MMIO
from pynq import PL
from pynq import Overlay
from pynq.board import LED

import asyncio

import functools
import re
import os

bitstream_name = '/home/xilinx/pynq/bitstream/base_irq_ddr.bit'
tcl_name = '/home/xilinx/pynq/bitstream/base_irq_ddr.tcl'

Overlay(bitstream_name).download()

intc_names = []
intc_map = {}
concat_blocks = {}
nets = []
pins = {}
intc_pins = {}
with open(tcl_name, 'r') as f:
    hier_pat = "create_hier_cell"
    concat_pat = "create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1"
    interrupt_pat = "create_bd_cell -type ip -vlnv xilinx.com:ip:axi_intc:4.1"
    ps7_pat = "create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5"
    prop_pat = "set_property -dict"
    end_prop_pat = "] $"
    config_pat = "CONFIG.NUM_PORTS"
    end_pat = "}\n"
    net_pat = "connect_bd_net -net"
    current_hier = ""
    last_concat = ""
    for line in f:
        if config_pat in line:
            m = re.search('CONFIG.NUM_PORTS \{([0-9]+)\}', line)
            concat_blocks[last_concat] = int(m.groups(1)[0])
        elif hier_pat in line:
            m = re.search('proc create_hier_cell_([^ ]*)', line)
            if m:
                current_hier = m.groups(1)[0] + "/"
        elif prop_pat in line:
            in_prop = True
        elif concat_pat in line:
            m = re.search ('create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 ([^ ]+)', line)
            last_concat = current_hier + m.groups(1)[0]
            concat_blocks[last_concat] = 2
        elif interrupt_pat in line:
            m = re.search ('create_bd_cell -type ip -vlnv xilinx.com:ip:axi_intc:4.1 ([^ ]+)', line)
            intc_names.append(current_hier + m.groups(1)[0])
        elif ps7_pat in line:
            m = re.search ('create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ([^ ]+)', line)
            ps7_name = current_hier + m.groups(1)[0]
        elif end_pat == line:
            current_hier = ""
            # print ('End block')
        elif net_pat in line:
            new_pins = [current_hier + v for v in re.findall('\[get_bd_pins ([^]]+)\]',                                        line,re.IGNORECASE)]
            indexes = set()
            for p in new_pins:
                if p in pins:
                    indexes.add(pins[p])
            if len(indexes) == 0:
                index = len(nets)
                nets.append(set())
            else:
                to_merge = []
                while len(indexes) > 1:
                    to_merge.append(indexes.pop())
                index = indexes.pop()
                for i in to_merge:
                    nets[index] |= nets[i]
            nets[index] |= set(new_pins)
            for p in nets[index]:
                pins[p] = index

ps7_irq_net = pins[ps7_name + "/IRQ_F2P"]

def add_interrupt_pins(net, parent, offset):
    net_pins = nets[net]
    # Find the next item up the chain
    for p in net_pins:
        m = re.match('(.*)/dout', p)
        if m is not None:
            name = m.groups(1)[0]
            if name in concat_blocks:
                print("Found concat: " + name)
                return add_concat_pins(name, parent, offset)
        m = re.match('(.*)/irq', p)
        if m is not None:
            name = m.groups(1)[0]
            if name in intc_names:
                print("Found Intc: " + name)
                add_interrupt_pins(pins[name + "/intr"], name, 0)
                intc_map[name] = (parent, offset)
                return offset + 1
    for p in net_pins:
        intc_pins[p] = (parent, offset)
    return offset + 1

def add_concat_pins(name, parent, offset):
    num_ports = concat_blocks[name]
    for i in range(num_ports):
        net = pins[name + "/In" + str(i)]
        offset = add_interrupt_pins(net, parent, offset)
    return offset
    
        
add_interrupt_pins(ps7_irq_net, "", 0)

print(intc_map)
print()
print(intc_pins)

def get_uio_device(irq):
    dev_name = None
    with open('/proc/interrupts', 'r') as f:
        for line in f:
            cols = line.split()
            if len(cols) >= 6:
                if cols[4] == str(irq):
                    dev_name = cols[5]
    if dev_name is None:
        return None
    for dev in os.listdir("/sys/class/uio"):
        with open('/sys/class/uio/' + dev + '/name', 'r') as f:
            name = f.read().strip()
        if name == dev_name:
            return '/dev/' + dev
    return None
    
print(get_uio_device(61))

class InterruptController(object):
    _controllers = []
    
    @staticmethod
    def get_controller(name):
        for con in InterruptController._controllers:
            if con.name == name:
                return con
        ret = InterruptController(name)
        InterruptController._controllers.append(ret)
        return ret

    @staticmethod
    def on_interrupt(pin, callback):
        parent, number = intc_pins[pin]
        InterruptController.get_controller(parent).set_callback(number, callback)
        
    def __init__(self, name):
        self.name = name
        self.uiodev = None
        self.mmio = MMIO(PL.ip_dict["SEG_" + name + "_Reg"][0],32)
        self.uio = None
        self.callbacks = [None] * 32
        self.events = [None] * 32
        
        # Enable global interrupt
        self.mmio.write(0x1C, 0x00000003)
        
        # Disable Interrupt lines
        self.mmio.write(0x08, 0);
        
        parent, number = intc_map[name]
        if parent == "":
            self.uiodev = get_uio_device(61 + number)
            self.uio = open(self.uiodev, 'r+b', buffering=0)
            # Register callback with asyncio 
            asyncio.get_event_loop().add_reader(self.uio, functools.partial(InterruptController.uio_callback, self))
            # Prime UIO for interrupts
            self.uio.write(bytes([0,0,0,1]))
        else:
            InterruptController.get_controller(parent).set_callback(number,                                         functools.partial(InterruptController.master_callback, self))            
        
    def set_callback(self, num, callback, reset=True):
        self.callbacks[num] = callback
        self.mmio.write(0x10, 1 << num)
        if reset:
            self.mmio.write(0x0C, 1 << num)
        
    def clear_callback(self, num):
        self.mmio.write(0x14, 1 << num)
        self.callbacks[num] = None
    
    def uio_callback(self):
        # print ('In uio callback: ' + self.name)
        # Clear interrupt from UIO
        self.uio.read(4)
        self.master_callback()
        # Prime UIO for interrupts
        self.uio.write(bytes([0,0,0,1]))
        
    def master_callback(self):
        # print ('In master callback: ' + self.name)
        # Pull pending interrupts
        irqs = self.mmio.read(0x04)
        # Call all active IRQs
        work = irqs
        irq = 0
        while work != 0:
            if work % 2 == 1:
                if self.callbacks[irq] is None:
                    raise "Interrupt raised but no callback"
                self.callbacks[irq]()
                if self.events[irq]:
                    self.events[irq].set()
            work = work >> 1
            irq = irq + 1
            
        # Acknowledge the interrupts
        self.mmio.write(0x0C, irqs)
        
    @staticmethod    
    @asyncio.coroutine
    def wait_for_interrupt(pin):
        parent, number = intc_pins[pin]
        yield from InterruptController.get_controller(parent)._wait_for_interrupt(number)
    
    @asyncio.coroutine
    def _wait_for_interrupt(self, num):
        if not self.events[num]:
            self.events[num] = asyncio.Event()
        
        self.events[num].clear()
        #print('_wait_for_interrupt: ' + str(self.events[num].is_set()))
        yield from self.events[num].wait()

@asyncio.coroutine
def wait_for_interrupt(pin):
    parent, number = intc_pins[pin]
    yield from InterruptController.wait_for_interrupt(pin)
        
def register_interrupt(pin, callback):
    InterruptController.on_interrupt(pin, callback)

class InterruptGPIO(object):
    def __init__(self, name, channel=1):
        self.channel = channel - 1
        self.mmio = MMIO(PL.ip_dict["SEG_" + name + "_Reg"][0], 512)
        register_interrupt(name + '/ip2intc_irpt',                                         functools.partial(InterruptGPIO._callback, self))
        self.mmio.write(0x11C, 0x80000000)
        self.mmio.write(0x128, 1 << (self.channel))
        self.mmio.write(0x120, 1 << (self.channel))
        
        self.value = self.mmio.read(4 * self.channel)
        self.callbacks = set()
        
    def _callback(self):
        irqs = self.mmio.read(0x120)
        if irqs & (1 << self.channel) != 0:
            old_value = self.value
            self.value = self.mmio.read()
            for c in self.callbacks:
                c(old_value, self.value)
            self.mmio.write(0x120, 1 << self.channel)
    
    def register_callback(self, c):
        self.callbacks.add(c)
        return c
    
    def unregister_callbacks(self, c):
        self.callbacks.remove(c)
   
class InterruptSwitch(object):
    _gpio = None
    
    def __init__(self, index):
        if InterruptSwitch._gpio is None:
            InterruptSwitch._gpio = InterruptGPIO("swsleds_gpio")
        
        self.callback_handle = InterruptSwitch._gpio.register_callback(functools.partial(InterruptSwitch._callback, self))
        
        self.index = index
        self.next_callback = None
        self.condition = None
         
    def close(self):
        if self.callback_handle is not None:
            InterruptSwitch._gpio.unregister_callback(self.callback_handle)
            
    def _callback(self, old_value, new_value):
        if (old_value ^ new_value) & (1 << self.index) != 0:
            if not self.condition is None:
                asyncio.get_event_loop().create_task(self.signal_waiters())
            if not self.next_callback is None:
                self.next_callback()

    def read(self):
        curr_val = InterruptSwitch._gpio.value
        return (curr_val & (1 << self.index)) >> self.index
            
    def on_change(self, callback):
        self.next_callback = callback
        
    @asyncio.coroutine
    def signal_waiters(self):
        yield from self.condition.acquire()
        self.condition.notify_all()
        self.condition.release()
    
    @asyncio.coroutine
    def wait_for_high(self):
        if self.condition is None:
            self.condition = asyncio.Condition()
        if self.read():
            return
        yield from self.condition.acquire()
        while not self.read():
            yield from self.condition.wait()
        self.condition.release()
        
    @asyncio.coroutine
    def wait_for_low(self):
        if self.condition is None:
            self.condition = asyncio.Condition()
        if not self.read():
            return
        yield from self.condition.acquire()
        while self.read():
            yield from self.condition.wait()
        self.condition.release()        

class InterruptButton(object):
    _gpio = None
    
    def __init__(self, index):
        if InterruptButton._gpio is None:
            InterruptButton._gpio = InterruptGPIO("btns_gpio")
        
        self.callback_handle = InterruptButton._gpio.register_callback(functools.partial(InterruptButton._callback, self))
        
        self.index = index
        self.next_callback = None
        self.condition = None
         
    def close(self):
        if self.callback_handle is not None:
            InterruptButton._gpio.unregister_callback(self.callback_handle)
            
    def _callback(self, old_value, new_value):
        if (old_value ^ new_value) & (1 << self.index) != 0:
            if not self.condition is None:
                asyncio.get_event_loop().create_task(self.signal_waiters())
            if self.next_callback is not None and self.read():
                self.next_callback()

    def read(self):
        curr_val = InterruptButton._gpio.value
        return (curr_val & (1 << self.index)) >> self.index
            
    def on_press(self, callback):
        self.next_callback = callback
        
    @asyncio.coroutine
    def signal_waiters(self):
        yield from self.condition.acquire()
        self.condition.notify_all()
        self.condition.release()
    
    @asyncio.coroutine
    def wait_for_press(self):
        if self.condition is None:
            self.condition = asyncio.Condition()
        if self.read():
            return
        yield from self.condition.acquire()
        while not self.read():
            yield from self.condition.wait()
        self.condition.release()
        
    @asyncio.coroutine
    def wait_for_release(self):
        if self.condition is None:
            self.condition = asyncio.Condition()
        if not self.read():
            return
        yield from self.condition.acquire()
        while self.read():
            yield from self.condition.wait()
        self.condition.release()

from pynq.iop import Pmod_PWM
from pynq.iop import PMODA

pwm = Pmod_PWM(PMODA,0)

from pynq import GPIO
from pynq import PL
print (PL.gpio_dict)
pin = GPIO(GPIO.get_gpio_pin(PL.gpio_dict['iop1_intr_ack'][0]), 'out')

def print_message():
    print('In callback')
    pin.write(1)
    pin.write(0)

    
register_interrupt('iop1/iop1_intr_req', print_message)

import time
import asyncio



# Generate a 10 us clocks with 50% duty cycle
period=10
duty=50
pwm.generate(period,duty)

# Sleep for 4 seconds and stop the timer
loop = asyncio.get_event_loop()
loop.run_until_complete(loop.create_task(asyncio.sleep(4)))
pwm.stop()
pwm.generate(period,duty)
loop.run_until_complete(loop.create_task(asyncio.sleep(4)))
pwm.stop()

from pynq.iop import Pmod_PWM
from pynq.iop import PMODA, PMODB
from pynq.iop import Arduino_LCD18
from pynq.iop import ARDUINO

lcd = Arduino_LCD18(ARDUINO)
pwm_A = Pmod_PWM(PMODA,0)
pwm_B = Pmod_PWM(PMODB,0)

from pynq import GPIO
from pynq import PL
print (PL.gpio_dict)
iop1_pin = GPIO(GPIO.get_gpio_pin(PL.gpio_dict['iop1_intr_ack'][0]), 'out')
iop2_pin = GPIO(GPIO.get_gpio_pin(PL.gpio_dict['iop2_intr_ack'][0]), 'out')
iop3_pin = GPIO(GPIO.get_gpio_pin(PL.gpio_dict['iop3_intr_ack'][0]), 'out')

def iop1_print_message():
    print('In IOP1 callback')
    iop1_pin.write(1)
    iop1_pin.write(0)

def iop2_print_message():
    print('In IOP2 callback')
    iop2_pin.write(1)
    iop2_pin.write(0)

def iop3_print_message():
    print('In IOP3 callback')
    iop3_pin.write(1)
    iop3_pin.write(0)

register_interrupt('iop1/iop1_intr_req', iop1_print_message)
register_interrupt('iop2/iop2_intr_req', iop2_print_message)
register_interrupt('iop3/mb3_intr_req', iop3_print_message)

import time
import asyncio

#Create three independent tasks as coroutines
@asyncio.coroutine
def pwm_task():

    # Generate a 10 us clocks with 50% duty cycle
    period=10
    duty=50
    pwm_A.generate(period*2,duty)
    pwm_B.generate(period,duty)
    yield from asyncio.sleep(4)

    pwm_A.stop()
    pwm_B.stop()
    pwm_A.generate(period,duty)
    pwm_B.generate(period,duty)
    yield from asyncio.sleep(4)
    pwm_A.stop()
    pwm_B.stop()
    print('PWM Finished')

@asyncio.coroutine
def lcd_task():
    lcd.animate('data/download.png',3,0,0,76,25,3,0)
    yield from wait_for_interrupt('iop3/mb3_intr_req')
    print('LCD finished')

@asyncio.coroutine
def print_task():
    while True:
        print('.')
        yield from asyncio.sleep(1)

    
loop = asyncio.get_event_loop()
#Create tasks table having two tasks running
tasks = [
    asyncio.ensure_future(pwm_task()),
    asyncio.ensure_future(lcd_task())
]
#Invoke print task and then wait for the tasks in the task table to finish
task_to_cancel = asyncio.ensure_future(print_task())
loop.run_until_complete(asyncio.gather(*tasks))
#Cancel the print task as the table tasks are finished
task_to_cancel.cancel()
print("Finished!")


# Set the number of Switches
MAX_LEDS = 4
MAX_SWITCHES = 2
MAX_BUTTONS = 4

leds = [0] * MAX_LEDS
switches = [0] * MAX_SWITCHES
buttons = [0] * MAX_BUTTONS

loop = asyncio.get_event_loop()

# Create lists for each of the IO component groups
for i in range(MAX_LEDS):
    leds[i] = LED(i)              
for i in range(MAX_SWITCHES):
    switches[i] = InterruptSwitch(i)      
for i in range(MAX_BUTTONS):
    buttons[i] = InterruptButton(i) 

@asyncio.coroutine
def flash_led(num):
    print('Starting flash: ' + str(num))
    while buttons[num].read():
        leds[num].toggle()
        yield from asyncio.sleep(0.1)
    leds[num].off()
    print('Stopping flash: ' + str(num))

def start_flashing(num):
    loop.create_task(flash_led(num))

for i in range (MAX_BUTTONS):
    buttons[i].on_press(functools.partial(start_flashing, i))


loop.run_until_complete(switches[0].wait_for_high())

print("Finished!")



