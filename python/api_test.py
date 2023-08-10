import pynq

class Overlay:
    def __init__(self, bitstream, hardware_dict, download=True):
        self.raw_overlay = pynq.Overlay(bitstream)
        if download:
            self.raw_overlay.download()
        self.hardware_dict = hardware_dict
        
    def __getattr__(self, name):
        if name in self.hardware_dict:
            setattr(self, name, self.hardware_dict[name]())
        return getattr(self, name)
    
    def __dir__(self):
        return self.hardware_dict.keys()

from pynq.board import Switch,LED,Button
import functools

class ArrayWrapper:
    def __init__(self, cls, num):
        self.elems = [None for i in range(num)]
        self.cls = cls
        
    def __getitem__(self, val):
        if not self.elems[val]:
            self.elems[val] = self.cls(val)
        return self.elems[val]
    
    def __len__(self):
        return len(self.elems)
    
BaseSwitches = functools.partial(ArrayWrapper, Switch, 2)    
BaseLEDs = functools.partial(ArrayWrapper, LED, 4)
BaseButtons = functools.partial(ArrayWrapper, Button, 4)

from pynq.drivers import HDMI, Audio

class BaseOverlay(Overlay):
    def __init__(self):
        hardware_dict = {
            'switches' : BaseSwitches,
            'leds' : BaseLEDs,
            'buttons' : BaseButtons,
            'hdmi_in' : functools.partial(HDMI, 'in'),
            'hdmi_out' : functools.partial(HDMI, 'out'),
            'audio' : Audio
        }
        Overlay.__init__(self, 'base.bit', hardware_dict)
        

base = BaseOverlay()

dir(base)

base.leds[1].on()
base.audio.load("/home/xilinx/pynq/drivers/tests/pynq_welcome.pdm")
base.audio.play()
print(base.switches[0].read())

class Peripheral:
    def __init__(self, iop_class, *args, **kwargs):
        self.iop_class = iop_class
        self.args = args
        self.kwargs = kwargs
        
    def __call__(self, if_id):
        return self.iop_class(if_id, *self.args, **self.kwargs)

from pynq.iop import PMODA, PMODB, ARDUINO

iop_map = {
    'pmoda' : PMODA, 
    'pmodb' : PMODB,
    'arduino' : ARDUINO
}

class IOPOverlay(BaseOverlay):
    def __init__(self):
        BaseOverlay.__init__(self)
        
    def __dir__(self):
        return BaseOverlay.__dir__(self) + ['pmoda', 'pmodb', 'arduino']
    
    def __setattr__(self, name, val):
        if name in iop_map:
            obj = val(iop_map[name])
        else:
            obj = val
        BaseOverlay.__setattr__(self, name, obj)

base = IOPOverlay()

from pynq.iop import Pmod_OLED
base.pmodb = Peripheral(Pmod_OLED)
base.pmodb.write('Hello World')

from pynq.iop import Grove_LEDbar
import pynq.iop
base.pmoda = Peripheral(Grove_LEDbar, pynq.iop.PMOD_GROVE_G3)
base.pmoda.write_level(5, 3, 1)

from pynq.iop import PMOD_GROVE_G1, PMOD_GROVE_G2, PMOD_GROVE_G3, PMOD_GROVE_G4

grove_map = {
    'G1' : PMOD_GROVE_G1,
    'G2' : PMOD_GROVE_G2,
    'G3' : PMOD_GROVE_G3,
    'G4' : PMOD_GROVE_G4,
}

class GroveAdapter:
    def __init__(self, if_id):
        self.if_id = if_id
        
    def __setattr__(self, name, val):
        if name in grove_map:
            obj = val(self.if_id, grove_map[name])
        else:
            obj = val
        object.__setattr__(self, name, obj)

base = IOPOverlay()
base.pmoda = GroveAdapter
base.pmoda.G3 = Grove_LEDbar
base.pmoda.G3.write_level(10, 3, 1)

class SingleTone(object):
    __instance = None
    def __new__(cls, val):
        if SingleTone.__instance is None:
            SingleTone.__instance = object.__new__(cls)
        SingleTone.__instance.val = val
        return SingleTone.__instance

a = SingleTone(1)
print(f'Value in a is {a.val}')

b = SingleTone(2)
print(f'Value in b is {b.val}')
print(f'Value in a is {a.val}')

a.__class__.__name__

class Parent():
    def __init__(self, age, gender):
        self.age = age
        self.gender = gender
    def get_older(self):
        self.age += 1

class Boy(Parent):
    __person = None
    __born = False
    __instance_list = set()
    def __new__(cls, age, color):
        if cls.__person is None:
            cls.__person = Parent.__new__(cls)
        cls.__person.age = age
        cls.__instance_list.add(cls.__person)
        return cls.__person
    def __init__(self, age, color):
        if not self.__class__.__born:
            self.age = age
            self.haircolor = color
            self.__class__.__born = True
    def get_list(self):
        return self.__class__.__instance_list
    def __del__(self):
        self.__class__.instance_list.pop()

age1 = 9
age2 = 15
tom = Boy(age1, 'BLACK')
print(f'Last year, age of Tom: {tom.age}')
print(f'Last year, haircolor of Tom: {tom.haircolor}')
jack = Boy(age2, 'RED')
print(f'After {age2-age1} years, age of Jack: {jack.age}')
print(f'After {age2-age1} years, haircolor of Jack: {jack.haircolor}')
print(f'After {age2-age1} years, age of Tom: {tom.age}')
print(f'After {age2-age1} years, haircolor of Tom: {tom.haircolor}')
tom.get_older()
print(f'This year, age of Tom: {tom.age}')
print(f'This year, haircolor of Tom: {tom.haircolor}')
print(f'This year, age of Jack: {jack.age}')
print(f'This year, haircolor of Jack: {jack.haircolor}')

jack.get_list()

class RootLicense():
    def __init__(self, date, time):
        self.date = date
        self.time = time

class License(RootLicense):
    __root = list()
    __license_index = 0
    __num_licenses = 3
    __instance_dict = {}
    def __new__(cls, date, time):
        if len(cls.__root) < cls.__num_licenses:
            cls.__root.append(RootLicense.__new__(cls))
        current_license_index = cls.__license_index
        cls.__license_index = (cls.__license_index + 1) % cls.__num_licenses
        cls.__instance_dict[current_license_index] =             cls.__root[current_license_index]
        return cls.__root[current_license_index]
    def __init__(self, date, time):
        super().__init__(date, time)
    def get_instance(self):
        return self.__class__.__instance_dict
    def __del__(self):
        current_license_index = cls.__license_index
        cls.__license_index = (cls.__license_index - 1) % cls.__num_licenses
        self.__class__.__instance_dict[current_license_index] = None

license0 = License('06-21-2017', '10:33:21')
license1 = License('06-23-2017', '09:12:12')
license2 = License('06-24-2017', '00:56:08')
print(f'License 0 issued on: {license0.date}-{license0.time}')
print(f'License 1 issued on: {license1.date}-{license1.time}')
print(f'License 2 issued on: {license2.date}-{license2.time}')
license3 = License('06-24-2017', '08:55:24')
license4 = License('06-25-2017', '07:26:37')
license5 = License('06-25-2017', '19:37:18')
license6 = License('06-26-2017', '13:23:24')
print(f'License 0 issued on: {license0.date}-{license0.time}')
print(f'License 1 issued on: {license1.date}-{license1.time}')
print(f'License 2 issued on: {license2.date}-{license2.time}')

license0.get_instance()

del(license0)
license1.get_instance()

BUILDER_STATUS_DICT = {'BOOLEAN_BUILDER': 1,
                       'PATTERN_BUILDER': 2,
                       'FSM_BUILDER': 3,
                       'TRACE_ANALYZER': 4}
for a in BUILDER_STATUS_DICT.keys():
    print(a)

license0.__class__.__name__.upper()

sys.platform
sys.version_info

import json
import os
import IPython.core.display

def draw_wavedrom(data):
    """Display the waveform using the Wavedrom package.

    This method requires 2 javascript files to be copied locally. Users
    can call this method directly to draw any wavedrom data.

    Example usage:

    >>> a = {
        'signal': [
            {'name': 'clk', 'wave': 'p.....|...'},
            {'name': 'dat', 'wave': 'x.345x|=.x', 
                            'data': ['head', 'body', 'tail', 'data']},
            {'name': 'req', 'wave': '0.1..0|1.0'},
            {},
            {'name': 'ack', 'wave': '1.....|01.'}
        ]}
    >>> draw_wavedrom(a)

    """
    htmldata = '<script type="WaveDrom">' + json.dumps(data) + '</script>'
    IPython.core.display.display_html(IPython.core.display.HTML(htmldata))
    jsdata = 'WaveDrom.ProcessAll();'
    IPython.core.display.display_javascript(
        IPython.core.display.Javascript(
            data=jsdata,
            lib=[relative_path + '/js/WaveDrom.js', 
                 relative_path + '/js/WaveDromSkin.js']))

a = {'signal': [
            {'name': 'clk', 'wave': 'p.....|...'},
            {'name': 'dat', 'wave': 'x.345x|=.x', 
                            'data': ['head', 'body', 'tail', 'data']},
            {'name': 'req', 'wave': '0.1..0|1.0'},
            {},
            {'name': 'ack', 'wave': '1.....|01.'}
        ]}

draw_wavedrom(a)

PYNQ_JUPYTER_NOTEBOOKS = '/home/xilinx/jupyter_notebooks'

import os
current_path = os.getcwd()
print(current_path)

relative_path = os.path.relpath(PYNQ_JUPYTER_NOTEBOOKS, current_path)
print(relative_path)

PYNQZ1_DIO_SPECIFICATION = {'clock_mhz': 10,
                            'interface_width': 20,
                            'monitor_width': 64,
                            'traceable_outputs': {'D0': 0,
                                                  'D1': 1,
                                                  'D2': 2,
                                                  'D3': 3,
                                                  'D4': 4,
                                                  'D5': 5,
                                                  'D6': 6,
                                                  'D7': 7,
                                                  'D8': 8,
                                                  'D9': 9,
                                                  'D10': 10,
                                                  'D11': 11,
                                                  'D12': 12,
                                                  'D13': 13,
                                                  'D14': 14,
                                                  'D15': 15,
                                                  'D16': 16,
                                                  'D17': 17,
                                                  'D18': 18,
                                                  'D19': 19
                                                  },
                            'traceable_inputs': {'D0': 20,
                                                 'D1': 21,
                                                 'D2': 22,
                                                 'D3': 23,
                                                 'D4': 24,
                                                 'D5': 25,
                                                 'D6': 26,
                                                 'D7': 27,
                                                 'D8': 28,
                                                 'D9': 29,
                                                 'D10': 30,
                                                 'D11': 31,
                                                 'D12': 32,
                                                 'D13': 33,
                                                 'D14': 34,
                                                 'D15': 35,
                                                 'D16': 36,
                                                 'D17': 37,
                                                 'D18': 38,
                                                 'D19': 39
                                                 },
                            'traceable_tri_states': {'D0': 42,
                                                     'D1': 43,
                                                     'D2': 44,
                                                     'D3': 45,
                                                     'D4': 46,
                                                     'D5': 47,
                                                     'D6': 48,
                                                     'D7': 49,
                                                     'D8': 50,
                                                     'D9': 51,
                                                     'D10': 52,
                                                     'D11': 53,
                                                     'D12': 54,
                                                     'D13': 55,
                                                     'D14': 56,
                                                     'D15': 57,
                                                     'D16': 58,
                                                     'D17': 59,
                                                     'D18': 60,
                                                     'D19': 61
                                                     },
                            'non_traceable_inputs': {'PB0': 20,
                                                     'PB1': 21,
                                                     'PB2': 22,
                                                     'PB3': 23
                                                     },
                            'non_traceable_outputs': {'LD0': 20,
                                                      'LD1': 21,
                                                      'LD2': 22,
                                                      'LD3': 23
                                                      }
                            }

pin_list = list(set(PYNQZ1_DIO_SPECIFICATION['traceable_outputs'].keys())|
           set(PYNQZ1_DIO_SPECIFICATION['traceable_inputs'].keys())|
           set(PYNQZ1_DIO_SPECIFICATION['non_traceable_outputs'].keys())|
           set(PYNQZ1_DIO_SPECIFICATION['non_traceable_inputs'].keys()))

from pynq import PL
PL.__class__.__name__

from collections import OrderedDict
key_list = ['key3', 'key2']
value_list = [3, 2]
a = OrderedDict({k: v for k, v in zip(key_list,value_list)})

a[list(a.keys())[0]] = 4

a

for i,j in zip(a.keys(), key_list):
    print(i,j)

import os
os.environ['PYNQ_JUPYTER_NOTEBOOKS']

num_input_bits = 4
for i in range(1<< num_input_bits):
    bin_value = format(i, f'0{num_input_bits}b')
    print(bin_value)

'1'+'0'.zfill(5)

from pynq.lib.dio.waveform import draw_wavedrom
a = {
        'signal': [['stimulus',
            {'name': 'clk', 'wave': 'p'},
            {'name': 'dat', 'wave': '1'},
            {'name': 'req', 'wave': '0'}],
            {},
            {'name': 'ack', 'wave': '1'}
        ]}
draw_wavedrom(a)

from random import choice

operations = ['&', '|', '^']
for i in range(5):
    operation = choice(operations)
    print(operation.join([str(i), str(i+1)]))

import numpy as np
a = np.array([1,2,3])[:2]
print(a)
len(a)

3 % 1

from math import ceil
tile = np.array([0,0,1])
a = np.tile(tile, ceil(4/4))
print(a)

a = {'foot': {'tick':1, 'text':'Test'},
    'head': {'tock':1}}
for annotation in ['tick', 'tock']:
    for key in a:
        if annotation in a[key]:
            del a[key][annotation]
print(a)

a = '123'
b = '345'
if a[-1] == b[0]:
    c = a + '.' + b[1:]
print(c)

b = [5, 6, 7]
a = [1,2,3,4]
print(b + a.append(a.pop(0))) 



