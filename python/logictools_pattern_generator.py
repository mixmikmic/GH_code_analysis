from pynq import Overlay
from pynq.lib.logictools import Waveform
from pynq.lib.logictools import PatternGenerator
from pynq.lib.logictools import ARDUINO

Overlay('logictools.bit').download()

loopback_test = {'signal': [
    ['stimulus',
        {'name': 'clk0', 'pin': 'D0', 'wave': 'lh' * 64},
        {'name': 'clk1', 'pin': 'D1', 'wave': 'l.h.' * 32},
        {'name': 'clk2', 'pin': 'D2', 'wave': 'l...h...' * 16},
        {'name': 'clk3', 'pin': 'D3', 'wave': 'l.......h.......' * 8},
        {'name': 'clk4', 'pin': 'D4', 'wave': 'lh' * 32},
        {'name': 'clk5', 'pin': 'D5', 'wave': 'l.h.' * 32},
        {'name': 'clk6', 'pin': 'D6', 'wave': 'l...h...' * 16},
        {'name': 'clk7', 'pin': 'D7', 'wave': 'l.......h.......' * 8},
        {'name': 'clk8', 'pin': 'D8', 'wave': 'lh' * 16}, 
        {'name': 'clk9', 'pin': 'D9', 'wave': 'l.h.' * 32}, 
        {'name': 'clk10', 'pin': 'D10', 'wave': 'l...h...' * 16},
        {'name': 'clk11', 'pin': 'D11', 'wave': 'l.......h.......' * 8},
        {'name': 'clk12', 'pin': 'D12', 'wave': 'lh' * 8},
        {'name': 'clk13', 'pin': 'D13', 'wave': 'l.h.' * 32},
        {'name': 'clk14', 'pin': 'D14', 'wave': 'l...h...' * 16},
        {'name': 'clk15', 'pin': 'D15', 'wave': 'l.......h.......' * 8},
        {'name': 'clk16', 'pin': 'D16', 'wave': 'lh' * 4},
        {'name': 'clk17', 'pin': 'D17', 'wave': 'l.h.' * 32},
        {'name': 'clk18', 'pin': 'D18', 'wave': 'l...h...' * 16}, 
        {'name': 'clk19', 'pin': 'D19', 'wave': 'l.......h.......' * 8}], 
      
    ['analysis',
        {'name': 'clk10', 'pin': 'D10'},
        {'name': 'clk01', 'pin': 'D1'},
        {'name': 'clk02', 'pin': 'D2'},
        {'name': 'clk03', 'pin': 'D3'},
        {'name': 'clk04', 'pin': 'D4'},
        {'name': 'clk05', 'pin': 'D5'},
        {'name': 'clk06', 'pin': 'D6'},
        {'name': 'clk07', 'pin': 'D7'},
        {'name': 'clk08', 'pin': 'D8'},
        {'name': 'clk09', 'pin': 'D9'},
        {'name': 'clk0', 'pin': 'D0'},
        {'name': 'clk11', 'pin': 'D11'},
        {'name': 'clk12', 'pin': 'D12'},
        {'name': 'clk13', 'pin': 'D13'},
        {'name': 'clk14', 'pin': 'D14'},
        {'name': 'clk15', 'pin': 'D15'},
        {'name': 'clk16', 'pin': 'D16'},
        {'name': 'clk17', 'pin': 'D17'},
        {'name': 'clk18', 'pin': 'D18'},
        {'name': 'clk19', 'pin': 'D19'}]], 

    'foot': {'tock': 1},
    'head': {'text': 'Loopback Test'}}

waveform = Waveform(loopback_test)
waveform.display()

pattern_generator = PatternGenerator(ARDUINO)

pattern_generator.trace(num_analyzer_samples=128)

pattern_generator.setup(loopback_test,
                        stimulus_group_name='stimulus',
                        analysis_group_name='analysis',
                        frequency_mhz=10)

pattern_generator.run()

pattern_generator.show_waveform()

pattern_generator.stop()

from time import sleep

for _ in range(10):
    pattern_generator.step()
    sleep(1)

pattern_generator.show_waveform()

pattern_generator.step()
pattern_generator.show_waveform()

pattern_generator.reset()

del pattern_generator

