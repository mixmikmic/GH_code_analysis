from pprint import pprint
from pynq import Overlay
from pynq.lib.logictools import FSMGenerator
from pynq.lib.logictools import PatternGenerator
from pynq.lib.logictools import BooleanGenerator
from pynq.lib.logictools import LogicToolsController
from pynq.lib.logictools import ARDUINO

Overlay('logictools.bit').download()

logictools_controller = LogicToolsController(
                            ARDUINO, 'PYNQZ1_LOGICTOOLS_SPECIFICATION')
pprint(logictools_controller.status)

fsm_spec = {'inputs': [('reset','D0'), ('direction','D1')],
        'outputs': [('bit2','D3'), ('bit1','D4'), ('bit0','D5')],
        'states': ['S0', 'S1', 'S2', 'S3', 'S4', 'S5'],
        'transitions': [['00', 'S0', 'S1', '000'],
                        ['01', 'S0', 'S5', '000'],
                        ['00', 'S1', 'S2', '001'],
                        ['01', 'S1', 'S0', '001'],
                        ['00', 'S2', 'S3', '010'],
                        ['01', 'S2', 'S1', '010'],
                        ['00', 'S3', 'S4', '011'],
                        ['01', 'S3', 'S2', '011'],
                        ['00', 'S4', 'S5', '100'],
                        ['01', 'S4', 'S3', '100'],
                        ['00', 'S5', 'S0', '101'],
                        ['01', 'S5', 'S4', '101'],
                        ['1-', '*',  'S0', '']]}

fsm_generator = FSMGenerator(ARDUINO)
fsm_generator.trace(use_analyzer=True, num_analyzer_samples=128)
fsm_generator.setup(fsm_spec)
fsm_generator.show_state_diagram()

loopback_test = {'signal': [
    ['stimulus',
        {'name': 'clk6',  'pin': 'D6', 'wave': 'l...h...' * 16},
        {'name': 'clk7',  'pin': 'D7', 'wave': 'l.......h.......' * 8},
        {'name': 'clk8',  'pin': 'D8', 'wave': 'lh' * 16}, 
        {'name': 'clk9',  'pin': 'D9', 'wave': 'l.h.' * 32}, 
        {'name': 'clk10', 'pin': 'D10', 'wave': 'l...h...' * 16},
        {'name': 'clk11', 'pin': 'D11', 'wave': 'l.......h.......' * 8},
        {'name': 'clk12', 'pin': 'D12', 'wave': 'lh' * 8},
        {'name': 'clk13', 'pin': 'D13', 'wave': 'l.h.' * 32}], 
      
    ['analysis',
        {'name': 'clk6',  'pin': 'D6'},
        {'name': 'clk7',  'pin': 'D7'},
        {'name': 'clk8',  'pin': 'D8'},
        {'name': 'clk9',  'pin': 'D9'},
        {'name': 'clk10', 'pin': 'D10'},
        {'name': 'clk11', 'pin': 'D11'},
        {'name': 'clk12', 'pin': 'D12'},
        {'name': 'clk13', 'pin': 'D13'}]], 

    'foot': {'tock': 1, 'text': 'Loopback Test'},
    'head': {'tick': 1, 'text': 'Loopback Test'}}

pattern_generator = PatternGenerator(ARDUINO)
pattern_generator.trace(use_analyzer=True, num_analyzer_samples=128)
pattern_generator.setup(loopback_test, 
                        stimulus_group_name='stimulus', 
                        analysis_group_name='analysis')
pattern_generator.waveform.display()

expressions = ["LD0 = D14",
               "LD1 = D15",
               "D18 = PB0 | PB1",
               "D19 = D16 & D17"]

boolean_generator = BooleanGenerator(ARDUINO)
boolean_generator.trace(use_analyzer=True, num_analyzer_samples=128)
boolean_generator.setup(expressions)

pprint(logictools_controller.status)

logictools_controller.run([boolean_generator, 
                           pattern_generator, 
                           fsm_generator])

pprint(logictools_controller.status)

fsm_generator.show_waveform()

pattern_generator.show_waveform()

boolean_generator.show_waveform()

logictools_controller.stop([boolean_generator, 
                            pattern_generator, 
                            fsm_generator])

pprint(logictools_controller.status)

from time import sleep

for _ in range(5):
    logictools_controller.step([boolean_generator, 
                                pattern_generator, 
                                fsm_generator])
    sleep(1)

for _ in range(5):
    logictools_controller.step([boolean_generator, 
                                pattern_generator, 
                                fsm_generator])
    sleep(1)

fsm_generator.show_waveform()

pattern_generator.show_waveform()

boolean_generator.show_waveform()

logictools_controller.stop([boolean_generator, 
                            pattern_generator, 
                            fsm_generator])

pprint(logictools_controller.status)

del boolean_generator, pattern_generator, fsm_generator

