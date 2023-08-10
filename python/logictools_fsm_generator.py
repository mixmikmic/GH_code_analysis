from pynq import Overlay
from pynq.lib.logictools import FSMGenerator
from pynq.lib.logictools import ARDUINO

Overlay('logictools.bit').download()

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

fsm_generator.trace(num_analyzer_samples=128)

fsm_generator.setup(fsm_spec, use_state_bits=False)

fsm_generator.show_state_diagram()

fsm_generator.run()
fsm_generator.show_waveform()

fsm_generator.stop()

fsm_generator.reset()

from pprint import pprint

fsm_spec = {'inputs': [('rst','D0'), ('direction','D1')],
        'outputs': [('overflow','D2')],
        'states': ['S0', 'S1', 'S2', 'S3'],
        'transitions': [['00', 'S0', 'S1', '0'],
                        ['01', 'S0', 'S3', '0'],
                        ['00', 'S1', 'S2', '0'],
                        ['01', 'S1', 'S0', '0'],
                        ['00', 'S2', 'S3', '0'],
                        ['01', 'S2', 'S1', '0'],
                        ['00', 'S3', 'S0', '1'],
                        ['01', 'S3', 'S2', '1'],
                        ['1-', '*',  'S0', '']]}

fsm_generator.setup(fsm_spec, use_state_bits=True, 
                  frequency_mhz=50)
pprint(fsm_generator.fsm_spec)

fsm_generator.show_state_diagram()

fsm_generator.run()
fsm_generator.show_waveform()

fsm_generator.stop()

fsm_generator.reset()
fsm_generator.trace(num_analyzer_samples=20)
fsm_generator.setup(fsm_spec, use_state_bits=True)
print(f'FSM generator is in {fsm_generator.status} state.')
print(f'Trace analyzer is in {fsm_generator.analyzer.status} state.')

from time import sleep

for _ in range(5):
    fsm_generator.step()
    sleep(1)

fsm_generator.show_waveform()

fsm_generator.step()
fsm_generator.show_waveform()

for _ in range(5):
    fsm_generator.step()
    sleep(1)

fsm_generator.show_waveform()

fsm_generator.stop()

fsm_generator.run()
fsm_generator.show_waveform()

fsm_generator.stop()
fsm_generator.reset()
del fsm_generator

