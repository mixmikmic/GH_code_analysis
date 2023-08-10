from pynq import Overlay

Overlay('interface.bit').download()

fsm_spec = {'inputs': [('not_start','D14'),('Q','D15'),
                       ('E','D16')],
        'outputs': [('Clear','D0'), ('Clk','D1'), ('A','D2'), ('B','D3'), 
                    ('C','D4'), ('D','D5'), ('Enable_P','D8'), ('Load','D7'),
                    ('Enable_T','D6')],
        'states': ['S0', 'S1', 'S2'],
        'transitions': [['0--', 'S0', 'S0', '000000000'],
                        ['1--', 'S0', 'S1', '000000000'],
                        ['1--', 'S1', 'S2', '000000000'],
                        ['0--', 'S1', 'S0', '000000000'],
                        ['1--', 'S2', 'S1', '110000111'],
                        ['0--', 'S2', 'S0', '110000111'],
                        ['0--', '*',  'S0', '']]}
            

from pynq.intf import ARDUINO
from pynq.intf import FSMGenerator

fsm = FSMGenerator(ARDUINO, fsm_spec, use_analyzer=True)

fsm.display()

fsm.start(num_samples=40, frequency_mhz=10)
fsm.waveform.display()

fsm.data_samples

fsm.transitions



