from pynq import Overlay
Overlay('interface.bit').download()

fsm_spec = {'inputs': [('Run_Counter','D14'), ('Load_Counter', 'D9'), 
                       ('m1', 'D15'), ('m2', 'D16'), ('m3', 'D17')], #, ('m4', 'D18'), ('m5', 'D19')],
            'outputs': [('Clear','D0'), ('Clk','D1'), ('Load_Bit<0>','D2'),
                        ('Load_Bit<1>','D3'),('Load_Bit<2>','D4'),
                        ('Load_Bit<3>','D5'), ('Enable_P','D6'), 
                        ('Enable_T','D7'), ('Load','D8')],
            'states': ['Clear', 'Count_clkn', 'Count_clkp', 'Load_clkn',
                       'Load_clkp'],
            'transitions': [['00---', 'Clear', 'Clear', '000000000'],
                            ['01---', 'Clear', 'Clear', '000000000'],
                            ['10---', 'Clear', 'Count_clkn', '000000000'],
                            ['11---', 'Clear', 'Load_clkn', '000000000'],        
                            ['00---', 'Count_clkn', 'Clear', '100000111'],
                            ['01---', 'Count_clkn', 'Clear', '100000111'],
                            ['10---', 'Count_clkn', 'Count_clkp', '100000111'],
                            ['11---', 'Count_clkn', 'Load_clkn', '100000111'],       
                            ['00---', 'Count_clkp', 'Clear', '110000111'],
                            ['01---', 'Count_clkp', 'Clear', '110000111'],
                            ['10---', 'Count_clkp', 'Count_clkn', '110000111'],
                            ['11---', 'Count_clkp', 'Load_clkn', '110000111'],        
                            ['00---', 'Load_clkn', 'Clear', '101100110'],
                            ['01---', 'Load_clkn', 'Clear', '101100110'],
                            ['10---', 'Load_clkn', 'Load_clkp', '101100110'],
                            ['11---', 'Load_clkn', 'Load_clkp', '101100110'],                            
                            ['00---', 'Load_clkp', 'Clear', '111100110'], 
                            ['01---', 'Load_clkp', 'Clear', '111100110'],
                            ['10---', 'Load_clkp', 'Count_clkn', '111100110'],
                            ['11---', 'Load_clkp', 'Load_clkp', '111100110']]}
            

from pynq.lib import FSMBuilder
from pynq.lib import request_intf

microblaze_intf = request_intf()

fsm = FSMBuilder(microblaze_intf, fsm_spec, num_analyzer_samples=128, use_state_bits=True)

fsm.show_state_diagram()

fsm.arm()

microblaze_intf.run()

fsm.show_waveform()

microblaze_intf.stop()
microblaze_intf.reset_buffers()



