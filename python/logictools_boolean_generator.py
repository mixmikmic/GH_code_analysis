from pynq import Overlay
from pynq.lib.logictools import BooleanGenerator
from pynq.lib.logictools import ARDUINO

Overlay('logictools.bit').download()

single_expression = {'addr_decode': 'D4 = D0 & D1 & D2 & D3'}

boolean_generator = BooleanGenerator(ARDUINO)
boolean_generator.status

boolean_generator.trace(use_analyzer=True)
boolean_generator.setup(single_expression)
boolean_generator.status

boolean_generator.expressions['addr_decode']

boolean_generator.input_pins

boolean_generator.output_pins

boolean_generator.run()

boolean_generator.status

boolean_generator.show_waveform()

boolean_generator.stop()
boolean_generator.status

multiple_expressions = ["LD0 = D0",
                        "LD1 = D0 & PB1",
                        "D19 = D0",
                        "D9 = PB0 & PB1 & PB2",
                        "D10 = D0 & D1 & D2",
                        "D11 = D0 & D1 & D2 & D3"]

boolean_generator.reset()
boolean_generator.status

boolean_generator.setup(multiple_expressions)
boolean_generator.status

print(f"Input pins are {', '.join(boolean_generator.input_pins)}.")
print(f"Output pins are {', '.join(boolean_generator.output_pins)}.")

boolean_generator.run()
boolean_generator.status

boolean_generator.show_waveform()

boolean_generator.stop()
boolean_generator.status

del boolean_generator

