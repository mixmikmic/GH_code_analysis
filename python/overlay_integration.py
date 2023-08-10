BASE_ADDRESS          = 0x40000000
ADDRESS_LENGTH        = 0x1000
CMD_OFFSET            = 0x800
ACK_OFFSET            = 0x804
INPUT_DATA_OFFSET     = 0x0
OUTPUT_DATA_OFFSET    = 0x400
        
from pynq import MMIO

class my_new_accelerator:
    """Python class for the PL Acclererator.
    
    Attributes
    ----------
    mmio : MMIO
        MMIO object that can be read / written between PS and PL.
    array_length : int
        Length of the array to be processed.
       
    """
    def __init__(self):
        self.mmio = MMIO(BASE_ADDRESS,ADDRESS_LENGTH)
        self.array_length = 0
        self.mmio.write(CMD_OFFSET, 0x0)
     
    def load_data(self, input_data):
        self.array_length = len(input_data)
        for i in range(self.array_length):
            self.mmio.write(INPUT_DATA_OFFSET + i * 4, input_data[i])
            
    def process(self):     
        # Send start command to accelerator
        self.mmio.write(CMD_OFFSET, 0x1)
        output_data = [0] * self.array_length
        
        # ACK is set to check for 0x0 in the ACK offset
        while (self.mmio.read(ACK_OFFSET)) != 0x1:
            pass
        
        # Ack has been received
        for i in range(self.array_length):
            output_data[i] = self.mmio.read(OUTPUT_DATA_OFFSET + i * 4)
            
        # Reset Ack
        self.mmio.write(ACK_OFFSET, 0x0)      
        return output_data

from pynq import Overlay
Overlay("base.bit").download()

# declare accelerator with an array length of 10
acc = my_new_accelerator()
input_data = [i for i in range(10)]
print("Data to be sent to the accelerator:", input_data)
acc.load_data(input_data)

from pynq import MMIO
       
mmio = MMIO(BASE_ADDRESS, ADDRESS_LENGTH)

for i in range(len(input_data)):
    mmio.write(OUTPUT_DATA_OFFSET + i * 4, input_data[i] + 1)

mmio.write(ACK_OFFSET, 1)

output_data = acc.process()
print("Input Data   : ", input_data)
print("Output Data  : ", output_data)

