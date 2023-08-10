import numpy as np
from pynq import MMIO

fclk0 = MMIO(0xF8000170)
hex(fclk0.read())

class register:
    
    def __init__(self,buffer):    
        self.buffer = buffer.astype(np.uint32,copy=False)
        
    def __getitem__(self, index):
        currval = int.from_bytes(self.buffer, byteorder='little')
        if isinstance(index, int):
            mask = 1<<index
            return (currval & mask) >> index
        elif isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            if step is None or step==-1:
                if start is None:
                    start = 31
                if stop is None:
                    stop = 0
            elif step == 1:
                if start is None:
                    start = 0
                if stop is None:
                    stop = 31
            else:
                raise ValueError("Slicing step is not valid.")

            if start>=stop:
                mask = ((1<<(start-stop+1)) - 1) << stop  
                return (currval & mask) >> stop
            else:
                length = stop-start+1
                mask = ((1<<length) - 1) << start
                regval = (currval & mask) >> start
                return int('{:0{length}b}'.format(regval, 
                                                  length=length)[::-1], 2)    
        else:
            raise ValueError("Index must be int or slice.")

    def __setitem__(self, index, value):
        currval = int.from_bytes(self.buffer, byteorder='little')
        if isinstance(index, int):
            if value!=0 and value!=1:
                raise ValueError("Value to be set should be either 0 or 1.")
            mask = 1<<index
            self.buffer[0] = (currval & ~mask) | (value << index)
        elif isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            if step is None or step==-1:
                if start is None:
                    start = 31
                if stop is None:
                    stop = 0
            elif step == 1:
                if start is None:
                    start = 0
                if stop is None:
                    stop = 31
            else:
                raise ValueError("Slicing step is not valid.")

            if start>=stop:
                mask = ((1<<(start-stop+1)) - 1) << stop  
                self.buffer[0] = (currval & ~mask) | (value << stop)
            else:
                length = stop-start+1
                mask = ((1<<length) - 1) << start
                regval = int('{:0{length}b}'.format(value, 
                                                    length=length)[::-1], 2)
                self.buffer[0] = (currval & ~mask) | (regval << start)
        else:
            raise ValueError("Index must be int or slice.")
    
    def __str__(self):
        currval = int.from_bytes(self.buffer, byteorder='little')
        return hex(currval)

# Give a register view over the MMIO's numpy array
fclk0_reg = register(fclk0.array)

print("{}:{}  regval:   {}".format(31,0,hex(fclk0_reg[31:0])))
print("{}:{} div1:     {}".format(25,20,hex(fclk0_reg[25:20])))
print("{}:{}  div0:     {}".format(13,8,hex(fclk0_reg[13:8])))
print("{}:{}   srcsel:   {}".format(5,4,hex(fclk0_reg[5:4])))

fclk0_reg[21] = 0x1

print("{}:{}  regval:   {}".format(31,0,hex(fclk0_reg[31:0])))
print("{}:{} div1:     {}".format(25,20,hex(fclk0_reg[25:20])))
print("{}:{}  div0:     {}".format(13,8,hex(fclk0_reg[13:8])))
print("{}:{}   srcsel:   {}".format(5,4,hex(fclk0_reg[5:4])))

print("MMIO and Register sharing buffer? {}".format(fclk0.read() == fclk0_reg[:]))

print(fclk0_reg)



