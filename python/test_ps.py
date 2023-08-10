from pynq import Clocks
from pynq import Overlay

ol = Overlay("base.bit")
ol.download()

print(f'CPU:   {Clocks.cpu_mhz:.6f}MHz')
print(f'FCLK0: {Clocks.fclk0_mhz:.6f}MHz')
print(f'FCLK1: {Clocks.fclk1_mhz:.6f}MHz')
print(f'FCLK2: {Clocks.fclk2_mhz:.6f}MHz')
print(f'FCLK3: {Clocks.fclk3_mhz:.6f}MHz')

Clocks.fclk0_mhz = 27.123456
Clocks.fclk1_mhz = 31.436546
Clocks.fclk2_mhz = 14.597643
Clocks.fclk3_mhz = 53.894231

print(f'CPU:   {Clocks.cpu_mhz:.6f}MHz')
print(f'FCLK0: {Clocks.fclk0_mhz:.6f}MHz')
print(f'FCLK1: {Clocks.fclk1_mhz:.6f}MHz')
print(f'FCLK2: {Clocks.fclk2_mhz:.6f}MHz')
print(f'FCLK3: {Clocks.fclk3_mhz:.6f}MHz')

# Set fclk0 to 10MHz
Clocks.set_fclk(0,div0=10,div1=10)
# Set fclk1 to 12.5MHz
Clocks.set_fclk(1,clk_mhz=12.500000)
# Set fclk2 to 20MHz
Clocks.set_fclk(2,div0=5,clk_mhz=25.000000)
# Set fclk3 to 50MHz
Clocks.set_fclk(3,div1=2,clk_mhz=50.000000)

# Show all the modified PL clocks
print(f'FCLK0: {Clocks.fclk0_mhz:.6f}MHz')
print(f'FCLK1: {Clocks.fclk1_mhz:.6f}MHz')
print(f'FCLK2: {Clocks.fclk2_mhz:.6f}MHz')
print(f'FCLK3: {Clocks.fclk3_mhz:.6f}MHz')

ol.download()

print(f'FCLK0: {Clocks.fclk0_mhz:.6f}MHz')
print(f'FCLK1: {Clocks.fclk1_mhz:.6f}MHz')
print(f'FCLK2: {Clocks.fclk2_mhz:.6f}MHz')
print(f'FCLK3: {Clocks.fclk3_mhz:.6f}MHz')



