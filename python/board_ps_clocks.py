# Example PS Register Access Python
# Calculating system clockrates from PS Register Values

from pynq import MMIO as mmio

# Zynq PS Constants
SCLR_BASE             = 0xf8000000
ZYNQ_NUM_FCLKS        = 4
FCLK_CTRL_REG_OFFSETS = [0x170,0x180,0x190,0x1a0]


def get_reg_value(addr):
    '''Returns register value at address given.'''
    return get_regfield_value(addr,0,0xffffffff)

def get_regfield_value(addr,shift,mask):
    '''Returns register field value at address.'''
    currval = mmio(addr).read()
    return (currval & (mask << shift)) >> shift

def get_zynq_clockrates(src_clockrate=50):
    '''Returns zynq system clockrates dictionary (in MHz).
    
    The returned dictionary has the following contents:
    'cpu'  : Cortex-A9 freqency
    'fclk0': PL fclk0 frequency
    'fclk1': PL fclk1 frequency
    'fclk2': PL fclk2 frequency
    'fclk3': PL fclk3 frequency
     
    '''  
    # Read Clock Registers from Zynq Memory Map
    arm_pll_fdiv = get_regfield_value(SCLR_BASE+0x100,12,0x7f)
    ddr_pll_fdiv = get_regfield_value(SCLR_BASE+0x104,12,0x7f)    
    io_pll_fdiv = get_regfield_value(SCLR_BASE+0x108,12,0x7f)
    
    arm_clk_sel = get_regfield_value(SCLR_BASE+0x120,4,0x3)
    arm_clk_div  = get_regfield_value(SCLR_BASE+0x120,8,0x3f)    
    
    fclk_config = list()
    for ix,offset in enumerate(FCLK_CTRL_REG_OFFSETS): 
        fclk_config.append(dict())
        fclk_config[ix]["src"] = get_regfield_value(
                                    SCLR_BASE+offset,4,0x3)
        fclk_config[ix]["div0"] = get_regfield_value(
                                    SCLR_BASE+offset,8,0x3f)  
        fclk_config[ix]["div1"] = get_regfield_value(
                                    SCLR_BASE+offset,20,0x3f)
        
    # Calculate Clock rates based on register reads above
    clock_values = list()
    
    # Arm clock
    if arm_clk_sel == 0 or arm_clk_sel == 1 :
        arm_clk_mult = arm_pll_fdiv
    elif arm_clk_sel == 2:
        arm_clk_mult = ddr_pll_fdiv
    else:
        arm_clk_mult = io_pll_fdiv
        
    armclk_value = src_clockrate*arm_clk_mult/arm_clk_div  
    clock_values.append({"cpu" : armclk_value})
    
    # x4 fclks
    for ix in range(4):
        if fclk_config[ix]["src"] == 0 or                     fclk_config[ix]["src"] == 1:
            fclk_mult = io_pll_fdiv            
        elif fclk_config[ix]["src"] == 2:
            fclk_mult = arm_pll_fdiv
        else:
            fclk_mult = ddr_pll_fdiv
                
        fclk_div0 = fclk_config[ix]["div0"]
        fclk_div1 = fclk_config[ix]["div1"]
    
        fclk_value = src_clockrate*fclk_mult/                             (fclk_div0*fclk_div1)
        clock_values.append({"fclk" + str(ix) :                                      round(fclk_value,2)})
        
    return clock_values

from pprint import pprint
pprint(get_zynq_clockrates())



