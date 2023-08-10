get_ipython().magic('run ThetaS.ipynb')
# get best_class consensus_keys
# the firmware
firmwares = ["ap1_v110.frm", "ap1_v130.frm", "ap1_v131.frm", "gy1_v121.frm", "gy1_v130.frm"]
simple_filenames = [f+'.part1' for f in firmwares] + [f+'.part2' for f in firmwares]
filenames = [os.path.sep.join(['firmwares', 'parts', f]) for f in simple_filenames]

firmwares_parts = dict([(os.path.basename(fname),FirmwarePart(fname)) for fname in filenames])

keyfilename = os.path.sep.join(['firmwares', 'keys', 'consensus.pkeys'])
key_manager.add_keys_from_file('consensus', keyfilename)
kt_attack_keys = key_manager.get_bitview('consensus')
try:
    keyfilename = os.path.sep.join(['firmwares', 'keys', 'bitgen-attack.pkeys'])
    key_manager.add_keys_from_file('generated', keyfilename)
except IOError as e:
    pass

int_keys = kt_attack_keys.keys()
valid_slice_1 = slice(1300, 2704) 
s_valid_slice_1 = slice(1300, 1364) # from ap1/theta m15
s_valid_slice_2 = slice(3102, 3166) # from gy1/theta S
all_keys = slice(0, len(int_keys))

# let's look at XOR KEY bits for a second.
print kt_attack_keys.show_key_slice(slice(1300,1316))

# Lets look at them VERTICALLY.
## BIT are from 0 to 32 - 4 b, with bit 0 leftmost and bit 32 rightmost (reverse convention)
## show same pattern in two point in files.
print "Vertical showing of XOR keys from", s_valid_slice_1
kt_attack_keys.show_bit_sliced(s_valid_slice_1)
print ''
print "Vertical showing of XOR keys from", s_valid_slice_2
kt_attack_keys.show_bit_sliced(s_valid_slice_2)

# focusing on bit (0,1) (8,9) (16,17) and (24,25) should give us most value for clear text attack
# as alphabetic characters are depending on these bit.( 0,1 == text)

kt_attack_keys.show_bit_vhexdump(1, 36, valid_slice_1)

# Bit Generator for bit 00
class Bit0_Generator(Generator):
    # base 0011, then skip a bit every 34 or 36 bit, alternating every 15 or 17 cycles
    def bit(self):
        c =''
        if self.counter == self.l1_val:
            c = self.tick_base.tick()
            # skip a beat
            self.tick_base.tick()
            self.counter = 0
            self.l1_val = self.tick_layer1.tick()
            if self.tick_layer1.counter == 0:
                mult = self.tick_layer2.tick()
                #print 'wrap layer2', self.i, "mtul", mult
                self.tick_layer1 = CyclicalTicker(mult, 34, [0,2,4,6,8,10,12,14,16], 36, 0)
        else:
            self.counter += 1
            c = self.tick_base.tick()
        return c

    def reset(self):
        self.counter = self.init
        self.tick_base = CyclicalTicker(4, "0", [0,1], "1")
        self.tick_layer1 = CyclicalTicker(17, 34, [0,2,4,6,8,10,12,14,16], 36, 0)
        self.l1_val = self.tick_layer1.tick()
        self.l2_counter = 1
        self.tick_layer2 = CyclicalTicker(6, 15, [1,3], 17)

# Bit Generator for bit 01
class Bit1_Generator(Generator):
    # bit1: '0110', '1001' inserts alternating every 14*(36 sections) then 37 sections then 15*(36 sections) then 37 sections
    # the pattern are inserted alternatively, until we hit a 37.
    # layer 1 reproduces the pattern of 14*36 + 37 + 15*36 + 37
    def __init__(self, init, m1,v1,t2,v2,l1i=-1):
        self.init = init
        self.values = (m1,v1,t2,v2,l1i)
        self.reset()
        
    def bit(self):
        c =''
        if self.counter >= self.cycle_layer1:
            # insert special artefact then go back to normal tick
            c = self.tick_layer2.tick()
            self.counter = 0
            # Check when is the next tick cycle (35 or 36)
            self.cl1_prev = self.cycle_layer1
            self.cycle_layer1 = self.tick_layer1.tick()
            # if we change cycle, reverse layer2 bit
            if self.tick_layer1.is_val2 and (self.cycle_layer1 != self.cl1_prev):
                #reverse it
                c = self.tick_layer2.tick()
        else:
            self.counter += 1
            c = self.tick_base.tick()
        return c

    def reset(self):
        self.counter = self.init
        self.tick_base = CyclicalTicker(2, "0", [0], "1") #checked
        ## self.tick_layer1 = CyclicalTicker(31, 35, [0,16], 36)
        self.tick_layer1 = CyclicalTicker(*self.values)
        # inserts a bit
        self.tick_layer2 = CyclicalTicker(2, "0", [1], "1") 
        # self.tick_layer3 = CyclicalTicker(4, "0", [0,1], "1") 
        self.cycle_layer1 = self.tick_layer1.tick()

# Bit Generator for bit 02
class Bit2_Generator(Generator):
    # bit2: ('1'*9 + '0'*9 )*16 + insert '0' or '1'
    def bit(self):
        c =''
        if self.tick_base.tick():
            c = self.base_val
        else:
            # check if we need to insert a bit every 33/35 sequence
            if self.tick_layer0.tick():
                self.tick_layer0 = CyclicalTicker(self.tick_layer1.tick(), False, [0], True, 0)
                # skip the bit
                self.tick_base.counter -=1
                c = self.base_val
            else:
                #wrapping 1 or 0 sequences, change the bit base_val
                self.base_val = self.tick_base2.tick() 
                c = self.base_val
        return c

    def reset(self):
        self.counter = self.init
        self.tick_base = CyclicalTicker(18, True, [0], False, self.init)
        self.tick_base2 = CyclicalTicker(2, "1", [1], "0")
        self.base_val = self.tick_base2.tick() 
        #
        self.tick_layer1 = CyclicalTicker(8, 32, [2,5], 33)
        self.tick_layer2 = CyclicalTicker(6, "0", [2,3], "1")
        self.tick_layer0 = CyclicalTicker(self.tick_layer1.tick(), False, [0], True, 0)
        
# Bit Generator for bit 03
class Bit3_Generator(Generator):
    def bit(self):
        c =''
        if self.tick_base.tick():
            c = self.base_val
        else:
            # check if we need to insert a bit every 33/35 sequence
            if self.tick_layer0.tick():
                self.tick_layer0 = CyclicalTicker(self.tick_layer1.pop(0), False, [0], True, 0)
                # skip the bit
                self.tick_base.counter -=1
                c = self.base_val
            else:
                #wrapping 1 or 0 sequences, change the bit base_val
                self.base_val = self.tick_base2.tick() 
                c = self.base_val
        return c

    def reset(self):
        self.counter = self.init
        self.tick_base = CyclicalTicker(9, True, [0], False, self.init)
        self.tick_base2 = CyclicalTicker(2, "1", [1], "0")
        self.base_val = self.tick_base2.tick() 
        self.tick_layer1 = [63,63,65,63,64,64,64,63,63,63, 63,63,63,63,63,63,63]
        self.tick_layer2 = CyclicalTicker(6, "0", [2,3], "1")
        self.tick_layer0 = CyclicalTicker(self.tick_layer1.pop(0), False, [0], True, 0)

# Bit Generator for bit 04
class Bit4_Generator(Generator):
    def bit(self):
        c =''
        self.counter += 1
        if not self.tick_base.tick():
            if self.tick_layer0.tick():
                self.tick_base3.tick()
                self.tick_layer0 = CyclicalTicker(self.tick_layer1.pop(0), False, [0], True, 0)
                # if self.tick_layer0.mod_val == 61: [1981, 3960, 5939]
            self.tick_base = CyclicalTicker(self.tick_base3.tick(), True, [0], False, 0)
            self.base_val = self.tick_base2.tick() 
        c = self.base_val
        return c

    def reset(self):
        self.counter = 0
        self.tick_base3 = CyclicalTicker(2, 4, [1], 5)
        self.tick_base = CyclicalTicker(self.tick_base3.tick(), True, [0], False, self.init)
        self.tick_base2 = CyclicalTicker(2, "0", [0], "1")
        self.base_val = self.tick_base2.tick() 
        #[4] is at 0x000b0000
        # fix 00183800
        self.tick_layer1 = [64,63,63,63,63,63,63,61,63,63,63,63,63,63,61,63,63,63,63,63,63,61,63,63,63,63,63,63,61,63,63,63,63,63,63]
        #self.tick_layer2 = CyclicalTicker(6, "0", [2,3], "1")
        self.tick_layer0 = CyclicalTicker(self.tick_layer1.pop(0), False, [0], True, 2)


# Bit Generator for bit 05
class Bit5_Generator(Generator):
    # 1100111 001100111x15 (14 sometimes) 
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.reset()
        
    def bit(self):
        self.counter += 1
        c = self.base_val
        #
        if not self.tick_base1.tick(): 
            # change base
            self.base_val = self.tick_base.tick()
            # if 15 or 14, wrap line
            if not self.tick_base2.tick():
                #print 'base2', self.counter
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                # skip a layer 1 group too and start on 7
                self.tick_layer1.reset()
                # skip one
                self.tick_layer1.tick()
                #if self.counter > 1400 and self.counter < 1700:
                #    c='x'
            # do next base count
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        if self.counter in [1010+852]:
            c = self.base_val
            self.base_val = self.tick_base.tick()
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [0], "1")
        self.base_val = self.tick_base.tick()
        # groups of short, long
        self.tick_layer1 = CyclicalTicker(4, 2, [3], 3)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        # cut in the middle
        self.tick_layer2 = CyclicalTicker(15, 63, [5], 59)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
        for i in range(852):
            self.bit()

# Bit Generator for bit 06
class Bit6_Generator(Generator):
    # 0 1 , skip a bit every 9 or 8 bits, 8 times
    def bit(self):
        c =''
        self.counter += 1
        # return base Val based on sequence
        if not self.tick_base1.tick():
            # skip a beat
            self.tick_base0.tick()
            if self.tick_layer2.tick():
                self.tick_base1 = CyclicalTicker(9, True, [0], False, 0)
            else:
                # after a cycle of 8 9-bit, do a 8-bit pattern
                self.tick_base1 = CyclicalTicker(8, True, [0], False, 0)
                ## debug
                if self.counter == 2048 or self.counter in [2048, 4098]:
                    # skip one
                    self.tick_layer2.tick()
        #
        c = self.tick_base0.tick()
        return c

    def reset(self):
        self.counter = 0
        self.tick_base0 = CyclicalTicker(2, "1", [0], "0")
        # how many normal ticks
        self.tick_base1 = CyclicalTicker(9, True, [0], False, 3)
        # after a cycle of 8 9-bit, do a 8-bit pattern
        self.tick_layer2 = CyclicalTicker(8, True, [0], False, self.init)

# Bit Generator for bit 07
class Bit7_Generator(Generator):
    # (0*4 1*5)*4 + 0*4 + (1*4 0*5)*4 + 1*4
    def bit(self):
        c =''
        self.counter += 1
        if self.tick_layer1.tick(): # wraps
            c = self.base_val
        else:
            # switch base bit value
            self.base_val = self.tick_base.tick()
            c = self.base_val
            if self.tick_layer2.has_wrapped:
                self.tick_layer2 = CyclicalTicker(self.tick_layer3.tick(), 4, [1,3,5,7], 5, 0)
                if self.counter in range(1062,7400,1025): #[1062, 2087, 3112, 4137, ]: #1275
                    self.tick_layer2 = CyclicalTicker(self.tick_layer3.tick(), 4, [1,3,5,7], 5, 0)
            self.tick_layer1 = BinaryCyclicalTicker(self.tick_layer2.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [1], "1")
        self.base_val = self.tick_base.tick()

        # switch base bit on 0 and 5
        self.tick_layer3 = CyclicalTicker(2, 7, [0], 9)
        self.tick_layer2 = CyclicalTicker(self.tick_layer3.tick(), 4, [1,3,5,7], 5, 0)
        self.tick_layer1 = BinaryCyclicalTicker(self.tick_layer2.tick(), self.init)
        
# Bit Generator for bit 08
class Bit8_Generator(Generator):
    # 00 11 00 11 , 4 or 5 times, then produce bit and skip next 2
    def bit(self):
        c =''
        self.counter += 1
        c = self.tick_base.tick()
        if not self.tick_layer1.tick(): # wraps
            # drop 3 bits
            self.tick_base.tick()
            self.tick_base.tick()
            self.tick_base.tick()
            # change rithm
            self.tick_layer1 = BinaryCyclicalTicker(self.tick_layer2.tick())
            if self.counter in [199, 605, 1011, 1428, 1630, 2107, 2655, 3132, 3680, 4157]: #1406, 1415]: #1417
                self.tick_layer1 = BinaryCyclicalTicker(self.tick_layer2.tick())
                self.tick_layer2.tick()
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(4, "0", [0,1], "1", 0)
        self.tick_layer2 = CyclicalTicker(7, 9, [0,1,3,5,7,8], 11, -1)
        self.tick_layer1 = BinaryCyclicalTicker(self.tick_layer2.tick(), self.init)
                
# Bit Generator for bit 09
class Bit9_Generator(Generator):
    # 0101, then some buit skip 
    def bit(self):
        c =''
        self.counter += 1
        if self.counter == 1:
            return "0"    
        c = self.tick_base.tick()
        if not self.tick_layer1.tick(): # wraps
            # drop 1 bits
            self.tick_base.tick()
            if self.counter in [1021, 2046, 3071, 4096, 5121, 6146, 7171]:
                self.tick_layer2 = CyclicalTicker(27, 10, [0,7,14,21,28 ], 11, 20)
            self.tick_layer1 = BinaryCyclicalTicker(self.tick_layer2.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [0], "1", 0)
        self.tick_layer2 = CyclicalTicker(27, 10, [0,7,14,21,28 ], 11, 20)
        self.tick_layer1 = BinaryCyclicalTicker(self.tick_layer2.tick(), self.init)

        
# Bit Generator for bit 10
class Bit10_Generator(Generator):
    # 5*0 5*1 *13 + bit
    def bit(self):
        c =''
        self.counter += 1
        if not self.tick_layer2.tick(): # wraps
            # add a bit from current base val, don't update layer1
            self.tick_layer2 = BinaryCyclicalTicker(self.tick_layer3.tick()+1)
            c = self.base_val
            return c
        #
        if not self.tick_layer1.tick(): # wraps
            # switch bit value
            self.base_val = self.tick_base.tick()
        c = self.base_val
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [1], "1", 0)
        self.base_val = self.tick_base.tick()
        self.tick_layer1 = BinaryCyclicalTicker(5, self.init)
        self.tick_layer3 = CyclicalTicker(15, 14*5, [0,2,4,6,8,10,12,14], 13*5, 1)
        self.tick_layer2 = BinaryCyclicalTicker(self.tick_layer3.tick()+1, self.init)
        
# Bit Generator for bit 11
class Bit11_Generator(Generator):
    # 3,2,3,2 , 15 times (73 bits), then switch bit value and start again
    def bit(self):
        c =''
        self.counter += 1
        #
        if not self.tick_base1.tick(): # wraps
            # switch bit value
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
            self.base_val = self.tick_base.tick()
            if not self.tick_layer2.tick(): # wraps
                self.tick_layer1.reset()
                self.tick_base1.reset()
                if self.counter in [1225, 3412, 5599]: #, 2869]:
                    self.tick_layer3 = CyclicalTicker(17, 13, [0,5,9,13], 15)
                self.tick_layer2 = BinaryCyclicalTicker(self.tick_layer3.tick())
        c = self.base_val
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [1], "1", -1)
        self.base_val = self.tick_base.tick()
        
        self.tick_layer1 = CyclicalTicker(2, 3, [1], 2, 1)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick(), -1)

        self.tick_layer3 = CyclicalTicker(17, 13, [0,5,9,13], 15, 10)
        self.tick_layer2 = BinaryCyclicalTicker(self.tick_layer3.tick(), self.init)
        
# Bit Generator for bit 12
class Bit12_Generator(Generator):
    def __init__(self, layer1, base1, base):
        self.x = layer1
        self.y = base1
        self.z = base
        self.reset()
        
    def bit(self):
        c =''
        self.counter += 1
        c = self.tick_base.tick()
        if not self.tick_base1.tick(): # wraps
            # switch sequence
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
            # skip a bit
            self.tick_base.tick()
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [1], "1", self.z)
        self.tick_layer1 = CyclicalTicker(170, 5, [ 0, 3, 7, 10, 14, 18, 21, 25, 29, 32, 36, 39, 43, 47, 50, 54, 58, 61, 65, 68, 72, 76, 79, 83, 86, 90, 94, 97, 101, 105, 108, 112, 115, 119, 123, 126, 130, 133, 137, 141, 144, 148, 152, 155, 159, 162, 166], 
                                          4, self.x)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick(), self.y)

# Bit Generator for bit 13
class Bit13_Generator(Generator):
    # 7 repetitive sequence of ~803 bits (a,b,a,a)*7 + a
    # b=00 111  00 11  000 11  00 111  00 11 000 11 00 111 = 14 base
    # a=00 111  00 11  000 11  00 111  00 11 000   = 11base
    # 7x(3*11+14) +11  = 340 groups tick_base1
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.reset()
        
    def bit(self):
        c =''
        self.counter += 1
        c = self.base_val
        #
        if not self.tick_base1.tick(): 
            # wraps between 2 and 3
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
            self.base_val = self.tick_base.tick()
            # if 14 or 11, wrap line
            if not self.tick_base2.tick():
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                self.tick_layer1 = CyclicalTicker(3, 2, [1], 3)
                self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
            # if 340 groups, wraps sequence all together. 
            if not self.tick_base3.tick():
                self.tick_layer2 = CyclicalTicker(4, 11, [1], 14) 
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                self.tick_layer1 = CyclicalTicker(3, 2, [1], 3)
                self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [1], "1", 0)
        self.base_val = self.tick_base.tick()
        self.tick_layer1 = CyclicalTicker(3, 2, [1], 3, 0) # nn mmm nn
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick(), 0)
        # how many groups
        self.tick_layer2 = CyclicalTicker(4, 11, [1], 14, self.x)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick(), self.y)
        self.tick_base3 = BinaryCyclicalTicker(340, 41+11)


# Bit Generator for bit 14
class Bit14_Generator(Generator):
    # groups of 6 or 7 bits, then skip a bit
    # repeat 6,7 bits 17 or 19 times
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.reset()
        
    def bit(self):
        c =''
        self.counter += 1
        c = self.tick_base.tick()
        #
        if not self.tick_base1.tick(): 
            # skip a bit
            self.tick_base.tick()
            # if 17 or 19, wrap line
            if not self.tick_base2.tick():
                #print 'base2', self.counter
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                # skip a layer 1 group too and start on 7
                self.tick_layer1.tick()
            # layer1 wraps between 6 and 7
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [1], "1", 0)
        # groups of 6, 7 
        self.tick_layer1 = CyclicalTicker(2, 7, [1], 6, 0)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick(), self.z)
        # Fix, probably longer pattern, bit key 3400,4260 are wrong
        self.tick_layer2 = CyclicalTicker(7, 17, [0,4], 19, self.x)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick(), self.y)
        
# Bit Generator for bit 15
class Bit15_Generator(Generator):
    # groups of 3or4 same bit, 19 groups max (or 15)
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.reset()
        
    def bit(self):
        c =''
        self.counter += 1
        c = self.base_val
        #
        if not self.tick_base1.tick(): 
            # change base
            self.base_val = self.tick_base.tick()
            # if 15 or 19, wrap line
            if not self.tick_base2.tick():
                #print 'base2', self.counter
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                # skip a layer 1 group too and start on 7
                self.tick_layer1.reset()
            # layer1 wraps between 6 and 7
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [1], "1")
        self.base_val = self.tick_base.tick()
        # groups of 3, 4
        self.tick_layer1 = CyclicalTicker(4, 3, [0], 4)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick(), self.z)
        # Fix, probably longer pattern, bit key 3400,4260 are wrong
        self.tick_layer2 = CyclicalTicker(14, 19, [2,5,8,11,13], 15, self.x)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick(), self.y)
        
# Bit Generator for bit 16
class Bit16_Generator(Generator):
    # 00100 11011011 00100 11011011 00100 repetead
    # then a shorter 00100 11011011 00100 every 4 or 5 
    # 2,1,2, 2,1,2,1,2, 2,1,2 ,2,1,2,1,2, 2,1,2
    # 1,2 => [0,2,3,5,7,8,10,11,13,15,16,18]
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.reset()
        
    def bit(self):
        c =''
        self.counter += 1
        c = self.base_val
        #
        if not self.tick_base1.tick(): 
            # change base
            self.base_val = self.tick_base.tick()
            # if 15 or 19, wrap line
            if not self.tick_base2.tick():
                #print 'base2', self.counter
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                # skip a layer 1 group too and start on 7
                self.tick_layer1.reset()
            # layer1 wraps between 6 and 7
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [1], "1")
        self.base_val = self.tick_base.tick()
        # groups of 3, 4
        self.tick_layer1 = CyclicalTicker(19, 1, [0,2,3,5,7,8,10,11,13,15,16,18], 2)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick(), self.z)
        # cut in the middle
        self.tick_layer2 = CyclicalTicker(7, 19*5+11, [0,3,5], 19*4+11, self.x)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick(), self.y)
        for i in range(9):
            self.bit()
        
# Bit Generator for bit 17
class Bit17_Generator(Generator):
    # 3x 001011010-short + (long-0010110100101 001011010 110100101) + {long, long-short-short-long}
    # 2,1,1,2,1,1,1,2,1,1,2,1,1,1, 2,1,1,2,1,1,1, long = 2,1,1,2,1,1,2,1,1,1 
    # 79, 1,2=> [0,3,7,10,14,17,21,24,27, 31,34,38,41, 45,48,51, 55,58,62,65, 69,72,75]        
    def bit(self):
        c =''
        self.counter += 1
        c = self.base_val
        #
        if not self.tick_base1.tick(): 
            # change base
            self.base_val = self.tick_base.tick()
            if not self.tick_base2.tick():
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                self.tick_layer1.reset()
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [0], "1")
        self.base_val = self.tick_base.tick()
        # groups of short, long
        self.tick_layer1 = CyclicalTicker(79, 1, [0,3,7,10,14,17,21,24,27, 31,34,38,41, 45,48,51, 55,58,62,65, 69,72,75], 2)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        # cut in the middle
        self.tick_layer2 = CyclicalTicker(7, 79, [1,2,3,5,6,8,9,11,12], 55)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
        for i in range(234):
            self.bit()

# Bit Generator for bit 18 - not perfect
class Bit18_Generator(Generator):
    # long-short X3 + long
    # long-short X2 + long
    # 00011001100 111001100 11100110011 000110011 00011001100 111001100 11100110011
    # 32 ; 2,3 => [0,5,9,14,18,23,27]
    def bit(self):
        self.counter += 1
        c = self.base_val
        if not self.tick_base1.tick(): 
            self.base_val = self.tick_base.tick()
            if not self.tick_base2.tick():
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                self.tick_layer1.reset()
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [0], "1")
        self.base_val = self.tick_base.tick()
        # groups of short, long
        self.tick_layer1 = CyclicalTicker(32, 2, [0,5,9,14,18,23,27], 3)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        # cut in the middle
        self.tick_layer2 = CyclicalTicker(20, 32, [1,3,5,7,9,12,14,16,18], 23)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
        for i in range(198):
            self.bit()

# Bit Generator for bit 19
class Bit19_Generator(Generator):
    def bit(self):
        self.counter += 1
        c = self.tick_base.tick()
        if not self.tick_base1.tick(): 
            # skip bit
            self.tick_base.tick()
            if not self.tick_base2.tick():
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                self.tick_layer1.reset()
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [0], "1")
        # groups of short, long
        self.tick_layer1 = CyclicalTicker(7, 10, [0], 11)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        # cut in the middle
        self.tick_layer2 = CyclicalTicker(19, 6, [0,10], 7)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
        for i in range(146):
            self.bit()

# Bit Generator for bit 20
class Bit20_Generator(Generator):
    # 6*bit 5*0 5*1 *11 + bit
    def bit(self):
        self.counter += 1
        c = self.base_val
        if not self.tick_base1.tick(): 
            self.base_val = self.tick_base.tick()
            if not self.tick_base2.tick():
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                self.tick_layer1.reset()
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [1], "1")
        self.base_val = self.tick_base.tick()
        # groups of short, long
        self.tick_layer1 = CyclicalTicker(14, 5, [0], 6)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        # cut in the middle.
        self.tick_layer2 = CyclicalTicker(33, 12, [0,5,10,14, 19,24,29], 13, 1)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
        for i in range(14):
            self.bit()

# Bit Generator for bit 21
class Bit21_Generator(Generator):
    # 
    def bit(self):
        self.counter += 1
        c = self.base_val
        if not self.tick_base1.tick(): 
            self.base_val = self.tick_base.tick()
            if not self.tick_base2.tick():
                if self.counter in [1024]:
                    # flip base
                    self.tick_base.tick()
                    self.tick_layer2 = CyclicalTicker(5, 13, [1,3], 11, 2)
                if self.counter in [775,1799,2823,3847]:
                    self.tick_layer2 = CyclicalTicker(5, 13, [1,3], 11)
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                self.tick_layer1.reset()
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [1], "1")
        self.base_val = self.tick_base.tick()
        # groups of short, long
        self.tick_layer1 = CyclicalTicker(2, 3, [1], 2)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        self.tick_layer2 = CyclicalTicker(5, 13, [1,3], 11, 2)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
        for i in range(11):
            self.bit()
            
# Bit Generator for bit 22
class Bit22_Generator(Generator):
    # 
    def bit(self):
        self.counter += 1
        c = self.base_val
        if not self.tick_base1.tick(): 
            self.base_val = self.tick_base.tick()
            if not self.tick_base2.tick():
                if self.counter in range(1024,7500, 1024):
                    self.tick_layer2.reset()
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                self.tick_layer1.reset()
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [0], "1")
        self.base_val = self.tick_base.tick()
        # 00-1-0-1
        self.tick_layer1 = CyclicalTicker(4, 1, [0], 2)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        #4 cycle +1
        self.tick_layer2 = CyclicalTicker(53, 11, [0,4,7, 10,14,17, 20,24,27, 30,34,37, 40,44,47,50], 15, 1)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
        for i in range(10):
            self.bit()
            
# Bit Generator for bit 23
class Bit23_Generator(Generator):
    # 
    def bit(self):
        self.counter += 1
        c = self.base_val
        if not self.tick_base1.tick(): 
            self.base_val = self.tick_base.tick()
            if not self.tick_base2.tick():
                if self.counter in range(1024,7500, 1024):
                    self.tick_layer2.reset()
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                self.tick_layer1.reset()
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [0], "1")
        self.base_val = self.tick_base.tick()
        # 000-11-00
        self.tick_layer1 = CyclicalTicker(3, 2, [0], 3)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        #4 cycle +1
        self.tick_layer2 = CyclicalTicker(5, 20, [0,2,4], 23, 2)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
        for i in range(23):
            self.bit()

# Bit Generator for bit 24
class Bit24_Generator(Generator):
    # 
    def bit(self):
        self.counter += 1
        c = self.base_val
        if not self.tick_base1.tick(): 
            self.base_val = self.tick_base.tick()
            if not self.tick_base2.tick():
                if self.counter in range(1024,7500, 1024):
                    self.tick_layer2.reset()
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                self.tick_layer1.reset()
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [1], "1")
        self.base_val = self.tick_base.tick()
        # 0010101, number of change of bits
        self.tick_layer1 = CyclicalTicker(6, 1, [0], 2)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        #4 cycle +1short
        self.tick_layer2 = CyclicalTicker(5, 23, [0], 17, 3)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
        for i in range(22):
            self.bit()

# Bit Generator for bit 25
class Bit25_Generator(Generator):
    # 
    def bit(self):
        self.counter += 1
        c = self.base_val
        if not self.tick_base1.tick(): 
            self.base_val = self.tick_base.tick()
            if not self.tick_base2.tick():
                if self.counter in range(1024,7500, 1024):
                    self.tick_layer2.reset()
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                self.tick_layer1.reset()
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [0], "1")
        self.base_val = self.tick_base.tick()
        # 3-4-3, number of change of bits
        self.tick_layer1 = CyclicalTicker(3, 3, [0], 4)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        #1 long cycle +1short
        self.tick_layer2 = CyclicalTicker(2, 8, [0], 11)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
        for i in range(4):
            self.bit()

# Bit Generator for bit 26
class Bit26_Generator(Generator):
    # 
    def bit(self):
        self.counter += 1
        c = self.base_val
        if not self.tick_base1.tick(): 
            self.base_val = self.tick_base.tick()
            if not self.tick_base2.tick():
                if self.counter in range(1024,7500, 1024):
                    self.tick_layer2.reset()
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                self.tick_layer1.reset()
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [0], "1")
        self.base_val = self.tick_base.tick()
        # 2-1-2, number of change of bits
        self.tick_layer1 = CyclicalTicker(3, 1, [0,2], 2)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        #1 long cycle +1short
        self.tick_layer2 = CyclicalTicker(2, 19, [10], 11)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
        for i in range(2):
            self.bit()
                        
# Bit Generator for bit 27
class Bit27_Generator(Generator):
    # 
    def bit(self):
        self.counter += 1
        c = self.base_val
        if not self.tick_base1.tick(): 
            self.base_val = self.tick_base.tick()
            if not self.tick_base2.tick():
                if self.counter in range(1024,7500, 1024):
                    self.tick_layer2.reset()
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                self.tick_layer1.reset()
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [1], "1")
        self.base_val = self.tick_base.tick()
        # 1-2-1-1, number of change of bits
        self.tick_layer1 = CyclicalTicker(4, 1, [1], 2)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        #1 long cycle +1short
        self.tick_layer2 = CyclicalTicker(2, 13, [10], 11)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
        for i in range(2):
            self.bit()
                        
# Bit Generator for bit 28
class Bit28_Generator(Generator):
    # 
    def bit(self):
        self.counter += 1
        c = self.base_val
        if not self.tick_base1.tick(): 
            self.base_val = self.tick_base.tick()
            if not self.tick_base2.tick():
                if self.counter in range(1024,7500, 1024):
                    self.tick_layer2.reset()
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                self.tick_layer1.reset()
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [0], "1")
        self.base_val = self.tick_base.tick()
        # 3-2-3, number of change of bits
        self.tick_layer1 = CyclicalTicker(3, 2, [0,2], 3)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        #1 long cycle +1short
        self.tick_layer2 = CyclicalTicker(2, 6, [10], 11)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
        for i in range(3):
            self.bit()
                        
# Bit Generator for bit 29
class Bit29_Generator(Generator):
    # 
    def bit(self):
        self.counter += 1
        c = self.base_val
        if not self.tick_base1.tick(): 
            self.base_val = self.tick_base.tick()
            if not self.tick_base2.tick():
                if self.counter in range(1024,7500, 1024):
                    self.tick_layer2.reset()
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                self.tick_layer1.reset()
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [0], "1")
        self.base_val = self.tick_base.tick()
        # 2-1-1, number of change of bits
        self.tick_layer1 = CyclicalTicker(3, 1, [0], 2)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        #1 long cycle +1short
        self.tick_layer2 = CyclicalTicker(2, 6, [10], 11)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
        for i in range(2):
            self.bit()

# Bit Generator for bit 30
class Bit30_Generator(Generator):
    # 
    def bit(self):
        self.counter += 1
        c = self.base_val
        if not self.tick_base1.tick(): 
            self.base_val = self.tick_base.tick()
            if not self.tick_base2.tick():
                if self.counter in range(1024,7500, 1024):
                    self.tick_layer2.reset()
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                self.tick_layer1.reset()
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [0], "1")
        self.base_val = self.tick_base.tick()
        # 2-2, number of change of bits
        self.tick_layer1 = CyclicalTicker(2, 1, [0,1], 2)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        #1 long cycle +1short
        self.tick_layer2 = CyclicalTicker(2, 6, [10], 11)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
        for i in range(1):
            self.bit()
                        
# Bit Generator for bit 31
class Bit31_Generator(Generator):
    # 
    def bit(self):
        self.counter += 1
        c = self.base_val
        if not self.tick_base1.tick(): 
            self.base_val = self.tick_base.tick()
            if not self.tick_base2.tick():
                if self.counter in range(1024,7500, 1024):
                    self.tick_layer2.reset()
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                self.tick_layer1.reset()
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [0], "1")
        self.base_val = self.tick_base.tick()
        # 1-1, number of change of bits
        self.tick_layer1 = CyclicalTicker(2, 1, [], 2)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        #1 long cycle +1short
        self.tick_layer2 = CyclicalTicker(2, 6, [10], 11)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
        for i in range(3):
            self.bit()

# generators
# we will generate each bit with a bit generator
generators = dict()
#for i in range(32):
#    # fake based on knwo keys
#    generators[i] = KeysBasedBitGenerator(key_manager.get_keys('consensus'), i)

generators[0] = Bit0_Generator(2)
generators[1] = Bit1_Generator(4, 31, 35, [1,16], 36)
generators[2] = Bit2_Generator(2)
generators[3] = Bit3_Generator(2)
generators[4] = Bit4_Generator(2)
generators[5] = Bit5_Generator(-1,-1,-1)
generators[6] = Bit6_Generator(1)
generators[7] = Bit7_Generator(2)
generators[8] = Bit8_Generator(4)        
generators[9] = Bit9_Generator(4)
generators[10] = Bit10_Generator(1)
generators[11] = Bit11_Generator(0)
generators[12] = Bit12_Generator(53, 2, -1)
generators[13] = Bit13_Generator(-1, 5, -1)        
generators[14] = Bit14_Generator(4, 12, 3)
generators[15] = Bit15_Generator(2, 10, 0)        
generators[16] = Bit16_Generator(-1, 18, -1)
generators[17] = Bit17_Generator(-1)
generators[18] = Bit18_Generator(-1)
generators[19] = Bit19_Generator(-1)            
generators[20] = Bit20_Generator(-1)            
generators[21] = Bit21_Generator(-1)
generators[22] = Bit22_Generator(-1)            
generators[23] = Bit23_Generator(-1)
generators[24] = Bit24_Generator(-1)
generators[25] = Bit25_Generator(-1)
generators[26] = Bit26_Generator(-1)
generators[27] = Bit27_Generator(-1)          
generators[28] = Bit28_Generator(-1)
generators[29] = Bit29_Generator(-1)
generators[30] = Bit30_Generator(-1)
generators[31] = Bit31_Generator(-1)



# generate a xor key list with the above bit generators
_gen = generators.items()
_gen.sort()
# reset the state
for bit_generator in [x[1] for x in _gen]:
    bit_generator.reset()
gen_keys = []
for i in range(len(key_manager.get_keys('consensus'))):
    b_k = ''.join([bit_generator.bit() for bit_generator in [x[1] for x in _gen]])
    h_key = "%0.8x" % int(b_k, 2)
    gen_keys.append(h_key)

# save them in our manager
key_manager.add_keys('generated', gen_keys)
generated_keys_view = key_manager.get_bitview('generated')
# save generated keys to file
key_manager.save_keys_to_file('generated', 'bitgen-attack')

# Quick fixes
# the bit generators are not perfect.
corrected_keys = key_manager.get_keys('generated')
key_manager.add_keys('corrected', corrected_keys)

# trying something, no sure
# keys 0 -1024 seems to have flipped [5,21] bits
if True:
    for i in range(1025):
        new_key = flip_bits(corrected_keys[i], [5,21])
        key_manager.change_key('corrected', i, new_key, verbose=False)
    
else:
    key_manager.change_key('corrected', 0, '071969ed') # kt @0x00000830
    key_manager.change_key('corrected', 1, 'b72047a5') # kt @0x00000a30
    key_manager.change_key('corrected', 2, '7559f938') # kt @0x00000c40


    ## a bit more sure
    key_manager.change_key('corrected', 8, 'ca94028a') # kt @0x00001994 ?????
    key_manager.change_key('corrected', 9, 'a8edd43d') # kt @0x00001a90

    ## 0x00 ?
    key_manager.change_key('corrected', 33, '7e567a05') # kt @0x00004a60
    key_manager.change_key('corrected', 34, '3c902b98') # kt @0x00004c60
    key_manager.change_key('corrected', 35, 'fac9dd2b') # kt @0x00004e60
    key_manager.change_key('corrected', 36, 'b9038ebe') # kt @0x00005060
    key_manager.change_key('corrected', 37, '773d4051') # kt @0x00005270

    ## totally sure
    key_manager.change_key('corrected', 17, '9abb60d5') # kt @0x00002bfc
    key_manager.change_key('corrected', 18, '58f51268') # revcase @0x00002c00
    key_manager.change_key('corrected', 19, '172ec3fb' ) # revcase 0x2d00
    key_manager.change_key('corrected', 20, 'd568758e' ) # revcase 0x3000
    key_manager.change_key('corrected', 21, '93a22721') # kt @0x00003200
    key_manager.change_key('corrected', 27, '08fc5093') # kt @0x00003f3c
    key_manager.change_key('corrected', 28, 'c7360226') # kt @0x00004000
    key_manager.change_key('corrected', 30, '43a9654c') # diff @0x00004400
    key_manager.change_key('corrected', 31, '01e316df') # kt @0x00004600
    key_manager.change_key('corrected', 32, 'c01cc872') # kt @0x00004800
    key_manager.change_key('corrected', 39, 'f3b0a377') # kt @0x0000576c
    key_manager.change_key('corrected', 40, 'b1ea550a') # kt @0x00005800
    key_manager.change_key('corrected', 42, '2e5db830') # kt @0x00005c64
    key_manager.change_key('corrected', 43, 'ec9769c3') # flipbits @0x00005e00
    key_manager.change_key('corrected', 47, 'e57e300f') # flipbits @0x00006600
    key_manager.change_key('corrected', 49, '61f19335') # flipbits @0x00006a00
    key_manager.change_key('corrected', 51, 'de64f65b') # flipbits @0x00006e00
    key_manager.change_key('corrected', 52, '9c9ea7ee') # flipbits @0x00007000

    key_manager.change_key('corrected', 1021, 'a4ffcd59') # kt @0x00080220
    
    
key_manager.change_key('corrected', 1226, 'f9330010') # kt @0x00099d60

key_manager.change_key('corrected', 4702, 'e092200c') # kt @0x0024c4b0
key_manager.change_key('corrected', 5355, '19bc1403') # kt @0x0029de20
key_manager.change_key('corrected', 5383, 'e80b8017') # kt @0x002a1690

key_manager.change_key('corrected', 5405, '4100c2b9') # kt @0x002a4200
key_manager.change_key('corrected', 5975, 'cd762407') # kt @0x002eb638
key_manager.change_key('corrected', 6208, 'eff8c2d2') # kt @0x00308914
key_manager.change_key('corrected', 7262, '2181de0c') # kt @0x0038c414
key_manager.change_key('corrected', 7280, '81905a62') # kt @0x0038e800
key_manager.change_key('corrected', 7282, 'fe03bd88') # kt @0x0038ec08

# FIX 00391400
key_manager.change_key('corrected', 7302, 'da859d04') # kt @0x00391400
key_manager.change_key('corrected', 7304, '56f9002a') # kt @0x00391800
key_manager.change_key('corrected', 7309, '0e197809') # kt @0x00392200
key_manager.change_key('corrected', 7313, '07003e55') # kt @0x00392a00
key_manager.change_key('corrected', 7317, 'ffe704a1') # kt @0x00393200
key_manager.change_key('corrected', 7322, 'b7077c80') # kt @0x00393c00

# TODO
# fix 0024c400 - uppercase byte 3. byte 21 ?
# byte 18 is bad, 21 too

corrected_keys_view = key_manager.get_bitview('corrected')
corrected_keys = key_manager.get_keys('corrected')

# save generated keys to file
key_manager.save_keys_to_file('corrected', 'bitgen-attack-corrected')

# decode all firmware with this new key list
unxor_suffix = '.unxor.3'
for fname, firm in firmwares_parts.items():
    print fname, "...",
    firm.apply_xor(key_manager.get_keys('corrected'))
    unxor_fname = os.path.sep.join(['firmwares', 'unxor', fname])
    firm.write_to_file(unxor_fname+unxor_suffix)
    print 'wrote to', unxor_fname+unxor_suffix

ap1_v130p1 = firmwares_parts['ap1_v130.frm.part1']
ap1_v110p1 = firmwares_parts['ap1_v110.frm.part1']
ap1_v130p2 = firmwares_parts['ap1_v130.frm.part2']
gy1_v121p1 = firmwares_parts['gy1_v121.frm.part1']
gy1_v121p2 = firmwares_parts['gy1_v121.frm.part2']

                       
# 0x17b10
addr = 0x7000
if True:
    firm_part = ap1_v130p2
    # firm_part = ap1_v130p2
    # firm_part = gy1_v121p2
    # firm_part = gy1_v121p1
    _section = firm_part[addr]

    #print hexdump(firm_part[addr-0x400].section_out, addr-0x400)
    #print hexdump(firm_part[addr-0x200].section_out, addr-0x200)
    print hexdump(firm_part[addr].section_out, addr)
    print try_flip_bits(firm_part, addr, [5,21])
    print try_known_text_attack(firm_part, addr+0x64, 'Asser')
    print hexdump(firm_part[addr+0x200].section_out, addr+0x200)
    print hexdump(firm_part[addr+0x400].section_out, addr+0x400)

else:
    a = gy1_v121p1
    # b = gy1_v121p2
    b = ap1_v130p2
    # b = ap1_v130p2
    print corrected_keys[get_i_from_address(addr)], gen_keys[get_i_from_address(addr)]
    _section_a = a.get_section_cleartext(addr, corrected_keys)
    _section_b = b.get_section_cleartext(addr, corrected_keys)
    _section_a_gen = a.get_section_cleartext(addr, gen_keys)
    _section_b_gen = b.get_section_cleartext(addr, gen_keys)
    print hexdump(_section_a, addr)
    print hexdump(_section_b, addr)
    #print hexdump_diff(_section_a, _section_b)
    #print hexdump_diff(_section_a_gen, _section_b_gen)
    print hexdump_4(_section_a, _section_b, _section_a_gen, _section_b_gen)
    
#print comp(gy1_v121p1[addr], gy1_v121p2[addr])
#print comp(gy1_v121p1[addr], ap1_v130p1[addr])
#print comp(ap1_v110p1, ap1_v130p1[addr])
#print try_known_text_attack(_section_b, addr+0x90, 'a01c')

# TODO Move to sandbox
## bit xx comparison between known good keys and bit generation


if False:
    test = slice(350, 800)
    for x in range(0,33):
        for y in range(0,33):
            generators[bitnum] = Bit12_Generator(x, y)            
            if show_bit_h(test, bitnum) == gen_bit_h(test, bitnum):
                print "!!!! Found ", (x, y)

# bitviews
# kt_attack_keys from consensus keys
# generated_keys_view from generated keys

# show_bit_vhexdump_diff(1, 36, small)
#show_bit_vhexdump_gen(1, 36, small)
#show_bit_vhexdump(bitnum, 36, all_keys)

#import difflib
#s = difflib.SequenceMatcher(None, show_bit_h(small, 1), gen_bit_h(small, 1))
#for block in s.get_matching_blocks():
#    print "a[%d] and b[%d] match for %d elements" % block
# compare the second layer cycle.


# Bit Generator for bit 05 - previous version is mostly good enough
class xBit5_Generator(Generator):
    def bit(self):
        self.counter += 1
        c = self.base_val
        if not self.tick_base1.tick(): 
            self.base_val = self.tick_base.tick()
            if not self.tick_base2.tick():
                if self.counter >950 and self.counter < 2000:
                    print self.counter
                if self.counter in range(991,7500, 1024):
                    self.tick_layer2.reset()
                    print 'reset', self.counter
                self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick())
                self.tick_layer1.reset()
            self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        if self.counter in [1012]: #, 1862]:
            print '** ', self.counter
            c = self.base_val
            c = 'x'
            self.base_val = self.tick_base.tick()
            #self.tick_layer2.reset()
        return c

    def reset(self):
        self.counter = 0
        self.tick_base = CyclicalTicker(2, "0", [1], "1")
        self.base_val = self.tick_base.tick()
        # 1-1, number of change of bits
        self.tick_layer1 = CyclicalTicker(4, 2, [0], 3)
        self.tick_base1 = BinaryCyclicalTicker(self.tick_layer1.tick())
        #1 long cycle +1short
        self.tick_layer2 = CyclicalTicker(8, 63, [1], 59, 6)
        self.tick_base2 = BinaryCyclicalTicker(self.tick_layer2.tick(), 12)
        for i in range(4):
            self.bit()

#generators[bitnum] = Bit5_Generator(-1)
#generators[bitnum].reset()

int_i = 0

def s_split(x):
    x = x.replace("0001100111","0001100,111").replace("1110011000","1110011,000")
    x = x.replace("1100111","1100 111").replace("0011000","0011 000")
    #x = x.replace("1111100001","111110000 1")
    #x = x.replace("0010100", "0010 100").replace("1101011","1101 011")
    #x = x.replace(" 000000 ", " ,000000 ")
    x = x.split(",")
    i = j = 0
    lines = []
    for l in x:
        i2 = l.count(" ")
        j += len(l)-i2
        #lines.append("(%0.3d/%d) %s" % (i, j, l))
        lines.append("(%d) %s" % (j, l))
        #print "%d,"%i,
        i+=i2+1
    x = '\n'.join(lines)
    #print
    return x

#print s_split(show_bit_h(all_keys, bitnum)[int_i:7000])
#print '--'*12
#print s_split(gen_bit_h(all_keys, bitnum)[int_i:7000])

#show_bit_vhexdump_diff(bitnum, 36, all_keys)

# bitviews
# kt_attack_keys from consensus keys
# generated_keys_view from generated keys

bitnum = 21

# Bit quality
# Pretty good )no obvious errors): 0,2,3
# medium (less than 10 obvious errors): 1
# bad (more than 10 obvious errors): 

#show_bit_vhexdump_diff(kt_attack_keys, generated_keys_view, bitnum, 36, all_keys)
show_bit_vhexdump_diff(generated_keys_view, corrected_keys_view, bitnum, 36, all_keys)


#generators[1].reset()
# in short the bit generation is perfect for 1000 bits
if True:
    import difflib
    _slice = slice(300,800)
    _slice = slice(650,1000)
    print _slice
    for bitnum in range(0,32):
        a = kt_attack_keys.show_bit_h(_slice, bitnum)
        b = generated_keys_view.show_bit_h(_slice, bitnum)
        s = difflib.SequenceMatcher(None, a, b)
        #print a[400:480]
        #print b[400:480]
        print "bit %0.2d match: %0.8f"% (bitnum, s.ratio())

#1562 keys updated from 0x9b0a0 (1236) to 0x15e2a0 (2797) in ap1_v130.frm.part1
#1551 keys updated from 0x17c0b0 (3036) to 0x23ddb0 (4586) in gy1_v121.frm.part1

