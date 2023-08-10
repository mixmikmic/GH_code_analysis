import crcmod
import struct
import pandas as pd
get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Flip a bit in a messsage
def flip_bit(msg, idx):
    bit_offset = idx % 8
    byte_offset = idx / 8
    mask = 0x1 << (7-bit_offset)
    return msg[:byte_offset] + chr(ord(msg[byte_offset]) ^ mask) + msg[(byte_offset+1):]

def xor_strings(s1,s2):
    xored_bytes = [ord(a) ^ ord(b) for a,b in zip(s1,s2)]
    return str(bytearray(xored_bytes))

def bin_str(s, sep=""):
    return sep.join(format(ord(c), '08b') for c in s)

def bin_short(short_val):
    return bin_str(struct.pack('>H', short_val))
    

# 1f014829080a1d1802ad2800002bdfff83c5 xor 
# 1f014829100a1d1802ad2800002bdfff014b = 
# 00000000180000000000000000000000828e (B1 & C1)

# 1f014829080a1d1802ad2800002bdfff83c5 xor
# 1f0148291c0a1d1802ad2800002bdfff020a = 
# 0000000014000000000000000000000081cf (B2 & C2)

# 1f02d5af180a1d1801d9180000102fff020a xor
# 1f02d5af140a1d1801d9180000102fff014b = 
# 000000000c00000000000000000000000341

# C2 = 0341
# B1 xor B2 = 000000000c0000000000000000000000 0341


poly = 0x104c1
crc = crcmod.mkCrcFun(poly, initCrc=0, xorOut=0x0, rev=False)

# Now consider two CRC values obtained from two 1-bit messages, 
# where the 1 bits are in adjacent positions. The resulting CRCs 
# will differ by just one shift-xor cycle. To be precise, if
# C1 corresponds to the message with a 1 in position i, and
# C2 corresponds to the message with a 1 in position i+1, then 
# C1 is derived from applying one shift-xor cycle to C2. 
# (If this seems backwards, it's because the further the 1 
# bit is from the end of the message, the more shift-xor cycles
# get applied to the CRC.)

# The unshift_xor() function tries to reverse a shift-xor cycle

def unshift_xor(a,b):
    return ((b << 1) ^ a) & 0xffff

def view_diff(msg, bit_to_flip):
    m1 = msg.decode('hex')
    m2 = flip_bit(m1,bit_to_flip)

    diff = xor_strings(m1,m2)

    print "m1   %s, crc = %s" % (bin_str(m1), bin_short(crc(m1)))
    print "m2   %s, crc = %s" % (bin_str(m2), bin_short(crc(m2)))
    print "diff %s, crc = %s" % (bin_str(diff), bin_short(crc(diff)))
    crc_diff = crc(m1) ^ crc(m2)
    print "                                xored crcs = %s" % bin_short(crc_diff)
    return crc_diff

msg = "deadbeef"
crc_diff4 = view_diff(msg, 4)
print "=" * 80
crc_diff5 = view_diff(msg, 5)
print "=" * 80
crc_diff6 = view_diff(msg, 6)

print "unshift_xor at 4 = %s" % bin_short(unshift_xor(crc_diff4, crc_diff5) & 0xffff)
print "unshift_xor at 5 = %s" % bin_short(unshift_xor(crc_diff5, crc_diff6) & 0xffff)
print "original poly    = %s" % bin_short(poly & 0xffff)

# The entry at crc_diff_dict[10][30] means we observed two 10 bytes messages that differed
# only by a single bit, and the value at that entry is the xor of their crcs. 
crc_diff_dict = {
    # Messages of length 10
    10: {
        30: 0b1000001010011011,
        31: 0b1000000101010001,
        34: 0b0000001110010010,
        35: 0b1000001100101000,
        36: 0b0000001100001110,
        37: 0b1000000100001011,
    },
    
    # Messages of length 3
    3: {
        30:  0b1000000100011111,
        31:  0b0000001000111100,
        34:  0b1000001011011100,
        35:  0b0000000111010111,
        36:  0b1000001101011001,
        37:  0b1000000000011000,
        38:  0b1000001010001110,
        116: 0b0000000000101010,
        117: 0b0000000011011010,
    }
}

def crc_for(l,n):
    crc = crc_diff_dict[l][n]
    #crc = single_bit_crcs10[n]
    
    # collapse 5 bit 'hole'
    #crc = (crc & 0b1111111111) + ((crc >> 5) & 0b10000000000)
        
    # Drop separate high bit
    #crc = crc & 0b1111111111
    
    # Just look at lowest 8 bits
    #crc = crc & 0b11111111
    
    return crc

def diff_at(l,n):
    return unshift_xor(crc_for(l,n), crc_for(l,n+1))

def show_diff_at(l,n):
    print "crc(%d) = %s" % (n, bin_short(crc_for(l,n)))
    print "crc(%d) = %s" % (n+1, bin_short(crc_for(l,n+1)))
    d = diff_at(l,n)
    print "unshift_xor at %d = %s" % (n, bin_short(d))
    print "=" * 80

show_diff_at(3,30)
show_diff_at(3,34)
show_diff_at(3,35)
show_diff_at(3,36)
show_diff_at(3,37)
show_diff_at(3,116)
print "*" * 80
show_diff_at(10,30)
show_diff_at(10,34)
show_diff_at(10,35)
show_diff_at(10,36)

#df = pd.DataFrame(map(lambda x: [x, diff_at(x)], [30, 34, 35, 36]), columns=['idx', 'crc'])
#df.plot.scatter(x='idx', y='crc', marker='.')

