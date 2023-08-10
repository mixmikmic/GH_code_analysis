get_ipython().system('binwalk ap1_v131.frm')

from IPython.display import Image
Image(filename='entropy.png')
# entropy.png was generated from binwalk

data = open('ap1_v131.frm','rb').read()
import binascii
print data[:0xff]
print '-'*80
print binascii.hexlify(data[:0xff])

def hexdump(src, offset=0, length=16):
    FILTER = ''.join([(len(repr(chr(x))) == 3) and chr(x) or '.' for x in range(256)])
    lines = []
    for c in xrange(0, len(src), length):
        chars = src[c:c+length]
        hex = ' '.join(["%02x" % ord(x) for x in chars])
        printable = ''.join(["%s" % ((ord(x) <= 127 and FILTER[ord(x)]) or '.') for x in chars])
        lines.append("%04x  %-*s  %s\n" % (c+offset, length*3, hex, printable))
    return ''.join(lines)

from itertools import cycle
def xor(data, key):
    return ''.join([chr(ord(c1) ^ ord(c2)) for c1,c2 in zip(data, cycle(key))])
print hexdump(xor('\x00'*12,'\x11\x22\x33\x44'))

print hexdump(data[:0xff], 0)

section1 = data[0x800:0xa00]
print hexdump(section1, 0x800)

section1_xor = xor(section1, binascii.unhexlify('071969ed'))
print hexdump(section1_xor, 0x800)

print hex(0xa00+0x200)
section2 = data[0xa00:0xc00]
print hexdump(section2, 0xa00)

section2_xor = xor(section2, binascii.unhexlify('b72047a5'))
print hexdump(section2_xor, 0xa00)

section_keys = ['071969ed', 'b72047a5']
start = 0xc00
length = 0x200
for i in [0,1,2,3]:
    print hexdump(data[start+i*length:start+(i+1)*length], start+i*length)

from collections import Counter
section3 = data[0xc00:0xe00]
# collect all 0x00000000 to find keys
words = map(''.join, zip(*[iter( binascii.hexlify(section3) )]*8))
freqs = Counter(words)
print freqs
# xor 0xffffffff and add these keys
#for w in list(words):
#    words.append(binascii.hexlify(xor(binascii.unhexlify(w), binascii.unhexlify('ffffffff'))))
#

def find_xor_key(_section):
    # collect stats on null values
    words = map(''.join, zip(*[iter( binascii.hexlify(_section) )]*8))
    # xor 0xffffffff and add these keys
    #for w in list(words):
    #    words.append(binascii.hexlify(xor(binascii.unhexlify(w), binascii.unhexlify('ffffffff'))))
    # now look at stats
    freqs = Counter(words)
    return freqs.most_common(1)[0][0], freqs
# Counter.most_common() is the winner

key, _ = find_xor_key(section1)
print 'section1', key
key, _ = find_xor_key(section2)
print 'section2', key

start = 0xc00
length = 0x200
section_freqs = dict()
section_keys = dict()
sections = []
data_out = data[:0x800]
#for start in range(0x800, 0x1000, length):
for start in range(0x800, len(data), length):
    section = data[start:start+length]
    key, stats = find_xor_key(section)
    section_keys[hex(start)] = key
    section_freqs[hex(start)] = stats
    section_xor = xor(section, binascii.unhexlify(key))
    sections.append(section_xor)
    data_out += section_xor
    #print hexdump(section_xor, start)

# write to file
with open('firm.xor.1','wb') as fout:
    fout.write(data_out)

Image(filename='entropy.png')
# entropy.png was generated from binwalk

Image(filename='entropy_after_section_xor.png')

get_ipython().system('binwalk firm.xor.1')

'CERTIFICATE' in data_out

nb_sections = float(len(data)-0x800)/0x200
print "There are", nb_sections, "sections"
nb_distinct = len(Counter(section_keys.values()).most_common())
print "and", nb_distinct, "distinct keys"

