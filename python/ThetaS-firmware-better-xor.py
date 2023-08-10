get_ipython().magic('run ThetaS.ipynb')
# the firmware
data = open('ap1_v131.frm','rb').read()

# parse all sections
sections = Sections(data, 0x800, 0x200)

sections.reset()
keys_1 = sections.search_xor_keys(find_xor_key)

def find_xor_key_2(_section):
    # try to find key by finding 0x00000000
    words = map(''.join, zip(*[iter(_section)]*4))
    # add best guess for key if value was 0xffffffff
    # look at unique keys
    unique_key = set(words)
    # for each key, if we found
    for word in list(words):
        if word in unique_key:
            continue
        if xor(word, '\xff\xff\xff\xff') in unique_key:
            key = xor(word, '\xff\xff\xff\xff')
            print 'found corresponding 0xf for key', binascii.hexlify(key)
            # reinforce key
            words.append(key)
    # now look at stats
    words = map(binascii.hexlify, words)
    freqs = Counter(words)
    key = freqs.most_common(1)[0][0]
    # unhex_key = binascii.unhexlify(key)
    return key, freqs

sections2 = Sections(data, 0x800, 0x200)
keys_2 = sections2.search_xor_keys(find_xor_key_2)

# search for differences
if keys_1 != keys_2:
    print "different keys have been calculated"
else:
    print "no improvements in find_xor_key_2"

def show_keys_diff(ks1, ks2):
    for a, b in enumerate(zip(ks1, ks2)):
        if a != b:
            print 'key diff at %d: %s-%s' % (i, a, b)
            print a
            print hexdump(xor(sections.sections[i], binascii.unhexlify(a)))
            print b
            print hexdump(xor(sections.sections[i], binascii.unhexlify(a)))
    

interestings_sections = [0x2c00, 0x3000, 0x247ab0]

look_at = 0x247ab0
section = sections[look_at]
print section
start = hex(section.start)
print start

key = section.xor_key
print 'key:', section.hex_key
freqs = section.xor_freq
print hexdump(section.section_out)
for key, freq in freqs.items():
    out = xor(section.data[0xa0:0x100], key)
    if 'BEGIN RSA' in out:
        print binascii.hexlify(key)
        print hexdump(out)
        break

#print 'key:', sections2.section_keys[start]
#print hexdump(sections2.section_outs[start], start)

valid_key = '623b75cd'
print section.xor_freq.most_common(10)


best, res = find_xor_key_3(section.data)
print "best meta-choice for key:", best
out = xor(section.data[0xa0:0x100], binascii.unhexlify(best))
print hexdump(out)

def get_best_bytes_pos(_data):
    # let's first identify section where the key is probably already found
    #from operator import itemgetter
    #keys = map(itemgetter(0), section.xor_freq.items())
    # frequency analysis per byte position
    bytes_1 = []
    bytes_2 = []
    bytes_3 = []
    bytes_4 = []
    for i in range(0, len(_data), 4):
        bytes_1.append(_data[i])
        bytes_2.append(_data[i+1])
        bytes_3.append(_data[i+2])
        bytes_4.append(_data[i+3])
    c_1 = Counter(bytes_1)
    c_2 = Counter(bytes_2)
    c_3 = Counter(bytes_3)
    c_4 = Counter(bytes_4)
    key = '%c%c%c%c' % ( c_1.most_common(1)[0][0], c_2.most_common(1)[0][0], c_3.most_common(1)[0][0], c_4.most_common(1)[0][0])
    print binascii.hexlify(key)
    res = [(binascii.hexlify(c), cnt) for c, cnt in c_1.items()]
    res.sort(key=lambda x: x[1], reverse=True)
    print res
    return key

res = get_best_bytes_pos(section.data)
if res != valid_key:
    print "FAIL"
    

section_a = sections[0x24D200]
section_b = sections[0x24D400]
print hexdump(section_a.section_out[-128:], 0x24D380)
print hexdump(section_b.section_out[:128], section_b.start)

# section_b.reset() 

# we expect 'E(60).checked' between the two sections
try_known_text_attack(sections, section_b.start, ').ch')

section_a = sections[0x24BE00]
section_b = sections[0x24C000]
print hexdump(section_a.section_out[-128:], 0x24BF80)
print hexdump(section_b.section_out[:128], section_b.start)

try_known_text_attack(sections, 0x24c000, 'A PRIVATE')

zero = '\x00\x00\x00\x00'
tmp = zero
for i in range(0, len(section_a.data), 4):
    word = section_a.data[i:i+4]
    tmp = xor(tmp, word)

print binascii.hexlify(tmp), 'not a xor on previous bytes'

# first section at 0x800 is coded with '071969ed'
binascii.hexlify(xor(binascii.unhexlify('ffffffff'), binascii.unhexlify('071969ed')))
'f8e69612'

print data.count('UNITY'), data.index('UNITY'), hex(data.index('UNITY',1))
unity_offset1 = 0x0
unity_offset2 = data.index('UNITY',1)
print hexdump(data[0x2a5c00:0x2a5c00+0x40])
offset1 = 0x800
offset2 = 0x2a6400
print hex(unity_offset2+0x800)
print sections[offset1].hex_key, sections[offset2].hex_key
print sections[offset1 + 0x4900].hex_key, sections[offset2 + 0x4900].hex_key

with open('unity.part1', 'wb') as fout:
    fout.write(data[:unity_offset2])

with open('unity.part2', 'wb') as fout:
    fout.write(data[unity_offset2:])

# save previous keys.
head = True
keys_1 = None
keys_2 = None
key_list = []
key_list.append(sections.sections[0].hex_key)
for i, _s in enumerate(sections.sections[1:]):
    if _s.hex_key == '071969ed':
        print 'rolling at i == %d, start = 0x%08x' % (i, _s.start)
        keys_1 = key_list
        key_list = []
    key_list.append(_s.hex_key)
print _s, hex(_s.start), _s.hex_key
keys_2 = key_list
print "keys_1 len:", len(keys_1)
print "keys_2 len:", len(keys_2)

# save keys to file
with open('unity.xor.keys.part1', 'w') as fout:
    for k in keys_1:
        fout.write("%s\n" % k)

with open('unity.xor.keys.part2', 'w') as fout:
    for k in keys_2:
        fout.write("%s\n" % k)        

# we now need to produce the list of xor keys, and apply to both Sections
# load unity.xor.keys.part1 as our key list and make some adjustments.
keys = keys_1
unity1 = Sections(open('unity.part1', 'rb').read(), 0x800, 0x200)
unity2 = Sections(open('unity.part2', 'rb').read(), 0x800, 0x200) # offset is 0x2a6400
unity1.apply_xor(keys_1)
unity2.apply_xor(keys_2)

# to file for manual check
unity1.write_to_file('unity.part1.unxor')
unity2.write_to_file('unity.part2.unxor')

# add ignore here
ignore = [3, 4]
for i, (x1, x2) in enumerate(zip(keys, keys_2)):
    if x1 == x2:
        continue
    if i in ignore:
        continue
    print '@ ', hex(0x800 + i*0x200)
    comp(unity1.sections[i], unity2.sections[i])
    break

# try other keys
# key = 'd0b25a10'
# print hexdump(unity1.sections[i].show( binascii.unhexlify( key )))
# print hexdump(unity2.sections[i].show( binascii.unhexlify( key )))



interestings_sections = [0x2c00, 0x3000, 0x247ab0, 0x24C400, 0x24C800]
misxor_sections = [0x24A400, 0x24B800, 0x24C000, 0x24C600]

#for section in sections.sections[2000:3000]:
for addr in misxor_sections[2:]:
    section = sections[addr]
    best1 = section.xor_freq.most_common(1)[0][0]
    ratio = 100*section.xor_freq.most_common(1)[0][1]/128
    print hex(section.start), "best at %d%%, nb others keys: %d" % (ratio, len(freqs))
    best3, freq3 = find_xor_key_3(section.data)
    #if best1 != best3:
    #    print 'Proposed is Different', best1, best3
    #    print hexdump(xor(section.data, binascii.unhexlify(best1)))
    #    print hexdump(xor(section.data, binascii.unhexlify(best3)))
    for k in freq3.elements():
        print hexdump(xor(section.data[:128], binascii.unhexlify(k)))

