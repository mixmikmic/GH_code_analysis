from PIL import Image
image_file = "mini_challenge1.jpg"
img = Image.open(image_file)
img

width = 251
height = 600
st_x = 510
st_y = 300
croped = img.crop((st_x, st_y, st_x+width, st_y+height))
croped

bits = []
for i in range(0, croped.size[0], 1):
    bits.append(0 if croped.getpixel((i,0))[0] < 50 else 1)
string = "".join([str(bit) for bit in bits])
string

import struct
bytestring = bytes()
for i in range(0, len(string), 8):
    bytestring += struct.pack('B', int(string[i:i+8], 2))
print(bytestring)
print([int(b) for b in bytestring])
len(string)

index = 0
count = 0
array = []
while index < len(string) - 1:
    if string[index] == '0':
        count+=1
        index+=1
    else:
        array.append(count)
        count = 0
        while string[index] == '1':
            index += 1
array.append(count+1)
len(array)
array = [i-1 for i in array]
array

conv = []
for i in range(0, len(array), 2):
    conv.append(array[i]<<4 | array[i+1])
[hex(i) for i in conv]

def checksol(array, shift):
    bstring = b''
    for i in array:
        num = 0x40 + i + shift
        if num > 0x40 + 26:
            num -= 26
        bstring += struct.pack('B', num)
    return bstring.decode('ascii')

checksol(array, 0)

image_file = "baby_help.JPG"
img2 = Image.open(image_file)
img2

chunksize = 120
array_im = []
for i in range(10):
    row = []
    for j in range(10):
        row.append(img2.crop((j*120, i*120, j*120+120, i*120+120)))
    array_im.append(row)
    
array_im[2][3]

result = Image.new('RGB', (1200, 1200))
for i in range(10):
    for j in range(10):
        result.paste(im=array_im[j][i], box=(j*120, i*120))
result



