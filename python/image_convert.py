# convert a .png image file to a .bmp image file using PIL
from PIL import Image

file_in = "./data/download.png"
img = Image.open(file_in)

b, g, r = img.split()
file_out = "./data/download_red.bmp"
r.save(file_out)
file_out = "./data/download_green.bmp"
g.save(file_out)
file_out = "./data/download_blue.bmp"
b.save(file_out)

img = Image.open(file_in)
new_img = img.resize((128,128))
new_img.save('./data/download-small.bmp')

from PIL import Image
img = Image.open('data/board_orig.jpg')
w_new = 160
h_new = 128
new_img = img.resize((w_new,h_new),Image.ANTIALIAS)
new_img.save('data/board_small.jpg','JPEG')
img.close()

from PIL import Image
img = Image.open('data/logo.png')
w,h = img.size
w_new = 100
h_new = int(160*h/w)
new_img = img.resize((w_new,h_new),Image.ANTIALIAS)
new_img.save('data/logo_small.png','PNG')
img.close()

help(Image.frombytes)

from numpy import array
from io import BytesIO
from struct import unpack

logo_path = './data/home.jpg'
new_name = './data/home_16.bmp'
with Image.open(logo_path) as img:
    arr = array(img)
    w,h = img.size

print(arr[0][0])
w_new = int(w/2)
h_new = int(h/2)
byte_list = [0]*w_new*h_new
for i in range(h_new):
    for j in range(w_new):
        red,green,blue = arr[i][j]
        temp = ((red & 0xF8) << 8)|((green & 0xFC) << 3)|((blue & 0xF8) >> 3)
        byte_list[i*w_new+j] = temp
print(byte_list[0])
#image = Image.open(BytesIO(bytearray(byte_list)))
#image.save(new_name)
print(len(arr),len(arr[0]),len(arr[0][0]))

arr[297][689]

import numpy as np
a = np.array([1, 2, 3])
np.bitwise_or(np.left_shift(a,2),np.left_shift(a,1))

import cffi
ffi=cffi.FFI()
ffi.cdef("""
uint16_t a[10];
""")
ffi.cdef("""
void *cma_alloc(uint32_t len, uint32_t cacheable);
uint32_t cma_get_phy_addr(void *buf);
""")
lib=ffi.dlopen("/usr/lib/libsds_lib.so")
buf = ffi.cast("uint8_t *",lib.cma_alloc(2*10,0))
buf1= ffi.buffer(buf,2*10)

buf1[0] = b'a'
print(buf1[0])

b = bytearray([100,156])
b[0] = 100
b[1] = 156

try:
    print(a)
finally:
    print("Final")

from pynq import MMIO

a = MMIO(0x40000000, 16)
a.write(0,32)
print(a.read(0))
print(a.read(4))

a = ['BLACK', 'RED']
a.index('BLACK')

b = 'black'
assert b.upper() in a

c = [255,240,255]

r,g,b = c

print(r,g,b)



