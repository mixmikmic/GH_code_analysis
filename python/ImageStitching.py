# Import modules
from math import floor, ceil
from PIL import Image, ImageDraw
from IPython import display 

# directory
try:
    directory
except NameError:
    directory = "F:\\PA_UC\\"
    print("Directory not specified, set to "+directory)

# stub
try:
    stub
except NameError:
    stub = 1
    print("Stub not specified, set to "+str(stub))
    
# crop size (in pixels)
try:
    CROP_SIZE
except NameError:
    CROP_SIZE = 24
    print("Crop_size not specified, set to "+str(CROP_SIZE)+" pixels")
    
# data
try:
    data
except NameError:
    print("No data available, running ImportData:")
    get_ipython().magic('run ./ImportData.ipynb')
    print("-----")

# Method 1: Square
#NUM_Y = floor(len(data)**0.5)
#NUM_X = ceil(len(data)/NUM_Y)

# Method 2: Fixed width
NUM_X = int(floor(980/CROP_SIZE))*2
NUM_Y = int(ceil(len(data)/NUM_X))

imNew = Image.new('L', (CROP_SIZE*NUM_X, CROP_SIZE*NUM_Y))

strStub = '{num:02d}'.format(num=stub)
img_size = Image.open(directory+"stub"+strStub+"\\fld0001\\search.png").size

def ImageGet(index=0):
    # Get field
    strField = '{num:04d}'.format(num=int(data.iloc[index]["fieldnum"]))
    
    # Calculate position of particle on image
    partX = int(data.iloc[index]["X_cent"])
    partY = img_size[1]-int(data.iloc[index]["Y_cent"])
    box = (
        partX-int(0.5*CROP_SIZE), 
        partY-int(0.5*CROP_SIZE), 
        partX+int(0.5*CROP_SIZE), 
        partY+int(0.5*CROP_SIZE)
    )
    
    # Open image and crop
    im = Image.open(directory+"stub"+strStub+"\\fld"+strField+"\\search.png")
    im = im.convert('L')
    imCrop = im.crop(box)
    
    return imCrop

for i in range(len(data)):
    imCrop = ImageGet(i)

    # Determine location to paste image
    yy=int(floor(i/NUM_X))
    xx=i%NUM_X
    box = (
        int(xx*CROP_SIZE), 
        int(yy*CROP_SIZE), 
        int((xx+1)*CROP_SIZE), 
        int((yy+1)*CROP_SIZE)
    )
    imNew.paste(imCrop, box)

d = ImageDraw.Draw(imNew)
for yy in range(1,NUM_Y):
    d.line((0, yy*CROP_SIZE, NUM_X*CROP_SIZE, yy*CROP_SIZE), fill=64, width=1)
for xx in range(1,NUM_X):
    d.line((xx*CROP_SIZE, 0, xx*CROP_SIZE, NUM_Y*CROP_SIZE), fill=64, width=1)

FileImage = directory+"Stub"+str(stub)+".png"
imNew.save(FileImage)
display.Image(filename=FileImage)



