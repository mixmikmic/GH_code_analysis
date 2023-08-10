from PIL import Image
import numpy as np

im = Image.open('cameraman.png')
im.show()

imlist = list(im.getdata())
print(imlist[0:10])
imarray = np.array(im.getdata())
print(imarray[0:10])

print(im.size)
imarray2d = np.reshape(imarray, im.size)
print(imarray2d)
imlist2d = np.reshape(imlist, im.size)
print(imlist2d)

imcolor = Image.open('smile.jpeg')
imcolorarray = np.array(imcolor)
print(imcolor.size)
imcolorarray2d = np.reshape(imcolorarray, [imcolor.size[0],imcolor.size[1], 3])
print(imcolorarray2d)

newcolorImage = Image.fromarray(imcolorarray2d)
newcolorImage.show()

newcolorImage.save('new.jpg')
newcolorImage.save('new.png')

