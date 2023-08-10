from pytube import YouTube
'''
Video 1: "https://www.youtube.com/watch?v=eZK2_-rIzJE"
Video 2: 
'''
yt = YouTube("https://www.youtube.com/watch?v=HLZheaytseQ")

# Once set, you can see all the codec and quality options YouTube has made
# available for the perticular video by printing videos.
print(yt.filename)
print yt.filter('mp4')

#downloads FLV from youtube
video = yt.get('mp4','360p')

video.download('data/{}.mp4'.format(yt.filename.split()[0].lower()))

import cv2
import matplotlib.pyplot as plt
import imageio
from skimage.util import pad

get_ipython().magic('matplotlib qt')

nums = range(1300,1500,10)

filename = 'data/derrick.mp4'
vid = imageio.get_reader(filename, 'ffmpeg')

#works on 1125, 1150, 1200, 1350

for num in nums:
    image = vid.get_data(num)
    #imageio.imwrite('test-images/test-{}.png'.format(num), pad(image,((60,60),(0,0),(0,0)),'constant'))
    imageio.imwrite('images/train-{}.png'.format(num), pad(image,((60,60),(0,0),(0,0)),'constant'))

# Test set starts at 675 to 1000

#1010 - 1200
#1200-1288
#1300-1500
#DO NEXT 1500-1700



im_num = 1500

image = vid.get_data(im_num)
fig = plt.figure(figsize=(11,10))
plt.suptitle('image #{}'.format(im_num), fontsize=20)
plt.imshow(image)

class TestClass():
    def __init__(self,fbase,txt,nums):
        self.fbase = fbase
        self.nums = nums
        self.img = self.nums.pop(0)
        
        #initialize filename
        self.fname = self.fbase+'{}.png'.format(self.img)
        
        self.point = ()
        self.pair = ()
        self.clicks = 0
        self.boxes = '"{}":'.format(self.fname)
        self.txt = txt
        self.show_img(self.fname)
    
    def show_img(self, name):
        self.boxes = '"{}":'.format(name)
        self.point = ()
        pic = imageio.imread(name)
        fig = plt.figure(figsize=(11,10))
        plt.suptitle('image #{}'.format(self.img), fontsize=20)
        plt.imshow(pic)
        cid = fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        clid = fig.canvas.mpl_connect('close_event', self.__handle_close__)
        plt.show()

    def __onclick__(self, click):
        self.clicks += 1
        if self.clicks % 2 == 0:
            x, y = self.point
            self.point = (x, y, float(int(click.xdata)), float(int(click.ydata)))
            self.boxes += ' {},'.format(self.point)
        else:
            self.point = (float(int(click.xdata)), float(int(click.ydata)))
        print self.point
        return self.point
    
    def __handle_close__(self, evt):
        self.boxes = self.boxes.strip(',')
        self.boxes += ';\n'
        if self.point:
            with open(self.txt, 'a') as text_file:
                text_file.write(self.boxes)
        self.img = self.nums.pop(0)
        self.fname = self.fbase+'{}.png'.format(self.img)
        self.show_img(self.fname)

get_ipython().magic('matplotlib qt')
# a = TestClass('test-images/test-','train.idl',nums)
a = TestClass('images/train-','train.idl',nums)





