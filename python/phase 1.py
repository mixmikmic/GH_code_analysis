import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pickle

'''
This function detects the percentage of image covered with text. It doesn't use techniques like OCR (optical character recognition)
rather it uses edge detection and thresholding to determin the location of text.

Because of this it never returns 0% ever, since all images have edges.
'''
def detectTextArea(img):
    img_gray = img
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_sobel = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    _, img_threshold = cv2.threshold(img_sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    img_threshold = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, element)
    image, contours, heirarchy = cv2.findContours(img_threshold, 0, 1)
    boundRect = []
    area = 0
    for contour in contours:
        # print(contour.size)
        if contour.size > 100:
            contours_poly = cv2.approxPolyDP(contour, 3, True)
            rect = cv2.boundingRect(contours_poly)
#             print(rect)
            area += rect[2]*rect[3]
            if rect[2] > rect[3]:
                boundRect.append(rect)
                
    return area/img_gray.size, boundRect

'''
This function detects and returns the number of faces in the provided image. The faces are supposed to have an almost neutral
emotion and SHADES don't work.
'''

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')


def faceCounter(img, cascade=face_cascade):
    img_gray = img
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_eq = cv2.equalizeHist(img_gray)
#     key = 0
#     while key != 27:
#         cv2.imshow('disp',img_eq)
#         key = cv2.waitKey()
    faces = cascade.detectMultiScale(img_eq,minSize=(50, 50),flags=1,scaleFactor=1.2)
    return len(faces)

'''
This function returns the number of unique shades of gray present in the image. It filters out shades that occupy less than
0.5% of the image.

The reason for doing this is to get only the visually impactful colors instead of all of them. For example, a screenshot of a 
notepad document with text contains all 256 shades, but the visually impactful are just two ,i.e., Black and White.
'''
def getUniqueColors(img):
    img_gray = img
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total_pixels = img_gray.size
    x, counts = np.unique(img_gray, return_counts=True)
    colors = []
    area = []
#     print(x.size)
    for i in range(x.size):
        if float(counts[i])/total_pixels > 0.005:
#             print(x[i])
            colors.append(x[i])
#             print()
            area.append(float(counts[i])/total_pixels)
    area.sort(reverse = True)
#     print(colors, area)
    return len(colors), area

'''
This code snippet forms the data and label matrices
'''

data = np.ones((2,2),np.float32)
labels = np.ones((1),np.int32)
if not os.path.exists('data.pckl') and not os.path.exists('labels.pckl'):
    data = np.ones((2,2),np.float32)
    labels = np.ones((1),np.int32)
    for dirname, dirnames, filenames in os.walk('data'):
        total_ex = len(filenames)
        data = np.zeros((total_ex,13),np.float32)
        labels = np.zeros((total_ex),np.int32)
        for i, file in enumerate(filenames):
            if file.startswith('spam'):
                labels[i] = 1
            else:
                labels[i] = 0
            print(os.path.join(dirname,file))
            img = cv2.imread(os.path.join(dirname,file),0)
            h,w = img.shape[:2]
            ratio = w/float(h)
            img = cv2.resize(img, (int(ratio*720),720))
            faces = faceCounter(img)
            colors, area = getUniqueColors(img)
            txtarea, _ = detectTextArea(img)
            data[i,0] = faces
            data[i,1] = colors
            data[i,2] = txtarea
            for j in range(10):
                if j < colors:
                    data[i,j+3] = area[j]
    
    pickle.dump(data, open('data.pckl','wb'))
    pickle.dump(labels, open('labels.pckl','wb'))
else:
    data = pickle.load(open('data.pckl','rb'))
    labels = pickle.load(open('labels.pckl','rb'))

data.shape

get_ipython().magic('matplotlib inline')

label, counts = np.unique(labels, return_counts = True)

fig = plt.figure()
ax = fig.add_subplot(111)

width = 0.35

N = np.arange(2)

rect1 = ax.bar(0.2, counts[0], width, color='green')

rect2 = ax.bar(0.2+width+0.2, counts[1], width, color='blue')

ax.set_ylabel('Number of Examples')
ax.set_title('Bar Plot for Spam and Notspam')
lab = []
if label[0] == 0:
    lab = ['Notspam', 'Spam']
else:
    lab = ['Spam', 'Notspam']
ax.set_xlim(0,1.3)
ax.set_xticks([0.2+0.175, 0.4+0.35+0.175])
ax.set_xticklabels(lab)
ax.set_xlabel('Classes')
plt.show()

xlabels = ['Number of faces', 'Number of unique colors', 'Percentage of text',
           'Percentage of top color 1',
           'Percentage of top color 2',
           'Percentage of top color 3',
           'Percentage of top color 4',
           'Percentage of top color 5',
           'Percentage of top color 6',
           'Percentage of top color 7',
           'Percentage of top color 8',
           'Percentage of top color 9',
           'Percentage of top color 10'
          ]
for i in range(13):
    fig = plt.figure()
    bins, counts = np.unique(data[:,i], return_counts = True)
    bins = bins.size
    if bins > 150:
        bins = 150
    maxcount = counts.max()
    n,bins,patches = plt.hist(data[:,i],bins,facecolor='green',alpha=0.75)
    plt.ylim(0,maxcount)
    plt.xlabel(xlabels[i])
    plt.ylabel('Number of Examples')
    plt.title('Feature {}'.format(i+1))
    plt.show()



