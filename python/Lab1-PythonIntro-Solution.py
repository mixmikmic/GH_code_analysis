#Function definition
def helloWorld():
    print "Hello World"

#Calling the function
helloWorld()

def rectangleInfo(width, height):
    perimeter = 2*width + 2*height
    area = width * height
    return perimeter, area

myPerimeter, myArea = rectangleInfo(10,10)
print "My perimeter is", myPerimeter, "and my area is", myArea, "."

def countByNum(startingNumber=1,stepSize=1,numIters=10):
    text = ""
    count = startingNumber
    for i in range(0,numIters):
        text += str(count)
        if i < numIters-1:
            text += ","
        count += stepSize
    print text

#The different ways to use default parameters 
countByNum()
countByNum(2, 2)
countByNum(stepSize=2)

def fibonacci(num, n0 = 1, n1 = 1):
    if num == 1:
        return n0, n1
    else:
        t0, t1 = fibonacci(num-1, n0, n1)
        return t1, t0+t1

print fibonacci(6, 2, 2)
print fibonacci(24, 3, 5)

from math import pi

class Circle():
    
    def __init__(self, radius, center_x=0.0, center_y=0.0):
        self.radius = radius
        self.x = center_x
        self.y = center_y
        
    def getCircumference(self):
        return 2*pi*self.radius
    
    def getArea(self):
        return pi*self.radius**2
    
    def getArc(self, angle):
        return self.radius * angle
    
myCirc = Circle(25)
print myCirc.getCircumference()
print myCirc.getArea()
print myCirc.getArc(1.5)

class Shape():
    
    def getPerimeter(self):
        pass
    
    def getArea(self):
        pass
    
class Square(Shape):
    
    def __init__(self, size):
        self.size = size
    
    def getPerimeter(self):
        return 4*self.size
    
    def getArea(self):
        return self.size**2
    
    
mySquare = Square(10)
if isinstance(mySquare, Shape):
    print mySquare.getPerimeter()

myList = []

#Add elements to the list
myList.append(3)
myList.append(4)
#Insert the number 10 at the 0th index
myList.insert(0, 10)
print myList
print myList[1]

#Sum up the list
total = 0
for num in myList:
    total += num
    
print total

myDictionary = {}

#Add elements to dictionary using key/value system
myDictionary["BYU"] = "University"
myDictionary["Provo"] = "City"

print myDictionary

#Another way to instatiate a dictionary
myDictionary2 = {'Name': 'Zara', 'Age': 7, 'Grade': '1'}

#Go through data in a dictionary
for key in myDictionary2:
    print myDictionary2[key]

#Remove data
del myDictionary["Provo"]
print myDictionary

myDictionary2.clear()
print myDictionary2



#Test Case
myQueue = MyPriorityQueue()
myQueue.push("were toiling upward in the night.", 2)
myQueue.push("The heights by great men reached and kept", 5)
myQueue.push("-Longfellow", 1)
myQueue.push("were not attained by sudden flight,", 4)
myQueue.push("but they, while their companions slept,", 3)
print myQueue.pop()
print myQueue.pop()
print myQueue.pop()
print myQueue.pop()
print myQueue.pop()

import math
print math.pi

from math import pi
print pi

from math import pi as taco
print taco

from math import *
print e

import numpy as np
print np.version.version

import numpy as np
a = np.matrix([[1, 2],[3,4]])

print a
print
print a[0,1] #Row, Column zero-based indexing
print
print a[:,1] #Grab all rows and the first column
print
print a + 1
print
print 3*a
print
print np.multiply(a,a) #Element-wise multiply
print
print a*a #True matrix multiply

from scipy.ndimage import imread
geese = imread('geese.jpg')

import matplotlib.pyplot as plt
plt.imshow(geese)
plt.show()

#Notice geese is a 3-dimensional Numpy array (row, col, rgb).
print geese

import numpy as np
import matplotlib.pyplot as plt

def flip(image):
    image = image[::-1]
    plt.imshow(image)
    plt.show()

def greenify(image):
    image = image[:,:,2]
    plt.imshow(image)
    plt.show()
    
from scipy.ndimage import imread
geese = imread('geese.jpg')

flip(geese)
greenify(geese)



