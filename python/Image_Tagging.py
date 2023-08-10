# Assumes Python 3
get_ipython().magic('reset')
get_ipython().magic('env CLARIFAI_APP_ID=<your_app_id>')
get_ipython().magic('env CLARIFAI_APP_SECRET=<your_app_secret>')
from clarifai.client import ClarifaiApi
clarifai_api = ClarifaiApi()
import pickle

import os
path = os.getcwd()
index = 0
name = str(index) + '.jpg'
fullname = os.path.join(path, name)
fullname
results = clarifai_api.tag_images(open(fullname, 'rb'))
print(results)

results['results'][0]['result']['tag']['classes'][0:5]

with open('uniqueImages.pkl','rb') as f:
    uniqueImages = pickle.load(f)
f.close()

len(uniqueImages)

parts = uniqueImages[0].split('.')
print(parts)
num = int(parts[0])
print(num)
name = parts[0] + '.jpg'
print(name)
altname = uniqueImages[0]

imTagDict = {}
uniqueTags = []
for name in uniqueImages:
    fullname = os.path.join(path, name)
    index = int(name.split('.')[0])
    print(index, end="")
    results = clarifai_api.tag_images(open(fullname, 'rb'))
    tags = results['results'][0]['result']['tag']['classes'][0:3]
    for tag in tags:
        uniqueTags.append(tag)
    uniqueTags = set(uniqueTags)
    uniqueTags = list(uniqueTags)
    imTagDict[index] = tags
print(imTagDict)
print(uniqueTags)

len(uniqueTags)

tagCount = []
for i in range(len(uniqueTags)):
    tagCount.append(0)
counter = 0
for i in range(len(uniqueTags)):
    if counter >= 205:
        break
    else:
        tag = uniqueTags[i]
        counter +=1
        #print(tag)
        for im in imTagDict:
            mylist = imTagDict[im]
            if im % 10 == 0:
                #print(mylist)
                pass
            for imtag in mylist: 
                if imtag == tag and im % 10 == 0:
                    #print(imtag)
                    pass
                if imtag == tag:
                    tagCount[i] += 1                
print(tagCount)

class Tag:
    def __init__(self, name, number):
        self.name = name
        self.number = number

finalTagList = []
for i in range(len(uniqueTags)):
    name = uniqueTags[i]
    number = tagCount[i]
    myTag = Tag(name, number)
    finalTagList.append(myTag)
len(finalTagList)

sortedTagList = sorted(finalTagList, key=lambda x: x.number, reverse=True)

print(sortedTagList[0].name, sortedTagList[0].number)
print(sortedTagList[1].name, sortedTagList[1].number)
print(sortedTagList[2].name, sortedTagList[2].number)
print(sortedTagList[3].name, sortedTagList[3].number)
print(sortedTagList[4].name, sortedTagList[4].number)
print(sortedTagList[5].name, sortedTagList[5].number)
print(sortedTagList[6].name, sortedTagList[6].number)
print(sortedTagList[7].name, sortedTagList[7].number)
print(sortedTagList[8].name, sortedTagList[8].number)
print(sortedTagList[9].name, sortedTagList[9].number)
print(sortedTagList[10].name, sortedTagList[10].number)
print(sortedTagList[11].name, sortedTagList[11].number)
print(sortedTagList[12].name, sortedTagList[12].number)
print(sortedTagList[13].name, sortedTagList[13].number)
print(sortedTagList[14].name, sortedTagList[14].number)

finalTagList = []
for i in range(15):
    tag = []
    tag.append(sortedTagList[i].name)
    finalTagList.append(tag)

finalTagList

for i in range(len(finalTagList)):
    name = finalTagList[i][0]
    for im in imTagDict:
        myList = imTagDict[im]
        for imTag in myList:
            if name == imTag:
                finalTagList[i].append(str(im) + '.jpg')
for i in range(len(finalTagList)):
    print(len(finalTagList[i]))

from IPython.display import display, Image

testTag = finalTagList[4]
print(testTag)
print(testTag[4])
print(len(testTag))

for i in range(len(testTag)):
    if i==0:
        pass
    else:
        im = Image(filename=testTag[i])
        #display(im)

index = 155
name = str(index) + '.jpg'
fullname = os.path.join(path, name)
test = clarifai_api.tag_images(open(fullname, 'rb'))
test['results'][0]['result']['tag']['classes'][0:5]

testTag = finalTagList[0]
print(testTag)
print(testTag[0])
print(len(testTag))

for i in range(len(testTag)):
    if i==0:
        pass
    else:
        im = Image(filename=testTag[i])
        #display(im)

testTag = finalTagList[5]
print(testTag)
print(testTag[5])
print(len(testTag))

for i in range(len(testTag)):
    if i==0:
        pass
    else:
        im = Image(filename=testTag[i])
        #display(im)

testTag = finalTagList[7]
print(testTag)
print(testTag[7])
print(len(testTag))

for i in range(len(testTag)):
    if i==0:
        pass
    else:
        im = Image(filename=testTag[i])
        #display(im)

testTag = finalTagList[8]
print(testTag)
print(testTag[8])
print(len(testTag))

for i in range(len(testTag)):
    if i==0:
        pass
    else:
        im = Image(filename=testTag[i])
        #display(im)

testTag = finalTagList[9]
print(testTag)
print(len(testTag))

for i in range(len(testTag)):
    if i==0:
        pass
    else:
        im = Image(filename=testTag[i])
        #display(im)

finalTagList

finalTagList2 = finalTagList[0:9]
finalTagList2.append(list(finalTagList[11]))
finalTagList2.append(list(finalTagList[14]))
finalTagList2

len(finalTagList2)
print(type(finalTagList2))

fname = open('taggedImageLists.pkl', 'wb')
pickle.dump(finalTagList2, fname)
fname.close()

