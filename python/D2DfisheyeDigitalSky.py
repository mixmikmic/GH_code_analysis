import pandas as pd
from IPython.display import Image, HTML
from datetime import date, timedelta
import requests
from PIL import Image as Im
import os

# D2D Feed
ESOimages = 'http://www.eso.org/public/images/d2d/'
howRecent = 200 # in days
pd.set_option('max_colwidth',5000) # so we will return the full description without truncation
numReturn=20 # maximum number of results to return
desiredResolution = 4096

page = ESOimages+'?after='+(date.today()-timedelta(days=howRecent)).strftime('%Y%m%d')+'0000' #the 0000 is the time component of the date
nImg = 0
nPage=0
feedImages=[]
while (page!='' and nImg < numReturn):
    print(page)
    df=pd.read_json(page)
    # The Next parameter may or may not exist
    try:
        page=df.Next[0]
    except:
        page=''
    nPage=nPage+1
    for feedImage in df.Collections:
        if (feedImage['Assets'][0]['Resources'][0]['ProjectionType']=='Fulldome'and nImg < numReturn):
            feedImages.append(feedImage)
            nImg=nImg+1
    print(str(nPage) + ' pages parsed ' + str(nImg) + ' fulldome images found')

titleList=[]
descriptionList=[]
thumbnailList=[]
pubdateList=[]
for pic in feedImages:
    titleList.append(pic['Title'])
    descriptionList.append(pic['Description'].replace("\r\n",''))
    pubdateList.append(pic['PublicationDate'])
    for resource in pic['Assets'][0]['Resources']:
        if (resource['ResourceType']=='Thumbnail'):
            thumbnailList.append('<img src="'+resource['URL']+'"/>')            

df3 = pd.DataFrame({'title':titleList,'description':descriptionList,'thumbnail':thumbnailList,'pubdate':pubdateList},columns=['title','thumbnail','description','pubdate'])

HTML(df3.to_html(escape=False))

DSShowsPath = r'C:\DigitalSky\Shows'
DSImageDirectory = 'D2Dcontent'
DSButtonDirectory = r'C:\DigitalSky\Buttons\D2D'
def addImageDS(imageNum,buttonNum,imageName):
    imURL=''
    imSize=0
    imDict = feedImages[imageNum]
    imDir = os.path.join(DSShowsPath,DSImageDirectory)
    # Select image asset, download image, resize if necessary, and save in SHOWS directory
    for resource in imDict['Assets'][0]['Resources']:
        if ((resource['Dimensions'][0]>=imSize and imSize < desiredResolution) or (resource['Dimensions'][0]<=imSize and imSize > desiredResolution and resource['Dimensions'][0] > desiredResolution)):
            imSize = resource['Dimensions'][0]
            imURL = resource['URL']
    print('downloading '+imURL)    
    im = Im.open(requests.get(imURL, stream=True).raw)
    if (imSize > desiredResolution):
        im = im.resize((desiredResolution,desiredResolution))
        print('resizing image from '+str(imSize)+' to '+str(desiredResolution))
        imSize=desiredResolution
    imageNameExt=imageName+'.jpg'
    im.save(os.path.join(imDir,imageNameExt))
    
    # Create a Button for this image
    scriptText=''';----------------------------------------------------------------------
; Script number = F'''+str(buttonNum)+'''
; Title         = "'''+imageName+'''"
; Color         = "255,255,0"
; Created on    : '''+date.today().strftime('%Y-%m-%d')+'''
; Modified      : '''+date.today().strftime('%Y-%m-%d')+'''
; Version       : 2.3
; Created by    : '''+imDict["Credit"]+'''
; Keywords      : 
; Description   : '''+imDict["Title"]+'''
;----------------------------------------------------------------------
;
+0.1  Text Add "'''+imageName+'''" "Showpath\\'''+os.path.join(DSImageDirectory,imageNameExt)+'''" '''+str(imSize)+''' '''+str(imSize)+''' 0 90 0 0 1 0
+2    Text Locate "'''+imageName+'''" 0 0 90 0 180 180
      Text View "'''+imageName+'''" 3 100 100 100 100
      ButtonText "Remove '''+imageName+'''"
STOP
      Text View "'''+imageName+'''" 2 0 0 0 0
+2    Text "Remove '''+imageName+'''"'''
    print('writing button file')
    buttonFileName='F'+str(buttonNum)+'.sct'
    writeFile = open(os.path.join(DSButtonDirectory,buttonFileName),'w')
    writeFile.write(scriptText)
    writeFile.close()

    print('Done :)')

addImageDS(3,19,'VLT') #Pass image number, button number, object nam3

for i in range(len(feedImages)):
    ni=i+1 # convert from 0 to 1 index
    addImageDS(i,ni,'VLT'+str(ni))



