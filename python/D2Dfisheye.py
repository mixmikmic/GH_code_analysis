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

def getD2Dimage(num,imageName):
    imURL=''
    imSize=0
    imDict = feedImages[num]
    
    # Download image, resize if necessary and save in module directory
    for resource in imDict['Assets'][0]['Resources']:
        if ((resource['Dimensions'][0]>=imSize and imSize < desiredResolution) or (resource['Dimensions'][0]<=imSize and imSize > desiredResolution and resource['Dimensions'][0] > desiredResolution)):
            imSize = resource['Dimensions'][0]
            imURL = resource['URL']
    print('downloading '+imURL)    
    im = Im.open(requests.get(imURL, stream=True).raw)
    if (imSize > desiredResolution):
        im = im.resize((desiredResolution,desiredResolution))
        print('resizing image from '+str(imSize)+' to '+str(desiredResolution))
    im.save(imageName)

getD2Dimage(11,'VLT_11.jpg')



