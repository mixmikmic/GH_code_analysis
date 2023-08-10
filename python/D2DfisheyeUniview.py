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

UniviewModulePath = r'C:\Users\msubbarao\SCISS\Uniview Theater 2.0\Custom Modules'
def makeUVmodule(num,moduleName):
    imURL=''
    imSize=0
    imDict = feedImages[num]
    
    # FIRST Create a Directory in Uniview Custom Modules
    UniviewModuleDir=os.path.join(UniviewModulePath,moduleName)
    if not os.path.exists(UniviewModuleDir):
        os.makedirs(UniviewModuleDir)
    os.path.exists(UniviewModuleDir)    
    
    # SECOND Download image, resize if necessary and save in module directory
    for resource in imDict['Assets'][0]['Resources']:
        if ((resource['Dimensions'][0]>=imSize and imSize < desiredResolution) or (resource['Dimensions'][0]<=imSize and imSize > desiredResolution and resource['Dimensions'][0] > desiredResolution)):
            imSize = resource['Dimensions'][0]
            imURL = resource['URL']
    print('downloading '+imURL)    
    im = Im.open(requests.get(imURL, stream=True).raw)
    if (imSize > desiredResolution):
        im = im.resize((desiredResolution,desiredResolution))
        print('resizing image from '+str(imSize)+' to '+str(desiredResolution))
    imageName=moduleName+'.jpg'
    im.save(os.path.join(UniviewModuleDir,imageName))
    
    # THIRD create a .mod file for this module
    modFileText='''
    filepath +:.:./modules/'''+moduleName+'''
    
    2dobject ''' + moduleName + ''' sgFisheyeScreenRenderObject
    {
    prop.Enabled false
    prop.Depth 200
    prop.fileName ./modules/'''+moduleName+'/'+imageName+'''
    prop.alpha 1.0
    prop.tilt -45
    }'''
    print('writing .mod file')
    modFileName=moduleName+'.mod'
    writeFile = open(os.path.join(UniviewModuleDir,modFileName),'w')
    writeFile.write(modFileText)
    writeFile.close()
    
    # FORTH create a module.definition file
    definitionText='''
    <uniview>
    <name>'''+moduleName+'''</name>
    <description>'''+imDict["Title"]+'''</description>
    <creator>'''+imDict["Credit"]+'''</creator>
    <legalinfo>'''+imDict["Rights"]+'''</legalinfo>
    <version></version>
    <creationdate>'''+imDict["PublicationDate"]+'''</creationdate>
    </uniview>
    '''
    print('writing module.definition file')
    writeFile = open(os.path.join(UniviewModuleDir,'module.definition'),'w')
    writeFile.write(definitionText)
    writeFile.close()
    print('Done :)')

makeUVmodule(4,'VLT_4')



