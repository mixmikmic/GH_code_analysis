import json
import csv
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as mplplot
import matplotlib.patches as patches
import dateutil.parser
import pickle
import sys
import regex
import copy
import itertools
import gc
from collections import Counter

verbose = True
if verbose :
    import pprint
    from IPython.core.display import display
    pprinter = pprint.PrettyPrinter(indent=4)

get_ipython().magic('matplotlib inline')

sampleDataFileName = 'decoding-the-civil-war-classifications.csv'
subjectDataFileName = 'decoding-the-civil-war-subjects.csv'
liveDate = dateutil.parser.parse("2016-06-20T00:00:00.00Z")

class BoxMatcher() :
    
    def __init__(self, _overlapThreshold = 0.7) :
        self.overlapThreshold = _overlapThreshold
        self.box = None  
        
    def compare(self, otherBox) :
        if self.box is None :
            self.setBox(otherBox)
            return True
        # define "identity" as a degree of area overlap
        selfArea = self.box.width * self.box.height
        otherArea = otherBox.width * otherBox.height
        dx = min(self.box.x + self.box.width, otherBox.x + otherBox.width) - max(self.box.x, otherBox.x) 
        dy = min(self.box.y + self.box.height, otherBox.y + otherBox.height) - max(self.box.y, otherBox.y)
        if dx < 0 or dy < 0 :
            return False
        areaOfOverlap = dx*dy
        unionOfAreas = selfArea + otherArea - areaOfOverlap
        overlapFraction = areaOfOverlap/unionOfAreas
        return overlapFraction > self.overlapThreshold
    
    def setBox(self, newBox) :
        self.box = newBox
          
    @staticmethod
    def mean(boxes):
            
        meanCoMX = meanCoMY = meanWidth = meanHeight = 0.0
        for box in boxes :
            boxCoM = (box.x + 0.5*box.width, box.y + 0.5*box.height)
            meanCoMX += boxCoM[0]
            meanCoMY += boxCoM[1]
            meanWidth += box.width
            meanHeight += box.height
        
        meanBox = TelegramBox(meanCoMX/float(len(boxes)) - 0.5*meanWidth/float(len(boxes)),
                             meanCoMY/float(len(boxes)) - 0.5*meanHeight/float(len(boxes)),
                             meanWidth/float(len(boxes)),
                             meanHeight/float(len(boxes)),
                             {'nBoxes' : len(boxes)})
        return meanBox

class TelegramBox() :
    
    def __init__(self, x, y, width, height, data) :
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.data = data
        
    def __str__(self) :
        return 'TelegramBox(x={}, y={}, width={}, height={}, subject={})'.format(self.x, self.y, self.width, self.height, self.data)
    
    def __repr__(self) :
        return self.__str__()

subject_data = []
subjectColumns = ['subject_id', 'huntington_id', 'url']
with open(subjectDataFileName) as csvfile :
    parsedSubjectCsv = csv.DictReader(csvfile)
    numPrinted = 0
    for subject in parsedSubjectCsv :
        parsedLocations = json.loads(subject['locations'])
        parsedMetaData = json.loads(subject['metadata'])
        if 'hdl_id' not in parsedMetaData :
            continue
        subject_data.append({'subject_id': int(subject['subject_id']), 
                             'huntington_id' : parsedMetaData['hdl_id'],
                             'collection' : parsedMetaData['hdl_id'][3:5],
                             'ledger' : int(parsedMetaData['hdl_id'][6:8]),
                             'page' : int(parsedMetaData['hdl_id'][9:12]),
                             'variant' : parsedMetaData['hdl_id'][12:],
                             'telegramNumbers' : [int(regex.findall(r'[0-9]+', telegramNumber)[0]) for telegramNumber in regex.findall(r"[\w']+", parsedMetaData['#telegrams']) if len(telegramNumber) > 3], 
                             'url' : parsedLocations['0']})
subjectsFrame = pd.DataFrame.from_records(subject_data, index = 'subject_id')
display(subjectsFrame)

allBoxes = {}

onePrinted = False

with open(sampleDataFileName) as csvfile :
    parsedCsv = csv.DictReader(csvfile)
    nSubjectsParsed = 0
    for recordIndex, record in enumerate(parsedCsv) :
        
        subjectBoxes = []
        
        # check the date that the classification was made
        parsedMetadata = json.loads(record["metadata"])
        parsedDate = dateutil.parser.parse(parsedMetadata['started_at'])

        # skip "testing" data before the site went live
        if parsedDate < liveDate :
            continue
        
        # parse the annotations and the subject data
        parsedAnnotations = json.loads(record["annotations"])
        parsedSubjectData = json.loads(record["subject_data"])
        
        #loop over tasks in the annotation
        for task in parsedAnnotations :
            # Check for recorded box data
            if task['task'] == "T2" and task['value'] :
                for box in task['value'] :
                    subjectBoxes.append(TelegramBox(box['x'], box['y'], box['width'], box['height'], int(record['subject_ids'])))
                
        nSubjectsParsed += 1
        if int(record['subject_ids']) in allBoxes :
            allBoxes[int(record['subject_ids'])].append((recordIndex, subjectBoxes))
        else :
            allBoxes.update({int(record['subject_ids']) : [(recordIndex, subjectBoxes)]}) 

allBoxData = []
for key, boxes in allBoxes.items() :
    for boxData in boxes :
        for boxDatum in boxData[1] :
            allBoxData.append({'subjectKey' : key, 
                               'box' : boxDatum, 
                               'boxX' : boxDatum.x, 
                               'boxY' : boxDatum.y, 
                               'boxW' : boxDatum.width, 
                               'boxH' : boxDatum.height })
            
allBoxesFrame = pd.DataFrame(data = allBoxData)

allBoxesFrameIndex = pd.MultiIndex.from_arrays([allBoxesFrame['subjectKey'], 
                                                allBoxesFrame['boxY'], 
                                                allBoxesFrame['boxH'], 
                                                allBoxesFrame['boxX'], 
                                                allBoxesFrame['boxW'],])

allBoxesFrame.set_index(allBoxesFrameIndex, inplace=True, drop=False)
allBoxesFrame.sort_index(inplace=True)
display(allBoxesFrame)

# Add new column listing the most likely box id
allBoxesFrame['bestBoxIndex'] = pd.Series(np.zeros_like(allBoxesFrame['subjectKey']),
                                          index=allBoxesFrame.index)
subjectKey = None
groupIndex = None
boxMatcher = BoxMatcher(0.7)

for index, data in allBoxesFrame.iterrows() :
    if subjectKey != index[0] :
        subjectKey = index[0]
        groupIndex = 0
    if not boxMatcher.compare(data['box']) :
        groupIndex += 1
    allBoxesFrame.ix[index, 'bestBoxIndex'] = groupIndex  
    boxMatcher.setBox(data['box'])
    
display(allBoxesFrame[['box', 'bestBoxIndex']])

allBoxesFrameReindexed = allBoxesFrame.reset_index(level=[1,2,3,4], drop=True)  
allBoxesFrameReindexed.set_index('bestBoxIndex', append=True, inplace=True)
display(allBoxesFrameReindexed)

aggregatedBoxesFrame = allBoxesFrameReindexed.groupby(level=[0,1]).aggregate({'box' : BoxMatcher.mean})
display(aggregatedBoxesFrame)

aggregatedBoxesFrame.reset_index(level=[1], drop=False, inplace=True)
aggregatedBoxesFrame = aggregatedBoxesFrame.merge(subjectsFrame, how='left', left_index=True, right_index=True)
display(aggregatedBoxesFrame)

def plotBox(box, color = None, axis = None) :
    boxFigure = mplplot.gcf()
    boxPlot = axis
    
    if axis is None :
        boxFigure = mplplot.figure(figsize=(5,5))
        boxPlot = boxFigure.add_subplot(111, aspect='equal')

    boxIsReliable = (box.data is None or not isinstance(box.data, dict) or ('nBoxes' in box.data and box.data['nBoxes'] > 1))
        
    boxPlot.add_patch(
        patches.Rectangle(
            (box.x, box.y),   # (x,y)
            box.width,          # width
            box.height,          # height
            fill=False,
            hatch = None if boxIsReliable else '//',
            alpha = 1.0 if boxIsReliable else 0.2,
            ls = '-' if boxIsReliable else '--',
            color='r' if color is None else color
        )
    )
    return boxPlot

def plotBoxes(boxes, colors = None, axis = None) :
    boxesPlot = axis
    
    maxYVals = []
    maxXVals = []
    for boxIndex, box in enumerate(boxes) :
        boxesPlot = plotBox(box, color = colors[boxIndex] if colors is not None else 'k', axis = boxesPlot)
        maxXVals.append(box.x + box.width)
        maxYVals.append(box.y + box.height)
        
    mplplot.xlim(0, np.max(maxXVals))
    mplplot.ylim(0, np.max(maxYVals))
    
    return boxesPlot

boxFigure = mplplot.figure(figsize=(10,5))
spectralColorMap = matplotlib.cm.get_cmap('viridis')
idx = pd.IndexSlice

allBoxPlot = boxFigure.add_subplot(121, aspect='equal')

allBoxes = allBoxesFrame.ix[idx[1959274], idx[:]][['box', 'bestBoxIndex']]
maxBoxGroup = np.max(allBoxes['bestBoxIndex'].values)
boxColors = [ spectralColorMap(boxGroup/float(maxBoxGroup)) for boxGroup in allBoxes['bestBoxIndex'].values ]

plotBoxes(allBoxes['box'].values, boxColors, axis=allBoxPlot)

meanBoxPlot = boxFigure.add_subplot(122, aspect='equal')

meanBoxes = aggregatedBoxesFrame.ix[idx[1959274], idx[:]][['box']]
maxBoxGroup = len(meanBoxes)
boxColors = [ spectralColorMap(boxGroup/float(maxBoxGroup)) for boxGroup, _ in enumerate(meanBoxes.values) ]
    
plotBoxes(meanBoxes['box'].values, boxColors, axis=meanBoxPlot)

sys.path.append('/Library/Python/2.7/site-packages')
import mysql.connector
#testSubjectData = lineGroupedTranscriptionLineDetails.iloc[0]
connection = mysql.connector.connect(user=os.environ['DCW_MYSQL_USER'], password=os.environ['DCW_MYSQL_PASS'],
                              host=os.environ['DCW_MYSQL_HOST'],
                              database='dcwConsensus')

cursor = connection.cursor()
sentence = ''
try:
    boxInsertQuery = ("INSERT INTO SubjectBoxes (subjectId, bestBoxIndex, collection, ledger, "
                      "page, meanX, meanY, meanWidth, meanHeight, numBoxesMarked) "
                      "SELECT id, %s, %s, %s, %s, %s, %s, %s, %s, %s "
                      "FROM Subjects WHERE huntingtonId = %s")
    
    telegramInsertQuery = ("INSERT INTO SubjectTelegrams (subjectId, collection, ledger, "
                      "page, telegramId) "
                      "SELECT id, %s, %s, %s, %s "
                      "FROM Subjects WHERE huntingtonId = %s")
    
    currentSubject = -1
    subjectId = None
    
    # Loop over aggregated lines in consensus data 
    for index, row in aggregatedBoxesFrame.iterrows() :
        # Insert the aggregated line data
        bestBoxIndex = int(row['bestBoxIndex'])
        boxData = (bestBoxIndex, 
                   row['collection'], 
                   row['ledger'], 
                   row['page'], 
                   row['box'].x,
                   row['box'].y,
                   row['box'].width,
                   row['box'].height,
                   row['box'].data['nBoxes'],
                   row['huntington_id']
                  )
        cursor.execute(boxInsertQuery, boxData)
        
    # Loop over subject frame and insert telegram data
    for index, row in subjectsFrame.iterrows() :
        for telegramNumber in row['telegramNumbers'] :
            telegramData = (row['collection'], 
                            row['ledger'], 
                            row['page'], 
                            telegramNumber, 
                            row['huntington_id'])
            cursor.execute(telegramInsertQuery, telegramData)
            
except mysql.connector.Error as err:
    print("Failed INSERT: {0}, {1}".format(sentence, err))
    
connection.commit()

cursor.close()
connection.close()



