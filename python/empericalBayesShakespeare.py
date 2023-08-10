import pandas as pd
import numpy as np
import re
import os
from collections import Counter

#use this list to exclude certain plays
excludedPlays = []

spDir = 'shakespeare-plays-plus'
spSubDir = [spDir + '/'+ x for x in os.listdir(spDir) if '.' not in x]
spPlays = []

for subDir in spSubDir:
    spPlays = spPlays + [subDir + '/' + x for x in os.listdir(subDir) if 'txt' in x and x not in excludedPlays]

CounterDict = {}

for spPlay in spPlays:
    playName = spPlay.split('/')[-1].split('.')[0]
    CounterDict[playName] = Counter()
    with open(spPlay, 'r', encoding='utf-16') as playTxt:
        for line in playTxt:
            if '<' in line:
                continue
            if line.split():
                #no additional text processing other than converting everything to lower case
                CounterDict[playName].update(Counter(re.findall('\w+', line.lower())))

#create a totalCounter for all plays
totalCounter = sum([playCounter for playCounter in CounterDict.values()], Counter())
print('distinct words:', len(totalCounter))
print('total words:', sum(totalCounter.values()))

for playName in CounterDict.keys():
    print(playName)
    #create hold one out counter
    hooCounter = totalCounter - CounterDict[playName]
    
    #use set difference to calculate actual number of "new words" in a play
    print('actual new words', len(set(CounterDict[playName].keys()) - set(hooCounter.keys())))
    
    #use "corbet formula" to calculate expected number of "new words" in a play
    t = sum(CounterDict[playName].values())/sum(hooCounter.values())
    expectedWords = sum(v*(t**k)*((-1)**(k-1)) for k, v in Counter(hooCounter.values()).items())
    print('expected new words', round(expectedWords))
    
    print('*'*50)



