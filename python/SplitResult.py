

outputPath = 'output/testData5Results.tsv'
lineIdx = []
with open(outputPath, 'r') as f:
    n = 0
    for l in f.readlines():
        if l[:3] =='# 1' or l[:3] == '# 0':
            # print(l)
            lineIdx.append(n)
        n+=1
    lineIdx.append(n-1)

import pandas as pd

outputPath = 'result.txt'

pd.read_csv(outputPath, sep = '\t', header = None)

## It should not be so many things
## We need a text only have one Sentence.
## This is an error that I should avoid in the new Project.
len(lineIdx)

lineIdx[-5:]

lineIdx[2246]

lineIdx[0]

with open(outputPath, 'r') as f:
    lines = f.readlines()
    for ii in range(len(lineIdx)-1):
        sentence = lines[lineIdx[ii]:lineIdx[ii+1]]
sentence

import pandas as pd
from io import StringIO
f = StringIO(''.join(sentence[1:]))
result = pd.read_csv(f, sep = '\t', comment = '#', header= None, skip_blank_lines= False)
score = sentence[0]
score.replace('# ', '').replace('\n','')

result

outputPath = 'output/testData5Results.tsv'
results = splitResult(outputPath)
len(results)

scores2 = [r[0] for r in results]
scores2

outputPath = 'output/testData5Results.tsv'
lineIdx = []
scores1 = []
with open(outputPath, 'r') as f:
    n = 0
    for l in f.readlines():
        if l[:3] =='# 1' or l[:3] == '# 0':
            scores1.append(l.replace('# ', '').replace('\n',''))
            lineIdx.append(n)
        n+=1
    lineIdx.append(n-1)

scores1 == scores2

import pandas as pd
from io import StringIO


def splitResult(outputPath, mode = None):
    lineIdx = []
    scores1 = []
    
    with open(outputPath, 'r') as f:
        n = 0
        for l in f.readlines():
            if l[:3] =='# 1' or l[:3] == '# 0':
                scores1.append(l.replace('# ', '').replace('\n',''))
                lineIdx.append(n)
            n+=1
        lineIdx.append(n-1)
    results = []
    with open(outputPath, 'r') as f:
        lines = f.readlines()
        for ii in range(len(lineIdx)-1):
            sentence = lines[lineIdx[ii]:lineIdx[ii+1]]
            f = StringIO(''.join(sentence[1:]))
            result = pd.read_csv(f, sep = '\t', 
                                 comment = '#', 
                                 header= None, 
                                 skip_blank_lines= False)
            score = sentence[0]
            score = score.replace('# ', '').replace('\n','')#
            results.append([score, result])
    
    assert scores1 == [r[0] for r in results] # Test the right result
    
    return results

outputPath = 'dev/Text/2ab病史特点-61-2Rslt.txt'
splitResult(outputPath)

