from ga4gh.client import client
c = client.HttpClient("http://1kgenomes.ga4gh.org")

import sys
import collections
import math
get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import interact, interactive, fixed
from IPython.display import display
import ipywidgets as widgets

dataset = c.search_datasets().next()

for variantSet in c.search_variant_sets(dataset.id):
    if variantSet.name == "functional-annotation":
        annotation = variantSet

annotationSet = c.search_variant_annotation_sets(variant_set_id=annotation.id).next()

def runSearch(startPos, endPos, chromosome, searchTerms, buckets):
    
    global formatSearch
    formatSearch = []
    
    for i in range(0,len(searchTerms)):
        formatSearch.append({"id":searchTerms[i]})
           
    global windowCount
    windowCount = int(buckets)
    
    global initStart
    global initEnd
    initStart = startPos
    initEnd = endPos
    
    global startPoint
    global endPoint
    startPoint = int(startPos)
    endPoint = (int(startPos)+(int(endPos)-int(startPos))/int(buckets))
    
    global yList
    global xTickList
    yList=[]
    xTickList=[]
    
    global allGraphData
    allGraphData = []
    
    global count
    count=0
    
    # formatSearch loop breaks up the search by different search terms
    for soTerms in formatSearch:
        # windowCount/bucket loop breaks up the search into multiple smaller searches from region to region
        for i in range(0,windowCount):
            searchedVarAnns=c.search_variant_annotations(variant_annotation_set_id=annotationSet.id, start=startPoint, end=endPoint, reference_name=chromosome, effects=[soTerms])

            idList = []
            startEndList = []

            for annotation in searchedVarAnns:
                idList.append(annotation.variant_id)

            countingStats(idList=idList, windowValue=windowCount, yValList=yList, startPos=startPoint, endPos=endPoint)

            startPoint+=(int(endPos)-int(startPos))/int(buckets)
            endPoint+=(int(endPos)-int(startPos))/int(buckets)
            
            del idList[:]

def countingStats(idList, windowValue, yValList,startPos, endPos):

    if len(yList)==0:
        yList.append([])
    
    yList[count].append(len(idList))
    
    if len(yList[count])==windowValue:
        global startPoint
        startPoint = int(initStart)-(int(initEnd)-int(initStart))/windowCount
    
        global endPoint
        endPoint = (int(initStart)+(int(initEnd)-int(initStart))/windowCount)-(int(initEnd)-int(initStart))/windowCount
        
        global count
        count+=1
    
        if count!=len(formatSearch):
            yList.append([])
    
    if len(yList)==len(formatSearch) and len(yList[count-1])==windowValue and count==len(formatSearch):
        plotWindowHistogram(xTickList, yList, windowValue, startPos, endPos)

def plotWindowHistogram(xAxisTicks, yAxisValues, windowVals, startPos, endPos):

    fig, ax = plt.subplots()
    
    endValues = np.empty([1,2], dtype=np.int32)

    endValues[0][0] = startPos
    endValues[0][1] = endPos
    
    colors = [str]*20
    colors[0]  = '#8B0000'
    colors[1]  = '#FF8C00'
    colors[2]  = '#8B008B'
    colors[3]  = '#556B2F'
    colors[4]  = '#006400'
    colors[5]  = '#9932CC'
    colors[6]  = '#BDB76B'
    colors[7]  = '#707B7C'
    colors[8]  = '#76D7C4'
    colors[9]  = '#F5B7B1'
    colors[10] = '#1A5276'
    colors[11] = '#BA4A00'
    colors[12] = '#AED6F1'
    colors[13] = '#F9E79F'
    colors[14] = '#6E2C00'
  
    # title and graph size formatting
    titleEffects=[]
    for key, value in searchOntologyDict.iteritems():
        for i in range(len(formatSearch)):
            if searchOntologyDict[key]==formatSearch[i]['id']:
                titleEffects.append(key)


    index=0
    for j in range(0,len(yAxisValues[index])):
        for i in range(0,count):
            if j==0:
                plt.bar(index, yAxisValues[i][j], width=1, color=colors[i], label=titleEffects[i])
            else:
                plt.bar(index, yAxisValues[i][j], width=1, color=colors[i])
            index+=1
            

    title=""
    if len(titleEffects)==1:
        ax.set_title(titleEffects[0]+"s"+" from "+str(initStart)+" to "+str(initEnd))
    else:
        if len(formatSearch)==2:
            title+=titleEffects[0]+"s"+" and "+titleEffects[1]+"s"+" "
        else:
            for i in range(0,len(titleEffects)):
                if i!=(len(titleEffects)-1):
                    title+=titleEffects[i]+"s"+", "
                else:
                    title+="and "+titleEffects[i]+"s"+" "
        ax.set_title(title+"from "+str(initStart)+" to "+str(initEnd))

        
    plt.legend(loc='upper right')
    plt.rcParams["figure.figsize"] = [15,15]
        
    plt.show()

shortDict = {'intron_variant' : 'SO:0001627', 'feature_truncation' : 'SO:0001906' , 'non_coding_transcript_exon_variant' : 'SO:0001792' , 'non_coding_transcript_variant' : 'SO:0001619', 'transcript_ablation' : 'SO:0001893'}
chromList = ('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18')

global searchOntologyDict
searchOntologyDict = {
    'stop_retained_variant' :              'SO:0001567',
    'regulatory_region_variant' :          'SO:0001566',
    'splice_acceptor_variant' :            'SO:0001574',
    'splice_donor_variant' :               'SO:0001575',
    'missense_variant' :                   'SO:0001583',
    'stop_gained' :                        'SO:0001587',
    'stop_lost' :                          'SO:0001578',
    'frameshift_variant' :                 'SO:0001589',
    'coding_sequence_variant' :            'SO:0001580',
    'non_coding_transcript_variant' :      'SO:0001619',
    'mature_miRNA_variant' :               'SO:0001620',
    'NMD_transcript_variant' :             'SO:0001621',
    '5_prime_UTR_variant' :                'SO:0001623',
    '3_prime_UTR_variant' :                'SO:0001624',
    'incomplete_terminal_codon_variant' :  'SO:0001626',
    'intron_variant' :                     'SO:0001627',
    'intergenic_variant' :                 'SO:0001628',
    'splice_region_variant' :              'SO:0001630',
    'upstream_gene_variant' :              'SO:0001631',
    'downstream_gene_variant' :            'SO:0001632',
    'TF_binding_site_variant' :            'SO:0001782',
    'non_coding_transcript_exon_variant' : 'SO:0001792',
    'protein_altering_variant' :           'SO:0001818',
    'synonymous_variant' :                 'SO:0001819',
    'inframe_insertion' :                  'SO:0001821',
    'inframe_deletion' :                   'SO:0001822',
    'transcript_amplification' :           'SO:0001889',
    'regulatory_region_amplification' :    'SO:0001891',
    'TFBS_ablation' :                      'SO:0001892',
    'TFBS_amplification' :                 'SO:0001892',
    'regulatory_region_ablation' :         'SO:0001894',
    'feature_truncation' :                 'SO:0001906',
    'feature_elongation' :                 'SO:0001907',
    'start_lost' :                         'SO:0002012',
}

multiSelect = widgets.SelectMultiple(
    description="Transcript Effects",
    options=searchOntologyDict
)

interact(runSearch,
         startPos="0",
         endPos="100000",
         chromosome=chromList,
         searchTerms=multiSelect,
         buckets="20",
         __manual="True"
         )

