# turn on inline plots
get_ipython().magic('pylab inline')

import os.path

# Convinience definitions (data locations, etc)

# the files are in the data subdirectory
baseDir='./data/'

fnSampleInfo = '%s/%s'%( baseDir, 'sample_info.txt' )
fnReadCounts = '%s/%s'%( baseDir, 'read_counts_by_indiv_and_chrom.txt' )

fnSampleList = '%s/%s'%( baseDir, 'sample_names.txt' )

fileSampleInfo = open(fnSampleInfo,'r')

line = fileSampleInfo.readline()

line = line.rstrip()
line

line = line.split('\t')
line

sampleName = line[0]
print sampleName

sampleSex = line[4]
print sampleSex

# now up and rerun line = fileSampleInfo.readline()  (In[5])
# and rerun the readline and split commands
# what has changed?

# a routine to parse the sample table, line-by-line, into a 
# dictionary mapping [individual name]-->[sex]

def parseSampleTable( filename ):
    fileSampTable = open(filename,'r')
    
    # skip past the file header
    fileSampTable.readline()

    sampleToSex={}
    
    # go line by line
    for line in fileSampTable:
        line = line.rstrip().split('\t')
        sampleToSex[ line[0] ] = line[4]
    
    return sampleToSex



sampToSex = parseSampleTable( fnSampleInfo )

#what is the sex of individual HG00099?
sampToSex[ 'HG00099' ]

n_male = 0
n_female = 0
for sample in sampToSex:
    if sampToSex[sample] == 'male':
        n_male+=1
    elif sampToSex[sample] == 'female':
        n_female+=1

print 'found ',n_male, 'males'
print 'found ',n_female, 'females'





# e.g., let's look at one file
fnReadCounts = '%s/NA18627.stats.txt'%( baseDir )
filReadCounts = open(fnReadCounts,'r')
line = filReadCounts.readline()

line

line = line.rstrip().split('\t')
print line



lSampleNames = [ line.strip() for line in open(fnSampleList,'r') ]
print lSampleNames[:4]
print len(lSampleNames)



def loadAutoAndXCounts( directory, sampName ):
    filename = '%s/%s.stats.txt'%( directory, sampName )
    fileReadCounts = open(filename,'r')
    
    lChromNamesAutosomes = [ '%d'%i for i in range(1,23) ]
    
    nAutosomalReads = 0
    nXReads = 0
    
    for line in fileReadCounts:
        line = line.rstrip().split('\t')
        if line[0] in lChromNamesAutosomes:
            nAutosomalReads += int( line[2] )
        elif line[0] == 'X':
            nXReads += int( line[2] )
        # line goes chromosome, chromosome size, number of mapped reads, number of unmapped reads
        
    return nAutosomalReads, nXReads

loadAutoAndXCounts( baseDir, 'NA18627')

def loadPerChromCounts( directory, sampName ):
    filename = '%s/%s.stats.txt'%( directory, sampName )
    fileReadCounts = open(filename,'r')
    
    lChromosomes = []
    lReadCounts = [] 
    lChromSizes = []
        
    for line in fileReadCounts:
        line = line.rstrip().split('\t')
        lChromosomes.append( line[0] )
        lReadCounts.append( int(line[2]) )
        lChromSizes.append( int(line[1]) )
        
    return lChromosomes, lReadCounts, lChromSizes

lChromosomes,lReadCounts,lChromSizes = loadPerChromCounts( baseDir, 'NA18627')



plt.scatter( lReadCounts, lChromSizes )

plt.scatter( lReadCounts, lChromSizes )
for i in range(len(lReadCounts)):
    if lChromSizes[i]>1e7:
        plt.text( lReadCounts[i], lChromSizes[i], s=lChromosomes[i], color='red' )

lFracXMale=[]
lFracXFemale=[]
for sample in lSampleNames:
    countAuto,countX = loadAutoAndXCounts( baseDir,  sample )

    if sampToSex[sample]=='male':
        lFracXMale.append( countX / float(countAuto+countX) )
    elif sampToSex[sample]=='female':
        lFracXFemale.append( countX / float(countAuto+countX) )
    
    

lFracXMale[0:10]

lFracXFemale[0:10]

_ = plt.hist( [lFracXMale,lFracXFemale],  bins=100 )

_ = plt.hist( [lFracXMale,lFracXFemale],  bins=100, label=['male','female'], edgecolor='none' )
plt.legend()



