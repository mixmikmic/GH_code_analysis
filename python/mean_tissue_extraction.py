get_ipython().magic('matplotlib inline')
# from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib as mpl # this module controls the default values for plotting in matplotlib

def get_sample_annotation(filename):
    ann_file = open(filename)
    
    ann_file.close()

def read_transcript_means_variances(filename,header,logscale):
    # compute and store the mean expression levels for each targetID
    meanExpressionLevels = [] 
    varExpressionLevels = [] 
    rpkm_file = open(filename)
    firstLine = True
    for line in rpkm_file:
        if header == True:
            if firstLine:
                firstLine = False
                continue
        if header == True:
            expLevels = np.array(line.split('\t')[4:]).astype(np.float)
        else:
            expLevels = [float(i)  for i in line.split('\t')]  
        if logscale == True:
            expLevels = [np.log10(x + 1) for x in expLevels]
        meanExpressionLevels.append(sum(expLevels))
        varExpressionLevels.append(np.var(expLevels))
    rpkm_file.close()

    return [meanExpressionLevels,varExpressionLevels]

def summarize_expLevels(values):
    series = pd.Series(values)
    print series.describe()

def plot_histogram(ax,values,xLabel):
    # plotting
    n, bins, patches = ax.hist(values, 50, normed=1, facecolor='green', alpha=0.75)
    ax.set_xlabel(xLabel)
    ax.set_ylabel('fraction of transcripts')
    # plt.title(filename)
    # plt.axis([0, 1e+06, 0, 20])
    # ax.grid(True)

def plot_multiple(values1,labels1,values2,labels2):
    fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2)
    plot_histogram(ax1,values1,labels1)
    plot_histogram(ax2,values2,labels2)

def write_array_to_file(array,filename):
    outfile = open(filename,"w")
    for item in array:
        outfile.write("%f\n" % item)
    outfile.close()
    
def read_array_from_file(filename):
    array = []
    infile = open(filename,"r")
    for line in infile:
        array.append(float(line))
    infile.close()
    return array

filename = '../data/tissue_test.txt' 
[mean,variance] = read_transcript_means_variances(filename,header=False,logscale=True)

summarize_expLevels(mean)

#summarize_expLevels(variance,'variance of log expression')

pfx = '../../local_data/transcript_rpkm_in_go'
[mean,variance] = read_transcript_means_variances(pfx + '.txt' ,header=True,logscale=True)
write_array_to_file(mean, pfx + '_mean.txt' )
write_array_to_file(variance, pfx + '_variance.txt' )

meanValues = read_array_from_file('../../local_data/transcript_rpkm_in_go_mean.txt')
varianceValues = read_array_from_file('../../local_data/transcript_rpkm_in_go_variance.txt')
mpl.rcParams['figure.figsize'] = (13, 6)
plot_multiple(meanValues,'mean of log experssion',varianceValues,'variance of log experssion')
summarize_expLevels(meanValues)

numberOfZeroMeanTranscripts = 0
for value in meanValues:
    if value <= 0.0001:
        numberOfZeroMeanTranscripts = numberOfZeroMeanTranscripts + 1
print 'Total Number of Transcripts:         ' , len(meanValues)
print 'Number of Zero Mean Transcripts:     ' , numberOfZeroMeanTranscripts

numberOfZeroVarTranscripts = 0
for value in varianceValues:
    if value <= 1e-9:
        numberOfZeroVarTranscripts = numberOfZeroVarTranscripts + 1

print 'Number of LOW Variance Transcripts:  ', numberOfZeroVarTranscripts

pfx = '../../local_data/transcript_rpkm_top_10000_var'

[mean,variance] = read_transcript_means_variances(pfx + '.txt' ,header=True,logscale=True)
write_array_to_file(mean, pfx + '_mean.txt' )
write_array_to_file(variance, pfx + '_variance.txt' )

meanValues = read_array_from_file( pfx + '_mean.txt' )
varianceValues = read_array_from_file(pfx + '_variance.txt')
mpl.rcParams['figure.figsize'] = (13, 6)
plot_multiple(meanValues,'mean of log experssion',varianceValues,'variance of log experssion')
summarize_expLevels(meanValues)



