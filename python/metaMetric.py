import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.metricBundles as metricBundles
import operator

# Load up all the summary stats for each run
runNames = ['enigma_1189', 'ops2_1098']#, 'ops2_1093']
summaryDictList = []
for run in runNames:
    database = db.ResultsDb(run)
    summaryList = database.getSummaryStats()
    summaryDict = {}
    for sumDict in summaryList:
        summaryDict[sumDict['metricMetadata']+','+sumDict['metricName']+','+sumDict['summaryName']] = sumDict['summaryValue']
    summaryDictList.append(summaryDict)

summaryDictList[0]





class baseCompareMetric(object):
    def __init__(self, sumName=None, normVal=None, reverse=False, compareMethod=operator.div):
        """
        Set up a metric value for comparing between runs
        """
        self.normVal = normVal
        self.reverse = reverse
        self.compareMethod = compareMethod
        self.sumName = sumName
        
    def _compute(self, summaryDict):
        """
        Take the data and do some stuff
        """
        # By default, just grab the summary name
        if self.sumName is not None:
            result = summaryDict[self.sumName]
        else:
            result = 0
        
        return result
        
    def __call__(self, summaryDict, raw=False):
        """
        
        """
        result = self._compute(summaryDict)
        if self.reverse:
            result = 1./result
        if not raw:
            if self.normVal is not None:
                result = compareMethod(result,self.normVal)
        return result
        
    

seeing = baseCompareMetric(sumName='all band, all props,Mean finSeeing,Identity')
#seeing = baseCompareMetric()
for sumDict in summaryDictList:
    print seeing(sumDict)

class m5merge(baseCompareMetric):
    """
    
    """
    def _compute(self, summaryDict):
        filters = 'ugrizy'
        nominals=[26.1,27.4,27.5,26.8,26.1,24.9]
        result = 0
        for filterName,nominal in zip(filters,nominals):
            result += summaryDict['%s band, WFD,CoaddM5,Median' %filterName]-nominal
        return result

m5 = m5merge()
for sumDict in summaryDictList:
    print m5(sumDict)

def compareRuns(object):
    def __init__(self, runList, metaMetricList, dbfiles=None, path=''):
        if dbfiles is None:
            dbfiles = [os.path.join(path,run,'resultsdb.sqlite') for run in runList]
        self.resultArray = np.zeros((len(runList), len(metaMetricList)), dtype=float)
        # Loop over dbfiles and metaMetricList to fill in values for resultArray
        
    def plotCompareAll(self, **kwargs):
        """
        Label x-axis with metaMetric names, legend each line with runName
        """
    def tableCompareAll(self):
        """
        Print a table comparing all the metrics and runs
        """
    def head2headPlot(self, run1,run2):
        """
        Plot just 2 runs in a head-to-head comparision
        """





