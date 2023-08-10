from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode()
import warnings
warnings.filterwarnings("ignore")
import plotly.plotly as py
import plotly
import plotly.figure_factory as ff
import plotly.graph_objs as go
import pandas_datareader
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stat
get_ipython().magic('matplotlib inline')

from pandas_datareader.base import _BaseReader


class TSPReader(_BaseReader):

    """
    Returns DataFrame of historical TSP fund prices from symbols, over date
    range, start to end.
    Parameters
    ----------
    symbols : string, array-like object (list, tuple, Series), or DataFrame
        Single stock symbol (ticker), array-like object of symbols or
        DataFrame with index containing stock symbols.
    start : string, (defaults to '1/1/2010')
        Starting date, timestamp. Parses many different kind of date
        representations (e.g., 'JAN-01-2010', '1/1/10', 'Jan, 1, 1980')
    end : string, (defaults to today)
        Ending date, timestamp. Same format as starting date.
    retry_count : int, default 3
        Number of times to retry query request.
    pause : int, default 0
        Time, in seconds, to pause between consecutive queries of chunks. If
        single value given for symbol, represents the pause between retries.
    session : Session, default None
        requests.sessions.Session instance to be used
    """

    def __init__(self,
                 symbols=['Linc', 'L2020', 'L2030', 'L2040', 'L2050', 'G', 'F', 'C', 'S', 'I'],
                 start=None, end=None, retry_count=3, pause=0.001,
                 session=None):
        super(TSPReader, self).__init__(symbols=symbols,
                                        start=start, end=end,
                                        retry_count=retry_count,
                                        pause=pause, session=session)
        self._format = 'string'

    @property
    def url(self):
        return 'https://www.tsp.gov/InvestmentFunds/FundPerformance/index.html'

    def read(self):
        """ read one data from specified URL """
        df = super(TSPReader, self).read()
        df.columns = map(lambda x: x.strip(), df.columns)
        return df

    @property
    def params(self):
        return {'startdate': self.start.strftime('%m/%d/%Y'),
                'enddate': self.end.strftime('%m/%d/%Y'),
                'fundgroup': self.symbols,
                'whichButton': 'CSV'}

    @staticmethod
    def _sanitize_response(response):
        """
        Clean up the response string
        """
        text = response.text.strip()
        if text[-1] == ',':
            return text[0:-1]
        return text

tspreader = TSPReader(start='2000-06-01', end='2017-03-01')
tspdata = tspreader.read()
tspdata

tspdata = tspdata.drop('L Income', 1)
tspdata = tspdata.drop('L 2020', 1)
tspdata = tspdata.drop('L 2030', 1)
tspdata = tspdata.drop('L 2040', 1)
tspdata = tspdata.drop('L 2050', 1)
tspdata

tspdataresampled = tspdata.resample('BMS').first()

monthlyreturns = pd.DataFrame(index=tspdataresampled.index, columns=['G Fund', 'F Fund', 'S Fund', 'C Fund', 'I Fund'])

for i in range(1,len(tspdataresampled)):
    monthlyreturns['G Fund'][i] = (float(tspdataresampled['G Fund'][i]) - float(tspdataresampled['G Fund'][i-1]))/float(tspdataresampled['G Fund'][i])
    monthlyreturns['F Fund'][i] = (float(tspdataresampled['F Fund'][i]) - float(tspdataresampled['F Fund'][i-1]))/float(tspdataresampled['F Fund'][i])
    monthlyreturns['S Fund'][i] = (float(tspdataresampled['S Fund'][i]) - float(tspdataresampled['S Fund'][i-1]))/float(tspdataresampled['S Fund'][i])
    monthlyreturns['C Fund'][i] = (float(tspdataresampled['C Fund'][i]) - float(tspdataresampled['C Fund'][i-1]))/float(tspdataresampled['C Fund'][i])
    monthlyreturns['I Fund'][i] = (float(tspdataresampled['I Fund'][i]) - float(tspdataresampled['I Fund'][i-1]))/float(tspdataresampled['I Fund'][i])

plt.figure(figsize=(12, 9))
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.xlabel("Monthly Returns", fontsize=16)
plt.ylabel("Count", fontsize=16)

plt.hist(monthlyreturns['G Fund'],color="#3F5D7D")
plt.text(0.0027, 35, "Percentage Monthly Returns Distribution for G Fund", fontsize=17, ha="center")
plt.text(0.00105, -4, "Data source: Thrift Savings Plan Fund Data (https://www.tsp.gov/InvestmentFunds/FundPerformance/index.html)", fontsize=10)
plt.plot()

plt.figure(figsize=(12, 9))
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.xlabel("Monthly Returns", fontsize=16)
plt.ylabel("Count", fontsize=16)

plt.hist(monthlyreturns['F Fund'],color="#3F5D7D")
plt.text(0.00, 52, "Percentage Monthly Returns Distribution for F Fund", fontsize=17, ha="center")
plt.text(-0.02, -5, "Data source: Thrift Savings Plan Fund Data (https://www.tsp.gov/InvestmentFunds/FundPerformance/index.html)", fontsize=10)
plt.plot()

plt.figure(figsize=(12, 9))
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.xlabel("Monthly Returns", fontsize=16)
plt.ylabel("Count", fontsize=16)

plt.hist(monthlyreturns['S Fund'],color="#3F5D7D")
plt.text(-0.05, 69, "Percentage Monthly Returns Distribution for S Fund", fontsize=17, ha="center")
plt.text(-0.25, -7, "Data source: Thrift Savings Plan Fund Data (https://www.tsp.gov/InvestmentFunds/FundPerformance/index.html)", fontsize=10)
plt.plot()

plt.figure(figsize=(12, 9))
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.xlabel("Monthly Returns", fontsize=16)
plt.ylabel("Count", fontsize=16)

plt.hist(monthlyreturns['C Fund'],color="#3F5D7D")
plt.text(-0.05, 75, "Percentage Monthly Returns Distribution for C Fund", fontsize=17, ha="center")
plt.text(-0.19, -7, "Data source: Thrift Savings Plan Fund Data (https://www.tsp.gov/InvestmentFunds/FundPerformance/index.html)", fontsize=10)
plt.plot()

plt.figure(figsize=(12, 9))
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.xlabel("Monthly Returns", fontsize=16)
plt.ylabel("Count", fontsize=16)

plt.hist(monthlyreturns['I Fund'],color="#3F5D7D")
plt.text(-0.05, 75, "Percentage Monthly Returns Distribution for I Fund", fontsize=17, ha="center")
plt.text(-0.25, -7, "Data source: Thrift Savings Plan Fund Data (https://www.tsp.gov/InvestmentFunds/FundPerformance/index.html)", fontsize=10)
plt.plot()

stackedtable1 = pd.DataFrame(np.zeros((len(monthlyreturns),7)), columns=['Year','Month', 'G Fund', 'F Fund', 'S Fund', 'C Fund', 'I Fund'])
for i in range(0,len(monthlyreturns)):
    stackedtable1['Year'][i] = str(monthlyreturns.index.year[i])
    stackedtable1['Month'][i] = str(monthlyreturns.index.month[i])
    stackedtable1['G Fund'][i] = float(monthlyreturns['G Fund'][i])
    stackedtable1['F Fund'][i] = float(monthlyreturns['F Fund'][i])
    stackedtable1['S Fund'][i] = float(monthlyreturns['S Fund'][i])
    stackedtable1['C Fund'][i] = float(monthlyreturns['C Fund'][i])
    stackedtable1['I Fund'][i] = float(monthlyreturns['I Fund'][i])

stackedTableN = pd.pivot_table(stackedtable1, index=["Month"], values=["G Fund", "F Fund", "S Fund", "C Fund", "I Fund"], aggfunc=[np.mean])

table = ff.create_table(stackedTableN, index=True, index_title="Month")
table.layout.width=1200
iplot(table, filename='PivotTableMonth')

GFundMean = [0,0,0,0,0,0,0,0,0,0,0,0]
FFundMean = [0,0,0,0,0,0,0,0,0,0,0,0]
SFundMean = [0,0,0,0,0,0,0,0,0,0,0,0]
CFundMean = [0,0,0,0,0,0,0,0,0,0,0,0]
IFundMean = [0,0,0,0,0,0,0,0,0,0,0,0]

for i in range(0,12):
    GFundMean[i] = stackedTableN.as_matrix()[i][2]
    FFundMean[i] = stackedTableN.as_matrix()[i][1]
    SFundMean[i] = stackedTableN.as_matrix()[i][4]
    CFundMean[i] = stackedTableN.as_matrix()[i][0]
    IFundMean[i] = stackedTableN.as_matrix()[i][3]

trace1 = go.Bar(
    x = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    y=GFundMean,
    name='G Fund',
    marker=dict(
        color='rgb(31, 119, 180)'
    )
)
trace2 = go.Bar(
    x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    y=FFundMean,
    name='F Fund',
    marker=dict(
        color='rgb(44, 160, 44)'
    )
)
trace3 = go.Bar(
    x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    y=SFundMean,
    name='S Fund',
    marker=dict(
        color='rgb(148, 103, 189)'
    )
)
trace4 = go.Bar(
    x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    y=CFundMean,
    name='C Fund',
    marker=dict(
        color='rgb(227, 119, 194)'
    )
)
trace5 = go.Bar(
    x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    y=IFundMean,
    name='I Fund',
    marker=dict(
        color='rgb(188, 189, 34)'
    )
)
data = [trace1, trace2, trace3, trace4, trace5]
layout = go.Layout(
    title='Mean Monthly Returns For Funds',
    xaxis=dict(
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Monthly Returns (mean)',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    legend=dict(
        x=0.9,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='Mean-Monthly-Returns-All-Funds')

fundMeanDataFrame = pd.DataFrame(index=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], columns=['G Fund', 'F Fund', 'S Fund', 'C Fund', 'I Fund'])
i=0
for month in fundMeanDataFrame.index.values:
    fundMeanDataFrame['G Fund'][month] = GFundMean[i]
    fundMeanDataFrame['F Fund'][month] = FFundMean[i]
    fundMeanDataFrame['S Fund'][month] = SFundMean[i]
    fundMeanDataFrame['C Fund'][month] = CFundMean[i]
    fundMeanDataFrame['I Fund'][month] = IFundMean[i]
    i = i+1
    if(i>11): break

totalannual = 1200
assetallocper = pd.DataFrame(np.zeros((1,5)), columns=['G Fund', 'F Fund', 'S Fund', 'C Fund', 'I Fund'])
assetallocper['G Fund'][0] = 0.2
assetallocper['F Fund'][0] = 0.2
assetallocper['S Fund'][0] = 0.2
assetallocper['C Fund'][0] = 0.2
assetallocper['I Fund'][0] = 0.2

#Implementing the above using the following function
allocoutp = pd.DataFrame(index=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], columns=['G Fund', 'F Fund', 'S Fund', 'C Fund', 'I Fund'])
def monthcontri (totalannual, assetallocper, bestMonth, stratnum, strattype, strat3dict):
    for fund in allocoutp.columns.values:
        for month in allocoutp.index.values:
            allocoutp[fund][month] = 0
    if (strattype=='cont'):
        equalmonthalloc = totalannual/12
        for fund in allocoutp.columns.values:
            for month in allocoutp.index.values:
                allocoutp[fund][month] =(assetallocper[fund][0] * equalmonthalloc)


    elif (strattype=='seas'):
        if (stratnum==1):
            for fund in allocoutp.columns.values:
                allocoutp[fund][bestMonth['firstBest'][fund]] =(assetallocper[fund][0] * totalannual)
        elif (stratnum==2):
            dict = {'G Fund':0, 'F Fund':0, 'S Fund':0, 'C Fund':0, 'I Fund':0}
            assignedMonthNo = {'G Fund':0, 'F Fund':0, 'S Fund':0, 'C Fund':0, 'I Fund':0}
            zerolist = ['G Fund','F Fund', 'S Fund', 'C Fund', 'I Fund']
            loop = True
            while(loop):
                for fund in zerolist:
                    if (assignedMonthNo[fund]==0):
                        for bm in bestMonth.columns.values:
                            if bestMonth[bm][fund] not in dict.values():
                                dict[fund] = bestMonth[bm][fund]
                                assignedMonthNo[fund] = bm
                                break
                            elif bestMonth[bm][fund] in dict.values():
                                if not zerolist:
                                    break
                                for key, value in dict.items():
                                    if value == bestMonth[bm][fund]:
                                        preconflictfund = key
                                preconflictmt = assignedMonthNo[preconflictfund]
                                if bestMonth[bm + 'V'][fund]>bestMonth[preconflictmt + 'V'][preconflictfund]:
                                    dict[fund] = bestMonth[bm][fund]
                                    assignedMonthNo[fund] = bm
                                    dict[preconflictfund]=0
                                    break
                    elif (assignedMonthNo[fund]!=0):
                        for bm in bestMonth.columns.values[bestMonth.columns.values.tolist().index(assignedMonthNo[fund]):]:
                            if bestMonth[bm][fund] not in dict.values():
                                dict[fund] = bestMonth[bm][fund]
                                assignedMonthNo[fund] = bm
                                break
                            elif bestMonth[bm][fund] in dict.values():
                                if not zerolist:
                                    break
                                for key, value in dict.items():
                                    if value == bestMonth[bm][fund]:
                                        preconflictfund = key
                                preconflictmt = assignedMonthNo[preconflictfund]
                                if bestMonth[bm + 'V'][fund]>bestMonth[preconflictmt + 'V'][preconflictfund]:
                                    dict[fund] = bestMonth[bm][fund]
                                    assignedMonthNo[fund] = bm
                                    dict[preconflictfund]=0
                                    break
                del zerolist[:]
                loop = False
                for key in dict.keys():
                    if dict[key]==0:
                        zerolist.append(key)
                        loop = True
            for key in dict.keys():
                allocoutp[key][dict[key]] =(assetallocper[key][0]*totalannual)


        elif (stratnum==3):
            for key in strat3dict.keys():
                allocoutp[strat3dict[key]][key] =(assetallocper[strat3dict[key]][0]*totalannual)


    return allocoutp

#Finding the best month to invest in each fund based on the corresponding monthly returns
GFundMeanSort = fundMeanDataFrame.sort_values(by='G Fund', ascending=False)
FFundMeanSort = fundMeanDataFrame.sort_values(by='F Fund', ascending=False)
SFundMeanSort = fundMeanDataFrame.sort_values(by='S Fund', ascending=False)
CFundMeanSort = fundMeanDataFrame.sort_values(by='C Fund', ascending=False)
IFundMeanSort = fundMeanDataFrame.sort_values(by='I Fund', ascending=False)



bestMonth = pd.DataFrame(index=['G Fund', 'F Fund', 'S Fund', 'C Fund', 'I Fund'], columns=['firstBest','secondBest','thirdBest','fourthBest','fifthBest','firstBestV','secondBestV','thirdBestV','fourthBestV','fifthBestV'])
i = 0
for bestno in bestMonth.columns.values:
        bestMonth[bestno]['G Fund'] = GFundMeanSort.index.values[i]
        bestMonth[bestno]['F Fund'] = FFundMeanSort.index.values[i]
        bestMonth[bestno]['S Fund'] = SFundMeanSort.index.values[i]
        bestMonth[bestno]['C Fund'] = CFundMeanSort.index.values[i]
        bestMonth[bestno]['I Fund'] = IFundMeanSort.index.values[i]
        i = i+1
        if(i>4): break

i = 0
for bestv in bestMonth.columns.values[5:10]:
        bestMonth[bestv]['G Fund'] = GFundMeanSort['G Fund'][GFundMeanSort.index.values[i]]
        bestMonth[bestv]['F Fund'] = FFundMeanSort['F Fund'][FFundMeanSort.index.values[i]]
        bestMonth[bestv]['S Fund'] = SFundMeanSort['S Fund'][SFundMeanSort.index.values[i]]
        bestMonth[bestv]['C Fund'] = CFundMeanSort['C Fund'][CFundMeanSort.index.values[i]]
        bestMonth[bestv]['I Fund'] = IFundMeanSort['I Fund'][IFundMeanSort.index.values[i]]
        i = i+1
        if(i>4): break

#Select Fund With Max Avg Returns For Each Month
strat3dict = {'Jan':0,'Feb':0, 'Mar':0, 'Apr':0, 'May':0, 'Jun':0, 'Jul':0, 'Aug':0, 'Sep':0, 'Oct':0, 'Nov':0, 'Dec':0}
for month in fundMeanDataFrame.index.values:
    max = fundMeanDataFrame['G Fund'][month]
    maxfund = 'G Fund'
    for fund in fundMeanDataFrame.columns.values:
        if fundMeanDataFrame[fund][month] > max:
            max = fundMeanDataFrame[fund][month]
            maxfund = fund
    strat3dict[month] = maxfund
#Calculating Number of Months Assigned To Each Fund
count = {'G Fund':0,'F Fund':0,'S Fund':0,'C Fund':0,'I Fund':0}
for month in strat3dict.keys():
    count[strat3dict[month]] = count[strat3dict[month]] + 1

# Dividing Each Funds Original Asset Allocation Between The Months Selected For That Fund
assetallocperStrat3 = pd.DataFrame(np.zeros((1,5)), columns=['G Fund', 'F Fund', 'S Fund', 'C Fund', 'I Fund'])
for month in strat3dict.keys():
    if count[strat3dict[month]]!=0:
        assetallocperStrat3[strat3dict[month]][0] = assetallocper[fund]/count[strat3dict[month]]

#Running The Investment Strategies Function
ContinuousStrategy = (monthcontri(totalannual,assetallocper,bestMonth,0,'cont',{}))
#Results For Continuous Strategy
table = ff.create_table(ContinuousStrategy, index=True, index_title="Month")
table.layout.width=600
table.layout.height = 650
iplot(table)

#Running The Investment Strategies Function
SeasonalStrategyOne = (monthcontri(totalannual,assetallocper,bestMonth,1,'seas',{}))
#Results For Seasonal Strategy One
table = ff.create_table(SeasonalStrategyOne, index=True, index_title="Month")
table.layout.width=600
table.layout.height = 650
iplot(table)

#Running The Investment Strategies Function
SeasonalStrategyTwo = (monthcontri(totalannual,assetallocper,bestMonth,2,'seas',{}))
#Results For Seasonal Strategy Two
table = ff.create_table(SeasonalStrategyTwo, index=True, index_title="Month")
table.layout.width=600
table.layout.height = 650
iplot(table)

#Running The Investment Strategies Function
SeasonalStrategyThree = (monthcontri(totalannual,assetallocperStrat3,bestMonth,3,'seas',strat3dict))
#Results For Seasonal Strategy Three
table = ff.create_table(SeasonalStrategyThree, index=True, index_title="Month")
table.layout.width=600
table.layout.height = 650
iplot(table)

#Developing A Function To Implement The Above
def actSimul(initialContri, StrategyTypeDFGenerated, fundMeanDataFrame, n, k):

    # Initializing DataFrame To Store Savings Associated With Each Fund And Total Savings For Each Iteration
    randIntSimulDF = pd.DataFrame(np.zeros((k, 6)),columns=['G Fund', 'F Fund', 'S Fund', 'C Fund', 'I Fund', 'Total'])
    randIntSimulDF['G Fund'] = randIntSimulDF['G Fund'].astype(object)
    randIntSimulDF['F Fund'] = randIntSimulDF['F Fund'].astype(object)
    randIntSimulDF['S Fund'] = randIntSimulDF['S Fund'].astype(object)
    randIntSimulDF['C Fund'] = randIntSimulDF['C Fund'].astype(object)
    randIntSimulDF['I Fund'] = randIntSimulDF['I Fund'].astype(object)
    randIntSimulDF['Total'] = randIntSimulDF['Total'].astype(object)

    #Initializing Lists To Store End Savings For Each Fund And Total End Savings
    GFundEndSavings = [0]*k
    FFundEndSavings = [0]*k
    SFundEndSavings = [0]*k
    CFundEndSavings = [0]*k
    IFundEndSavings = [0]*k
    TotalEndSavings = [0]*k

    #Outer For Loop Looping Through Many Replications To Get A Distribution
    for sb in range(0,k):
        
        # Randomly Selecting A Number From 0 to n where 'n' is the number of months considered for simulation
        randMonthInt = np.random.randint(0, n, 1)
        # Selecting A Month Based On The Above Randomly Selected Number
        randMonth = StrategyTypeDFGenerated.index.values[(randMonthInt[0])%12]

        # Assigning The Interest For Each Fund Based On It's Returns For The Above Randomly Selected Month
        interest = {'G Fund': fundMeanDataFrame['G Fund'][randMonth], 'F Fund': fundMeanDataFrame['F Fund'][randMonth],'S Fund': fundMeanDataFrame['S Fund'][randMonth], 'C Fund': fundMeanDataFrame['C Fund'][randMonth],'I Fund': fundMeanDataFrame['I Fund'][randMonth]}

        # To Store All Information, Initializing Arrays Of Length n+1 where 'n' is the number of months
        GFundArray = np.array([],dtype=np.float64)
        FFundArray = np.array([],dtype=np.float64)
        SFundArray = np.array([],dtype=np.float64)
        CFundArray = np.array([],dtype=np.float64)
        IFundArray = np.array([],dtype=np.float64)
        totalArray = np.array([],dtype=np.float64)

        for i in range(0,n+1):
            GFundArray = np.append(GFundArray, 0)
            FFundArray = np.append(FFundArray, 0)
            SFundArray = np.append(SFundArray, 0)
            CFundArray = np.append(CFundArray, 0)
            IFundArray = np.append(IFundArray, 0)
            totalArray = np.append(totalArray, 0)

        # The initial contributions will go in index 0
        GFundArray[0]=initialContri['G Fund']
        FFundArray[0]=initialContri['F Fund']
        SFundArray[0]=initialContri['S Fund']
        CFundArray[0]=initialContri['C Fund']
        IFundArray[0]=initialContri['I Fund']

        #I will loop through each month and apply the interest (Inner For Loop).
        for i in range(1, n+1):
            if i>11:
                j = (i-1)%12
            else:
                j = i-1
            GFundArray[i] = ((1+interest['G Fund'])*GFundArray[i-1])+(StrategyTypeDFGenerated['G Fund'][StrategyTypeDFGenerated.index.values[j]])
            FFundArray[i] = ((1+interest['F Fund'])*FFundArray[i-1])+(StrategyTypeDFGenerated['F Fund'][StrategyTypeDFGenerated.index.values[j]])
            SFundArray[i] = ((1+interest['S Fund'])*SFundArray[i-1])+(StrategyTypeDFGenerated['S Fund'][StrategyTypeDFGenerated.index.values[j]])
            CFundArray[i] = ((1+interest['C Fund'])*CFundArray[i-1])+(StrategyTypeDFGenerated['C Fund'][StrategyTypeDFGenerated.index.values[j]])
            IFundArray[i] = ((1+interest['I Fund'])*IFundArray[i-1])+(StrategyTypeDFGenerated['I Fund'][StrategyTypeDFGenerated.index.values[j]])
            totalArray[i] = GFundArray[i] + FFundArray[i] + SFundArray[i] + CFundArray[i] + IFundArray[i]


        #Storing Savings Associated With Each Fund And Total Savings For Each Iteration
        randIntSimulDF['G Fund'][sb] = GFundArray.tolist()
        randIntSimulDF['F Fund'][sb] = FFundArray.tolist()
        randIntSimulDF['S Fund'][sb] = SFundArray.tolist()
        randIntSimulDF['C Fund'][sb] = CFundArray.tolist()
        randIntSimulDF['I Fund'][sb] = IFundArray.tolist()
        randIntSimulDF['Total'][sb] = totalArray.tolist()

        #Storing End Savings For Each Fund And Total End Savings
        GFundEndSavings[sb] = GFundArray[n]
        FFundEndSavings[sb] = FFundArray[n]
        SFundEndSavings[sb] = SFundArray[n]
        CFundEndSavings[sb] = CFundArray[n]
        IFundEndSavings[sb] = IFundArray[n]
        TotalEndSavings[sb] = totalArray[n]


    return TotalEndSavings

#Setting Up The Initial Contribution In Each Fund By Taking User Input
GFundInitialContri = input('Enter your initial contribution in G Fund: ')
FFundInitialContri = input('Enter your initial contribution in F Fund: ')
SFundInitialContri = input('Enter your initial contribution in S Fund: ')
CFundInitialContri = input('Enter your initial contribution in C Fund: ')
IFundInitialContri = input('Enter your initial contribution in I Fund: ')

#Storing Above Values In A Dictionary
initialContri = {'G Fund': GFundInitialContri, 'F Fund': FFundInitialContri, 'S Fund': SFundInitialContri, 'C Fund': CFundInitialContri, 'I Fund': IFundInitialContri}

# Running The Function For Seasonal Strategy One
x1 = actSimul(initialContri, SeasonalStrategyOne, fundMeanDataFrame, 120, 10000)

# Plotting The Distribution For Seasonal Strategy One
trace1 = go.Histogram(
    x=x1,
    opacity=0.75
)

data = [trace1]

layout = go.Layout(
    title='Total End Savings Seasonal Strategy One',
    xaxis=dict(
        title='Total End Savings'
    ),
    yaxis=dict(
        title='Count'
    ),
    bargap=0.1,
    bargroupgap=0.1
)


fig = go.Figure(data=data, layout=layout)

iplot(fig)

# Running The Function For Seasonal Strategy Two
x2 = actSimul(initialContri, SeasonalStrategyTwo, fundMeanDataFrame, 120, 10000)

# Plotting The Distribution For Seasonal Strategy Two
trace1 = go.Histogram(
    x=x2,
    opacity=0.75
)

data = [trace1]

layout = go.Layout(
    title='Total End Savings Seasonal Strategy Two',
    xaxis=dict(
        title='Total End Savings'
    ),
    yaxis=dict(
        title='Count'
    ),
    bargap=0.1,
    bargroupgap=0.1
)


fig = go.Figure(data=data, layout=layout)

iplot(fig)

# Running The Function For Seasonal Strategy Three
x3 = actSimul(initialContri, SeasonalStrategyThree, fundMeanDataFrame, 120, 10000)

# Plotting The Distribution For Seasonal Strategy Three
trace1 = go.Histogram(
    x=x3,
    opacity=0.75
)

data = [trace1]

layout = go.Layout(
    title='Total End Savings Seasonal Strategy Three',
    xaxis=dict(
        title='Total End Savings'
    ),
    yaxis=dict(
        title='Count'
    ),
    bargap=0.2,
    bargroupgap=0.1
)


fig = go.Figure(data=data, layout=layout)

iplot(fig)

# Running The Function For Continuous Strategy
xc = actSimul(initialContri, ContinuousStrategy, fundMeanDataFrame, 120, 10000)

# Plotting The Distribution For Continuous Strategy
trace1 = go.Histogram(
    x=xc,
    opacity=0.75
)

data = [trace1]

layout = go.Layout(
    title='Total End Savings Continuous Strategy',
    xaxis=dict(
        title='Total End Savings'
    ),
    yaxis=dict(
        title='Count'
    ),
    bargap=0.2,
    bargroupgap=0.1
)


fig = go.Figure(data=data, layout=layout)

iplot(fig)

#Reporting Some Basic Summary Statistics For Total End Savings(e.g., mean, standard deviation, min, max).
endSavingsSummaryStats = pd.DataFrame(index=['Mean','Standard Deviation','Min','Max'], columns=['Seasonal Strategy One', 'Seasonal Strategy Two', 'Seasonal Strategy Three', 'Continuous Strategy'])

endSavingsSummaryStats['Seasonal Strategy One']['Mean'] = stat.mean(x1)
endSavingsSummaryStats['Seasonal Strategy One']['Standard Deviation'] = stat.stdev(x1)
endSavingsSummaryStats['Seasonal Strategy One']['Min'] = np.asarray(x1).min()
endSavingsSummaryStats['Seasonal Strategy One']['Max'] = np.asarray(x1).max()

endSavingsSummaryStats['Seasonal Strategy Two']['Mean'] = stat.mean(x2)
endSavingsSummaryStats['Seasonal Strategy Two']['Standard Deviation'] = stat.stdev(x2)
endSavingsSummaryStats['Seasonal Strategy Two']['Min'] = np.asarray(x2).min()
endSavingsSummaryStats['Seasonal Strategy Two']['Max'] = np.asarray(x2).max()

endSavingsSummaryStats['Seasonal Strategy Three']['Mean'] = stat.mean(x3)
endSavingsSummaryStats['Seasonal Strategy Three']['Standard Deviation'] = stat.stdev(x3)
endSavingsSummaryStats['Seasonal Strategy Three']['Min'] = np.asarray(x3).min()
endSavingsSummaryStats['Seasonal Strategy Three']['Max'] = np.asarray(x3).max()

endSavingsSummaryStats['Continuous Strategy']['Mean'] = stat.mean(xc)
endSavingsSummaryStats['Continuous Strategy']['Standard Deviation'] = stat.stdev(xc)
endSavingsSummaryStats['Continuous Strategy']['Min'] = np.asarray(xc).min()
endSavingsSummaryStats['Continuous Strategy']['Max'] = np.asarray(xc).max()

table = ff.create_table(endSavingsSummaryStats, index=True, index_title="Summary Statistic")
table.layout.width=1000
table.layout.height = 400
iplot(table)



