# Get our notebook ready for zipline 
get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext zipline')

import pandas as pd
import talib
from zipline.api import symbol
import alphalens

# Create global variables to feed alphalens
dfPrice=pd.DataFrame()
seSig=pd.Series()

# Zipline algo

def initialize(context):
    context.iNDays=400   # How many days of data we want
    
    context.iADOFast=5   # talib AD Osc constant
    context.iADOSlow=14  # talib AD Osc constant
    
    # DJI 30
    context.secs=[]
    context.secs.append(symbol("AAPL")) # Apple
    context.secs.append(symbol("AXP")) # American Express
    context.secs.append(symbol("BA")) # Boeing
    context.secs.append(symbol("CAT")) # Caterpillar
    context.secs.append(symbol("CSCO")) # Cisco
    context.secs.append(symbol("CVX")) # Chevron
    context.secs.append(symbol("DD")) # E I du Pont de Nemours and Co
    context.secs.append(symbol("DIS")) # Disney
    context.secs.append(symbol("GE")) # General Electric
    context.secs.append(symbol("GS")) # Goldman Sachs
    context.secs.append(symbol("HD")) # Home Depot
    context.secs.append(symbol("IBM")) # IBM
    context.secs.append(symbol("INTC")) # Intel
    context.secs.append(symbol("JNJ")) # Johnson & Johnson
    context.secs.append(symbol("JPM")) # JPMorgan Chase
    context.secs.append(symbol("KO")) # Coca-Cola
    context.secs.append(symbol("MCD")) # McDonald's
    context.secs.append(symbol("MMM")) # 3M
    context.secs.append(symbol("MRK")) # Merck
    context.secs.append(symbol("MSFT")) # Microsoft
    context.secs.append(symbol("NKE")) # Nike
    context.secs.append(symbol("PFE")) # Pfizer
    context.secs.append(symbol("PG")) # Procter & Gamble
    context.secs.append(symbol("TRV")) # Travelers Companies Inc
    context.secs.append(symbol("UNH")) # UnitedHealth
    context.secs.append(symbol("UTX")) # United Technologies
    context.secs.append(symbol("V")) # Visa
    context.secs.append(symbol("VZ")) # Verizon
    context.secs.append(symbol("WMT")) # Wal-Mart
    context.secs.append(symbol("XOM")) # Exxon Mobil

def handle_data(context, data):
    global seSig   
    liSeries=[]  # Used to collect the series as we go

    # Get data
    dfP=data.history(context.secs,'price',context.iNDays,'1d')
    dfL=data.history(context.secs,'low',context.iNDays,'1d')
    dfH=data.history(context.secs,'high',context.iNDays,'1d')
    dfV=data.history(context.secs,'volume',context.iNDays,'1d')

    ixP=dfP.index  # This is the date 

    for S in context.secs:
        # Save our history for alphalens
        dfPrice[S.symbol]=dfP[S]
        
        # Normalize for tablib
        seP=dfP[S]/dfP[S].mean()
        seL=dfL[S]/dfL[S].mean()
        seH=dfH[S]/dfH[S].mean()
        seV=dfV[S]/dfV[S].mean()
        
        # Get our ta-value
        ndADosc=talib.ADOSC(             seP.values,seL.values,seH.values,seV.values,             context.iADOFast,context.iADOSlow)

        # alphalens requires that the Series used for the Signal 
        # have a MultiIndex consisting of date+symbol

        # Build a list of symbol names same length as our price data
        liW=[S.symbol]*len(ixP)
        # Make a tuple
        tuW=zip(ixP,liW)
        # Create the required MultiIndex
        miW=pd.MultiIndex.from_tuples(tuW,names=['date','sym'])
        # Create series
        seW=pd.Series(ndADosc,index=miW)
        # Save it for later
        liSeries.append(seW)

    # Now make the required series
    seSig=pd.concat(liSeries).dropna()

    return

# We only need to run zipline for one day.... not the whole period
# now run run zipline for last day in period of interest 
get_ipython().magic('zipline --start=2016-8-31 --end=2016-8-31  --capital-base=100000')

# Lets take a look at what got built
print type(dfPrice),"length=",len(dfPrice)
print dfPrice.head(3)
print dfPrice.tail(3)

print type(seSig),"length=",len(seSig)
print seSig.head(3)
print seSig.tail(3)
# Make sure out MultiIndex is date+symbol
print seSig.index[0]

alphalens.tears.create_factor_tear_sheet(         factor=seSig,         prices=dfPrice,periods=(1,2,3))







