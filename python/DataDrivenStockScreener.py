get_ipython().system('pip install quandl')
get_ipython().system('pip install tabulate')

# Optional for backtesting with Quantopian
# !pip install zipline

# Import the Quandl API and configure the API key
import quandl
quandl.ApiConfig.api_key = "icjEB8FYLh6QyycLs6Xf"

import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# We will use the Quandl service to get the data for six months for AAPL
start = pd.to_datetime('2017-04-01')
end = pd.to_datetime('2017-10-01')
stock_price_df = quandl.get('WIKI/AAPL', start_date=start, end_date=end)

# Print some general information about the results from Quandl
stock_price_df.info()

# Print the last 10 items to see what we got
print tabulate(stock_price_df[['Open', 'Close', 'High', 'Low', 'Volume']].tail(10), headers='keys', tablefmt='psql')

# Sometimes the API may not work, then we'll need work with CSV files
# stock_price_df = pd.read_csv('WIKI_PRICES_AAPL.csv', sep=',')
# stock_price_df.index = pd.to_datetime(stock_price_df.pop('date'))
# stock_price_df.info()

# Create a plot function
get_ipython().run_line_magic('matplotlib', 'inline')
def plot_series(df, columns, last_n, title):
    plot_df = df[columns].tail(last_n)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    plot_df.plot(ax=ax, figsize=(20,8))

# Moving Average
def MA(df, n):
    MA = pd.Series(df['Close'].rolling(min_periods=1, center=True, window=n).mean(), name = 'MA_' + str(n))
    df = df.join(MA)
    return df

# Printing the Moving Average
print("MOVING AVERAGE")
stock_ma_df = MA(stock_price_df, 7)

# Print the last 10 items
print tabulate(stock_ma_df[['Close', 'MA_7']].tail(10), headers='keys', tablefmt='psql')

# Plot the moving averate for the last 90 days
plot_series(stock_ma_df, ['Close', 'MA_7'], 90, 'MA plot')

# Exponential Moving Average
def EMA(df, n):
    EMA = pd.Series(df['Close'].ewm(span=n, min_periods = 1).mean(), name='EMA_' + str(n))
    df = df.join(EMA)
    return df

# Printing the Exponential Moving Average
print("EXPONENTIAL MOVING AVERAGE")
stock_ema_df = EMA(stock_ma_df,14)

# Print the last 10 items
print tabulate(stock_ema_df[['Close', 'MA_7', 'EMA_14']].tail(10), headers='keys', tablefmt='psql')

# Plot the moving averate for the last 90 days
plot_series(stock_ema_df, ['Close', 'MA_7', 'EMA_14'], 90, 'EMA plot')

# Momentum  
def MOM(df, n):  
    M = pd.Series(df['Close'].diff(n), name = 'MOM_' + str(n))  
    df = df.join(M)  
    return df

# Printing the Momentum Indicator
print("MOMENTUM")
stock_mom_df = MOM(stock_ema_df, 7)
# print stock_ema_df.info()
# print stock_ema_df.head()

print tabulate(stock_mom_df[['Close', 'MOM_7']].tail(10), headers='keys', tablefmt='psql')
plot_series(stock_mom_df, ['Close', 'MOM_7'], 90, 'Momentum plot')

# Plot Momentum only
plot_series(stock_mom_df, ['MOM_7'], 90, 'Momentum only plot')

# Rate of Change  
def ROC(df, n):  
    M = df['Close'].diff(n - 1)  
    N = df['Close'].shift(n - 1)  
    ROC = pd.Series(M / N, name = 'ROC_' + str(n))  
    df = df.join(ROC)
    return df

# Printing the ROC Indicator
print("RATE OF CHANGE")
stock_roc_df = ROC(stock_mom_df, 7)
# print stock_ema_df.info()
# print stock_ema_df.head()

print tabulate(stock_roc_df[['Close', 'ROC_7']].tail(10), headers='keys', tablefmt='psql')
plot_series(stock_roc_df, ['Close', 'ROC_7'], 90, 'Rate of Change plot')

#Plot of the Rate of Change
plot_series(stock_roc_df, ['ROC_7'], 90, 'Rate of Change only plot')

# Implementation of the Moving Average Convergence Divergence (MACD) function
def MACD(df, n_fast, n_slow):
    EMAfast = pd.Series(df['Close'].ewm(span=n_fast, min_periods=1).mean(), name='EMAfast')
    EMAslow = pd.Series(df['Close'].ewm(span=n_slow, min_periods=1).mean(), name='EMAslow')

    name = 'MACD_' + str(n_fast) + '_' + str(n_slow)
    MACD = pd.Series(EMAfast - EMAslow, name = name)
    MACDsign = pd.Series(MACD.ewm(span=9, min_periods = 1).mean(), 
                         name='MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))

    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df

# Printing the Exponential Moving Average
print("MACD")
stock_macd_df = MACD(stock_ema_df,12,26)
# print stock_macd_df.info()
# print stock_macd_df.head()

print tabulate(stock_macd_df[['Close', 'MACD_12_26', 'MACDsign_12_26', 'MACDdiff_12_26']].tail(10), headers='keys', tablefmt='psql')
plot_series(stock_macd_df, ['Close', 'MACD_12_26', 'MACDsign_12_26', 'MACDdiff_12_26'], 90, 'MACD plot')

# Let's look at the MACD measures closely
plot_series(stock_macd_df, ['MACD_12_26', 'MACDsign_12_26', 'MACDdiff_12_26'], 90, 'EMA plot')

# Bollinger Bands  
def BBANDS(df, n):  
    MA = pd.Series(df['Close'].rolling(min_periods=1, center=False, window=n).mean(), name = 'MA1_' + str(n)) 
    MSD = pd.Series(df['Close'].rolling(min_periods=1, center=False, window=n).mean(), name = 'MSD_' + str(n))
    b1 = 4 * MSD / MA  
    B1 = pd.Series(b1, name = 'BollingerB_' + str(n))  
    df = df.join(B1)  
    b2 = (df['Close'] - MA + 2 * MSD) / (4 * MSD)  
    B2 = pd.Series(b2, name = 'Bollinger%b_' + str(n))  
    df = df.join(B2)  
    return df

# Printing the Exponential Moving Average
print("BOLLINGER BANDS")
stock_bband_df = BBANDS(stock_ema_df, 21)
# print stock_macd_df.info()
# print stock_macd_df.head()

print tabulate(stock_bband_df[['Close', 'BollingerB_21', 'Bollinger%b_21']].tail(10), headers='keys', tablefmt='psql')
plot_series(stock_bband_df, ['BollingerB_21', 'Bollinger%b_21'], 90, 'MA plot')

# Pivot Points, Supports and Resistances  
def PPSR(df):  
    PP = pd.Series((df['High'] + df['Low'] + df['Close']) / 3)  
    R1 = pd.Series(2 * PP - df['Low'])  
    S1 = pd.Series(2 * PP - df['High'])  
    R2 = pd.Series(PP + df['High'] - df['Low'])  
    S2 = pd.Series(PP - df['High'] + df['Low'])  
    R3 = pd.Series(df['High'] + 2 * (PP - df['Low']))  
    S3 = pd.Series(df['Low'] - 2 * (df['High'] - PP))  
    psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}  
    PSR = pd.DataFrame(psr)  
    df = df.join(PSR)  
    return df

# Printing the Pivot Points, Supports and Resistances
print("PIVOT POINTS, SUPPORTS AND RESISTANCES")
stock_ppsr_df = PPSR(stock_ema_df)
# print stock_ppsr_df.info()
# print stock_ppsr_df.head()

print tabulate(stock_ppsr_df[['Close', 'PP', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3']].tail(10), headers='keys', tablefmt='psql')
plot_series(stock_ppsr_df, ['Close', 'PP', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3'], 90, 'PPSR plot')

# Now we are ready to explore two commmon ossicilators
# Stochastic oscillator %K  
def STOK(df):  
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name = 'SO%k')  
    df = df.join(SOk)  
    return df

# Stochastic oscillator %D
def STOD(df, n):  
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name = 'SO%k')  
    SOd = pd.Series(SOk.ewm(span=n, min_periods=n-1).mean(), name = 'SO%d_' + str(n))  
    df = df.join(SOd)  
    return df

# Printing the Stochastic oscillator %K
print("STOCHASTIC OSCILLATORS")
stock_stok_df = STOK(stock_macd_df)

print tabulate(stock_stok_df[['Close', 'SO%k']].tail(10), headers='keys', tablefmt='psql')
plot_series(stock_stok_df, ['SO%k'], 90, 'STOK plot')

# Printing the Stochastic oscillator %D
print("STOCHASTIC OSCILLATORS")
stock_stod_df = STOD(stock_stok_df, 7)

print tabulate(stock_stod_df[['Close', 'SO%d_7']].tail(10), headers='keys', tablefmt='psql')
plot_series(stock_stod_df, ['SO%d_7'], 90, 'STOD plot')

# Ultimate Oscillator  
def ULTOSC(df):  
    # df.index = pd.to_datetime(df.pop('Date'))
    i = 0  
    TR_l = [0]  
    BP_l = [0]  
    df.reset_index(level=0, inplace=True)
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) 
        - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        TR_l.append(TR)  
        BP = df.get_value(i + 1, 'Close') 
        - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        BP_l.append(BP)  
        i = i + 1  
    UltO = pd.Series((4 * pd.Series(BP_l).rolling(window=7,center=False).sum() 
                        / pd.Series(TR_l).rolling(window=7,center=False).sum())
                     + (2 * pd.Series(BP_l).rolling(window=14,center=False).sum()
                        / pd.Series(TR_l).rolling(window=14,center=False).sum()) 
                     + (pd.Series(BP_l).rolling(window=28,center=False).sum() 
                        / pd.Series(TR_l).rolling(window=28,center=False).sum()), 
                     name = 'Ultimate_Osc')  
    df = df.join(UltO)  
    df.index = pd.to_datetime(df.pop('Date'))
    return df

# Printing the Ultimate Oscillator
print("ULTIMATE OSCILLATOR")
stock_ultsoc_df = ULTOSC(stock_stod_df)

print tabulate(stock_ultsoc_df[['Close', 'Ultimate_Osc']].tail(10), headers='keys', tablefmt='psql')
plot_series(stock_ultsoc_df, ['Ultimate_Osc'], 90, 'ULTOSC plot')

# data = quandl.get_table('MER/F1', paginate=True)
# for index, row in symbol_df.iterrows():
#    print row['Symbol']
#    data = quandl.get_table('ZACKS/FC', ticker = row['Symbol'])
#    print(data)

# This API call returns 250 columns
fund_df = quandl.get_table('ZACKS/FC', ticker='AAPL')
fund_df2 = quandl.get_table('ZACKS/CP', ticker='AAPL')

fund_df.info()
fund_df.columns
# print tabulate(fund_df.tail(10), headers='keys', tablefmt='psql')

#Daily USA Gold Prices
print("Daily USA Gold Prices")
gold_df = quandl.get("WGC/GOLD_DAILY_USD", start_date=start, end_date=end)
print tabulate(gold_df.tail(10), headers='keys', tablefmt='psql')

#Cushing, OK WTI Spot Price FOB, Daily
print("Cushing, OK WTI Spot Price FOB, Daily")
print("US Energy Information Administration Data")
oil_df = quandl.get("EIA/PET_RWTC_D", start_date=start, end_date=end)
print tabulate(oil_df.tail(10), headers='keys', tablefmt='psql')

#Natural Rate of Unemployment (Short Term)
print("Natural Rate of Unemployment (Short-term)")
s_uemp_df = quandl.get("FRED/NROUST", start_date=start, end_date=end)
print tabulate(s_uemp_df.tail(10), headers='keys', tablefmt='psql')

#Natural Rate of Unemployment (Long Term)
print("Natural Rate of Unemployment (Long-term)")
l_uemp_df = quandl.get("FRED/NROU", start_date=start, end_date=end)
print tabulate(l_uemp_df.tail(10), headers='keys', tablefmt='psql')

#Real Potential Gross Domestic Product
print("Real Potential Gross Domestic Product")
gdp_df = quandl.get("FRED/GDPPOT", start_date=start, end_date=end)
print tabulate(gdp_df.tail(10), headers='keys', tablefmt='psql')



