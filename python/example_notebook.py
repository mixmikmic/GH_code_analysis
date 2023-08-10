get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
from techindicators import * # This line imports all functions from techindicators.py
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_finance import candlestick2_ohlc

ticker='SPY'
stockdata = np.genfromtxt('example_data.csv', delimiter=',')
sd_open = stockdata[:,1] # Open
sd_high = stockdata[:,2] # High
sd_low = stockdata[:,3] # Low
sd_close = stockdata[:,4] # Close
sd_volume = stockdata[:,5] # Volume
sd_dates = np.loadtxt('example_data.csv', delimiter=',', usecols=(0), dtype='datetime64[D]') # Dates
tradedays = np.arange(len(sd_close)) # Array of number of trading days

sma50 = sma(sd_close,50) # calculate 50 day SMA of closing price
ema20 = ema(sd_close,20) # calculate 20 day EMA of closing price
wma50 = wma(sd_close,50) # calculated 50 day WMA of closing price
kama_sd = kama(sd_close,10,2,30) # calculate standard Kaufman adaptive moving average

# plot daily closing price of SPY along with 50-day SMA, 20-day EMA, and KAMA
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(sd_dates.astype(datetime),sd_close,color='green',lw=2,label="Close")
ax.plot(sd_dates[len(sd_dates)-len(sma50):].astype(datetime),sma50,color='red',lw=2,label='50-Day SMA')
ax.plot(sd_dates[len(sd_dates)-len(wma50):].astype(datetime),wma50,color='darkcyan',lw=2,label='50-Day WMA')
ax.plot(sd_dates[len(sd_dates)-len(ema20):].astype(datetime),ema20,color='blue',lw=2,label='20-Day EMA')
ax.plot(sd_dates[len(sd_dates)-len(kama_sd):].astype(datetime),kama_sd,color='black',lw=2,label='KAMA')
ax.set_title(ticker,fontsize=30)
ax.set_xlabel('Date',fontsize=24)
ax.set_ylabel('Price ($)',fontsize=24)
ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.95, fontsize=18, facecolor='#D9DDE1')
fig.autofmt_xdate()
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(tradedays,sd_close,color='green',lw=2,label="Close")
ax.plot(tradedays[len(sd_dates)-len(sma50):],sma50,color='red',lw=2,label='50-Day SMA')
ax.plot(tradedays[len(sd_dates)-len(wma50):],wma50,color='darkcyan',lw=2,label='50-Day WMA')
ax.plot(tradedays[len(sd_dates)-len(ema20):],ema20,color='blue',lw=2,label='20-Day EMA')
ax.plot(tradedays[len(sd_dates)-len(kama_sd):],kama_sd,color='black',lw=2,label='KAMA')
ax.set_title(ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('Price ($)',fontsize=24)
ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.95, fontsize=18, facecolor='#D9DDE1')
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

cci20 = cci(sd_high,sd_low,sd_close,20) # 20-day commodity channel index
atr14 = atr(sd_high,sd_low,sd_close,14) # 14-day average true range
rsi14 = rsi(sd_close,14) # 14-day relative strength index
rstd10 = rstd(sd_close,10) # 10-day rolling standard deviation
roc12 = roc(sd_close,12) # 12-day rate of change
print('Technical indicator values for {} on {}:'.format(ticker,sd_dates[-1]))
print('')
print('The 20-day CCI was {:.2f}.'.format(cci20[-1]))
print('The 14-day ATR was {:.2f}.'.format(atr14[-1]))
print('The 14-day RSI was {:.2f}.'.format(rsi14[-1]))
print('The 10-day rolling standard deviation was {:.2f}.'.format(rstd10[-1]))
print('The 12-day rate of change was {:.2f}.'.format(roc12[-1]))

adl_sd = adl(sd_high,sd_low,sd_close,sd_volume) # Accumulation/Distribution line

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(tradedays,adl_sd,color='green',lw=2)
ax.set_title('Accumulation/Distribution Line for %s' % ticker,fontsize=24)
ax.set_yticks([])
ax.set_xlabel('Trading Days in 2017',fontsize=18)
ax.set_ylabel('',fontsize=18)
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.show()

macd_line_sd = macd(sd_close,12,26,9)[0]
macd_signal_sd = macd(sd_close,12,26,9)[1]
macd_histogram_sd = macd_line_sd[len(macd_line_sd)-len(macd_signal_sd):]-macd_signal_sd

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(tradedays[len(tradedays)-len(macd_line_sd):],macd_line_sd,color='green',lw=2,label="Line")
ax.plot(tradedays[len(tradedays)-len(macd_signal_sd):],macd_signal_sd,color='red',lw=2,label="Signal")
ax.set_title('MACD (12,26,9) for %s' % ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.95, fontsize=16, facecolor='#D9DDE1')
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

fig, ax = plt.subplots(figsize=(10,4))
ax.bar(tradedays[len(tradedays)-len(macd_histogram_sd):],macd_histogram_sd,label="MACD Histogram")
ax.set_title(ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.95, fontsize=14, facecolor='#D9DDE1')
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

ppo_line_sd = ppo(sd_close,12,26,9)[0]
ppo_signal_sd = ppo(sd_close,12,26,9)[1]
ppo_histogram_sd = ppo_line_sd[len(ppo_line_sd)-len(ppo_signal_sd):]-ppo_signal_sd

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(tradedays[len(tradedays)-len(ppo_line_sd):],ppo_line_sd,color='green',lw=2,label="Line")
ax.plot(tradedays[len(tradedays)-len(ppo_signal_sd):],ppo_signal_sd,color='red',lw=2,label="Signal")
ax.set_title('PPO (12,26,9) for %s' % ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.95, fontsize=16, facecolor='#D9DDE1')
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

fig, ax = plt.subplots(figsize=(10,4))
ax.bar(tradedays[len(tradedays)-len(ppo_histogram_sd):],ppo_histogram_sd,label="PPO Histogram")
ax.set_title(ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.95, fontsize=14, facecolor='#D9DDE1')
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

trix_line_sd = trix(sd_close,15,9)[0]
trix_signal_sd = trix(sd_close,15,9)[1]

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(tradedays[len(tradedays)-len(trix_line_sd):],trix_line_sd,color='green',lw=2,label="Line")
ax.plot(tradedays[len(tradedays)-len(trix_signal_sd):],trix_signal_sd,color='red',lw=2,label="Signal")
ax.set_title('TRIX (15,9) for %s' % ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.95, fontsize=16, facecolor='#D9DDE1')
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

kelt_sd = kelt(sd_high,sd_low,sd_close,20,2.0,10) # Kelter Channel calculated with standard parameters
lowl = kelt_sd[0] # lower line
cenl = kelt_sd[1] # center line
uppl = kelt_sd[2] # upper line

fig, ax = plt.subplots(figsize=(10,8))
ax.set_ylabel('Price ($)')
ax.set_xlabel('Date')
candlestick2_ohlc(ax,sd_open,sd_high,sd_low,sd_close,width=0.8,colorup='g',colordown='r',alpha=0.75)
ax.plot(tradedays[len(sd_dates)-len(lowl):],lowl,color='blue',lw=1.5,label='Keltner Channels (20,2,10)')
ax.plot(tradedays[len(sd_dates)-len(uppl):],uppl,color='blue',lw=1.5,label='')
ax.plot(tradedays[len(sd_dates)-len(cenl):],cenl,color='blue',lw=1.0,linestyle='--',label='')
ax.set_title(ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('Price ($)',fontsize=24)
ax.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.95, fontsize=18, facecolor='#D9DDE1')
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
fig.autofmt_xdate()
fig.tight_layout()
plt.show()

boll_sd = boll(sd_close,20,2.0,20) # Bollinger Bands calculated with standard parameters
lowlb = boll_sd[0] # lower line
cenlb = boll_sd[1] # center line
upplb = boll_sd[2] # upper line

fig, ax = plt.subplots(figsize=(10,8))
ax.set_ylabel('Price ($)')
ax.set_xlabel('Date')
candlestick2_ohlc(ax,sd_open,sd_high,sd_low,sd_close,width=0.8,colorup='g',colordown='r',alpha=0.75)
ax.plot(tradedays[len(sd_dates)-len(lowlb):],lowlb,color='blue',lw=1.5,label='Bollinger Bands$^\circledR$ (20,2,20)')
ax.plot(tradedays[len(sd_dates)-len(upplb):],upplb,color='blue',lw=1.5,label='')
ax.plot(tradedays[len(sd_dates)-len(cenlb):],cenlb,color='blue',lw=1.0,linestyle='--',label='')
ax.set_title(ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('Price ($)',fontsize=24)
ax.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.95, fontsize=18, facecolor='#D9DDE1')
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
fig.autofmt_xdate()
fig.tight_layout()
plt.show()

stoch_sd = stoch(sd_high,sd_low,sd_close,14,3,3) # Full stochastics calculated with standard parameters
stoch_k = stoch_sd[0] # %K parameter
stoch_d = stoch_sd[1] # %D parameter

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(tradedays[len(tradedays)-len(stoch_k):],stoch_k,color='green',lw=2,label="%K")
ax.plot(tradedays[len(tradedays)-len(stoch_d):],stoch_d,color='red',lw=2,label="%D")
ax.set_title('Stochastic Ocsillator (14,3,3) for %s' % ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.95, fontsize=16, facecolor='#D9DDE1')
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

vort_sd = vortex(sd_high,sd_low,sd_close,14)
vort_p_sd = vort_sd[0]
vort_n_sd = vort_sd[1]

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(tradedays[len(tradedays)-len(vort_p_sd):],vort_p_sd,color='green',lw=2,label="+VM")
ax.plot(tradedays[len(tradedays)-len(vort_n_sd):],vort_n_sd,color='red',lw=2,label="$-$VM")
ax.set_title('Vortex Indicator (14) for %s' % ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.95, fontsize=16, facecolor='#D9DDE1')
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

adx_sd = adx(sd_high,sd_low,sd_close,14)
adx_pdm = adx_sd[0]
adx_ndm = adx_sd[1]
adx_line = adx_sd[2]

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(tradedays[len(tradedays)-len(adx_pdm):],adx_pdm,color='green',lw=1.5,label="$+$DI")
ax.plot(tradedays[len(tradedays)-len(adx_ndm):],adx_ndm,color='red',lw=1.5,label="$-$DI")
ax.plot(tradedays[len(tradedays)-len(adx_line):],adx_line,color='black',lw=2.5,label="ADX")
ax.set_title('ADX(14) for %s' % ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.95, fontsize=16, facecolor='#D9DDE1')
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

aroon_sd = aroon(sd_high,sd_low,25)
aroon_up = aroon_sd[0]
aroon_down = aroon_sd[1]
aroon_osc = aroon_sd[2]

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(tradedays[len(tradedays)-len(aroon_up):],aroon_up,color='green',lw=2,label="Up")
ax.plot(tradedays[len(tradedays)-len(aroon_down):],aroon_down,color='red',lw=2,label="Down")
ax.set_title('Aroon(25) Indicator for %s' % ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.95, fontsize=16, facecolor='#D9DDE1')
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(tradedays[len(tradedays)-len(aroon_osc):],aroon_osc,color='blue',lw=2)
ax.plot(tradedays[len(tradedays)-len(aroon_osc):],np.zeros(len(aroon_osc)),color='gray',linestyle='dashed')
ax.set_title('Aroon(25) Oscillator for %s' % ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
plt.ylim(-105,105)
plt.xlim(np.amin(tradedays[len(tradedays)-len(aroon_osc):]),np.amax(tradedays[len(tradedays)-len(aroon_osc):]))
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

chand_long = chand(sd_high,sd_low,sd_close,22,3,'long')
chand_short = chand(sd_high,sd_low,sd_close,22,3,'short')

fig, ax = plt.subplots(figsize=(10,8))
ax.set_ylabel('Price ($)')
ax.set_xlabel('Date')
candlestick2_ohlc(ax,sd_open,sd_high,sd_low,sd_close,width=0.8,colorup='g',colordown='r',alpha=0.75)
ax.plot(tradedays[len(sd_dates)-len(chand_long):],chand_long,color='blue',lw=1.5,label='Chandelier Exit (22,3)')
ax.set_title('Chandelier Exit for %s Long Position' % ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('Price ($)',fontsize=24)
ax.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.95, fontsize=18, facecolor='#D9DDE1')
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
fig.autofmt_xdate()
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10,8))
ax.set_ylabel('Price ($)')
ax.set_xlabel('Date')
candlestick2_ohlc(ax,sd_open,sd_high,sd_low,sd_close,width=0.8,colorup='g',colordown='r',alpha=0.75)
ax.plot(tradedays[len(sd_dates)-len(chand_short):],chand_short,color='blue',lw=1.5,label='Chandelier Exit (22,3)')
ax.set_title('Chandelier Exit for %s Short Position' % ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('Price ($)',fontsize=24)
ax.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.95, fontsize=18, facecolor='#D9DDE1')
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
fig.autofmt_xdate()
fig.tight_layout()
plt.show()

copp_sd = copp(sd_close,14,11,10)

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(tradedays[len(tradedays)-len(copp_sd):],copp_sd,color='blue',lw=2)
ax.plot(tradedays[len(tradedays)-len(copp_sd):],np.zeros(len(copp_sd)),color='black',linestyle='dashed',lw=2)
ax.set_title('Coppock Curve (14,11,10) for %s' % ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.xlim(np.amin(tradedays[len(tradedays)-len(copp_sd):]),np.amax(tradedays[len(tradedays)-len(copp_sd):]))
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

force_sd = force(sd_close,sd_volume,13)

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(tradedays[len(tradedays)-len(force_sd):],force_sd,color='blue',lw=2)
ax.plot(tradedays[len(tradedays)-len(force_sd):],np.zeros(len(force_sd)),color='gray',linestyle='dashed')
ax.set_title('Force Index(13) for %s' % ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
plt.xlim(np.amin(tradedays[len(tradedays)-len(force_sd):]),np.amax(tradedays[len(tradedays)-len(force_sd):]))
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

cmf_sd = cmf(sd_high,sd_low,sd_close,sd_volume,20)

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(tradedays[len(tradedays)-len(cmf_sd):],cmf_sd,color='blue',lw=2)
ax.plot(tradedays[len(tradedays)-len(cmf_sd):],np.zeros(len(cmf_sd)),color='gray',linestyle='dashed')
ax.set_title('Chaikin Money Flow(20) for %s' % ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
plt.xlim(np.amin(tradedays[len(tradedays)-len(cmf_sd):]),np.amax(tradedays[len(tradedays)-len(cmf_sd):]))
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

chosc_sd = chosc(sd_high,sd_low,sd_close,sd_volume,3,10)

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(tradedays[len(tradedays)-len(chosc_sd):],chosc_sd,color='blue',lw=2)
ax.plot(tradedays[len(tradedays)-len(chosc_sd):],np.zeros(len(chosc_sd)),color='gray',linestyle='dashed')
ax.set_title('Chaikin Oscillator(3,10) for %s' % ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
plt.xlim(np.amin(tradedays[len(tradedays)-len(chosc_sd):]),np.amax(tradedays[len(tradedays)-len(chosc_sd):]))
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

emv_sd = emv(sd_high,sd_low,sd_volume,14)

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(tradedays[len(tradedays)-len(emv_sd):],emv_sd,color='blue',lw=2)
ax.plot(tradedays[len(tradedays)-len(emv_sd):],np.zeros(len(emv_sd)),color='gray',linestyle='dashed')
ax.set_title('Ease of Movement(14) for %s' % ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
plt.xlim(np.amin(tradedays[len(tradedays)-len(emv_sd):]),np.amax(tradedays[len(tradedays)-len(emv_sd):]))
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

mindx_sd = mindx(sd_high,sd_low,25)

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(tradedays[len(tradedays)-len(mindx_sd):],mindx_sd,color='blue',lw=2)
ax.plot(tradedays[len(tradedays)-len(mindx_sd):],np.zeros(len(mindx_sd))+27,color='gray',linestyle='dashed')
ax.set_title('Mass Index(25) for %s' % ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
plt.xlim(np.amin(tradedays[len(tradedays)-len(mindx_sd):]),np.amax(tradedays[len(tradedays)-len(mindx_sd):]))
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

mfi_sd = mfi(sd_high,sd_low,sd_close,sd_volume,14)

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(tradedays[len(tradedays)-len(mfi_sd):],mfi_sd,color='blue',lw=2)
ax.plot(tradedays[len(tradedays)-len(mfi_sd):],np.zeros(len(mfi_sd))+50,color='black',lw=1)
ax.plot(tradedays[len(tradedays)-len(mfi_sd):],np.zeros(len(mfi_sd))+20,color='black',lw=1,linestyle='dashed')
ax.plot(tradedays[len(tradedays)-len(mfi_sd):],np.zeros(len(mfi_sd))+80,color='black',lw=1,linestyle='dashed')
ax.set_title('MFI (14) for %s' % ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
plt.ylim(0,100)
plt.xlim(np.amin(tradedays[len(tradedays)-len(mfi_sd):]),np.amax(tradedays[len(tradedays)-len(mfi_sd):]))
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

nvi_sd = nvi(sd_close,sd_volume,50)
nvi_line = nvi_sd[0]
nvi_signal = nvi_sd[1]

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(tradedays[len(tradedays)-len(nvi_line):],nvi_line,color='green',lw=2,label="NVI")
ax.plot(tradedays[len(tradedays)-len(nvi_signal):],nvi_signal,color='red',lw=2,label="Signal (50 day EMA)")
ax.set_title('NVI for %s' % ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.95, fontsize=16, facecolor='#D9DDE1')
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks([],[])
plt.xticks(fontsize=18)
plt.show()

obv_sd = obv(sd_close,sd_volume)

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(tradedays,obv_sd,color='green',lw=2)
ax.set_title('On Balance Volume for %s' % ticker,fontsize=24)
ax.set_yticks([])
ax.set_xlabel('Trading Days in 2017',fontsize=18)
ax.set_ylabel('',fontsize=18)
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.show()

pvo_line_sd = pvo(sd_close,12,26,9)[0]
pvo_signal_sd = pvo(sd_close,12,26,9)[1]
pvo_histogram_sd = pvo_line_sd[len(pvo_line_sd)-len(pvo_signal_sd):]-pvo_signal_sd

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(tradedays[len(tradedays)-len(pvo_line_sd):],pvo_line_sd,color='green',lw=2,label="Line")
ax.plot(tradedays[len(tradedays)-len(pvo_signal_sd):],pvo_signal_sd,color='red',lw=2,label="Signal")
ax.set_title('PVO (12,26,9) for %s' % ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.95, fontsize=16, facecolor='#D9DDE1')
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

fig, ax = plt.subplots(figsize=(10,4))
ax.bar(tradedays[len(tradedays)-len(pvo_histogram_sd):],pvo_histogram_sd,label="PVO Histogram")
ax.set_title(ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.95, fontsize=14, facecolor='#D9DDE1')
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()

kst_sd = kst(sd_close,10,15,20,30,10,10,10,15,9)
kst_line = kst_sd[0]
kst_signal = kst_sd[1]

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(tradedays[len(tradedays)-len(kst_line):],kst_line,color='green',lw=2,label="Line")
ax.plot(tradedays[len(tradedays)-len(kst_signal):],kst_signal,color='red',lw=2,label="Signal")
ax.set_title('Know Sure Thing (KST) for %s' % ticker,fontsize=30)
ax.set_xlabel('Trading Days in 2017',fontsize=24)
ax.set_ylabel('',fontsize=24)
ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.95, fontsize=16, facecolor='#D9DDE1')
ax.grid(color='gray', linestyle='--', linewidth=1)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.show()



