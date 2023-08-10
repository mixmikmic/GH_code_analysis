import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime

def get_data(symbols, 
             add_ref=True,
             data_source='yahoo',
             price='Adj Close',
             start='1/21/2010', 
             end='4/15/2016'):
    """Read stock data (adjusted close) for given symbols from."""
    
    if add_ref and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    df = web.DataReader(symbols, 
                        data_source=data_source,
                        start=start, 
                        end=end)
    
    return df[price,:,:]

def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # Note: Returned DataFrame must have the same number of rows
    daily_returns = (df / df.shift(1)) - 1 
    daily_returns.ix[0,:] = 0 
    return daily_returns

def fill_missing_values(df_data):
    """Fill missing values in data frame, in place."""
    df_data.fillna(method='ffill',inplace=True)
    df_data.fillna(method='backfill',inplace=True)
    return df_data

def cumulative_returns(df):
    return df/df.ix[0,:] - 1 

df = fill_missing_values(get_data(symbols=['GOOG','SPY','IBM','GLD'],
                             start='4/21/2015', 
                             end='7/15/2016'))
df.plot()
plt.show(1)

df.head()

simbols = ["A","AA","AAL","AAP","AAPL","ABB","ABBV","ABC","ABMD","ABT","ABX","ACC","ACGL","ACM","ACN","ADBE","ADI","ADM","ADNT","ADP","ADS","ADSK","AEE","AEG","AEM","AEP","AER","AES","AET","AFG","AFL","AGCO","AGN","AGNC","AGR","AGU","AIG","AIV","AIZ","AJG","AKAM","ALB","ALGN","ALK","ALKS","ALL","ALLE","ALLY","ALSN","ALV","ALXN","AM","AMAT","AMD","AME","AMG","AMGN","AMH","AMP","AMT","AMTD","AMX","AMZN","ANET","ANSS","ANTM","AON","AOS","APA","APC","APD","APH","APO","AR","ARCC","ARD","ARE","ARMK","ARNC","ARRS","ARW","ASH","ASML","ASR","ASX","ATH","ATO","ATR","ATVI","AVB","AVGO","AVY","AWK","AXP","AXS","AXTA","AYI","AZN","AZO","BA","BABA","BAC","BAH","BAM","BAP","BAX","BBBY","BBD","BBL","BBRY","BBT","BBVA","BBY","BC","BCE","BCH","BCR","BCS","BDX","BEN","BERY","BF.B","BG","BHI","BHP","BIDU","BIIB","BIO","BIP","BIVV","BK","BKFS","BLK","BLL","BMA","BMO","BMRN","BMY","BNS","BOKF","BP","BPL","BPY","BR","BRCD","BRFS","BRK.A","BRO","BRX","BSAC","BSBR","BSMX","BSX","BT","BUD","BURL","BWA","BX","BXP","C","CA","CAG","CAH","CAJ","CAT","CB","CBG","CBOE","CBS","CBSH","CC","CCE","CCI","CCK","CCL","CDK","CDNS","CDW","CE","CELG","CEO","CERN","CF","CFG","CFR","CG","CGNX","CHA","CHD","CHK","CHKP","CHL","CHRW","CHT","CHTR","CHU","CI","CINF","CIT","CL","CLNS","CLR","CLX","CM","CMA","CMCSA","CME","CMG","CMI","CMS","CNA","CNC","CNHI","CNI","CNK","CNP","CNQ","COF","COG","COH","COHR","COL","COMM","COO","COP","COST","COTY","CP","CPA","CPB","CPL","CPRT","CPT","CRH","CRM","CS","CSCO","CSGP","CSL","CSX","CTAS","CTL","CTRP","CTSH","CTXS","CUK","CVE","CVS","CVX","CX","CXO","D","DAL","DB","DCI","DCM","DCP","DD","DE","DEI","DEO","DFS","DG","DGX","DHI","DHR","DIS","DISH","DKS","DLB","DLPH","DLR","DLTR","DNKN","DOV","DOW","DOX","DPS","DPZ","DRE","DRI","DTE","DUK","DVA","DVMT","DVN","DXC","DXCM","E","EA","EBAY","EC","ECA","ECL","ED","EDU","EEP","EFX","EGN","EIX","EL","ELS","EMN","EMR","ENB","ENBL","ENIA","ENIC","ENLK","EOCC","EOG","EPD","EPR","EQGP","EQIX","EQM","EQR","EQT","ERIC","ERIE","ES","ESLT","ESRX","ESS","ETE","ETFC","ETN","ETR","EV","EVHC","EW","EWBC","EXC","EXEL","EXPD","EXPE","EXR","F","FANG","FAST","FB","FBHS","FBR","FCE.A","FCX","FDC","FDS","FDX","FE","FFIV","FIS","FISV","FITB","FL","FLEX","FLIR","FLR","FLS","FLT","FMC","FMS","FMX","FNF","FOXA","FRC","FRT","FTI","FTNT","FTV","G","GD","GDDY","GE","GG","GGG","GGP","GIB","GIL","GILD","GIS","GLPI","GLW","GM","GNTX","GOLD","GOOGL","GPC","GPN","GPS","GRMN","GS","GSK","GT","GWW","GXP","H","HAL","HAS","HBAN","HBI","HCA","HCN","HCP","HD","HDB","HDS","HES","HFC","HHC","HIG","HII","HIW","HLF","HLT","HMC","HOG","HOLX","HON","HP","HPE","HPP","HPQ","HRB","HRL","HRS","HSBC","HSIC","HST","HSY","HTA","HTHT","HUBB","HUM","HUN","IAC","IBKR","IBM","IBN","ICE","IDXX","IEP","IEX","IFF","IHG","ILMN","INCY","INFO","INFY","ING","INGR","INTC","INTU","INVH","IONS","IP","IPG","IPGP","IR","IRM","ISRG","IT","ITUB","ITW","IVZ","IX","JAZZ","JBHT","JBL","JBLU","JD","JEC","JHX","JKHY","JLL","JNJ","JNPR","JPM","JWN","K","KAR","KB","KEP","KEY","KEYS","KHC","KIM","KKR","KLAC","KMB","KMI","KMX","KO","KORS","KR","KRC","KSS","KSU","KT","KYO","L","LAMR","LAZ","LB","LBTYA","LDOS","LEA","LECO","LEG","LEN","LFC","LFL","LH","LII","LKQ","LLL","LLY","LMT","LN","LNC","LNT","LOGI","LOGM","LOW","LPL","LPT","LRCX","LUK","LULU","LUV","LUX","LVLT","LVS","LW","LYB","LYG","LYV","M","MA","MAA","MAC","MAN","MAR","MAS","MAT","MBLY","MBT","MCD","MCHP","MCK","MCO","MD","MDLZ","MDT","MDU","MELI","MET","MFC","MFG","MGA","MGM","MHK","MIC","MIDD","MJN","MKC","MKL","MKTX","MLCO","MLM","MMC","MMM","MMP","MNST","MO","MOMO","MON","MOS","MPC","MPLX","MRK","MRO","MRVL","MS","MSCC","MSCI","MSFT","MSI","MSM","MT","MTB","MTD","MTN","MTU","MU","MXIM","MYL","NBL","NCLH","NCR","NDAQ","NDSN","NEE","NEM","NEU","NFLX","NFX","NGG","NI","NKE","NLSN","NLY","NMR","NNN","NOC","NOK","NOV","NOW","NRZ","NSC","NTAP","NTES","NTRS","NUAN","NUE","NVDA","NVO","NVR","NVS","NWL","NXPI","NYCB","O","OA","OAK","OC","ODFL","OGE","OHI","OKE","OKS","OLED","OLN","OMC","ON","ORAN","ORCL","ORI","ORLY","OSK","OTEX","OXY","OZRK","PAA","PAC","PACW","PAGP","PANW","PAYX","PBCT","PBR","PCAR","PCG","PCLN","PE","PEG","PEP","PF","PFE","PFG","PG","PGR","PH","PHG","PHI","PHM","PII","PK","PKG","PKI","PKX","PLD","PM","PNC","PNR","PNRA","PNW","POOL","POST","POT","PPC","PPG","PPL","PRGO","PRU","PSA","PSO","PSX","PSXP","PTC","PTR","PUK","PVH","PWR","PX","PXD","PYPL","Q","QCOM","QGEN","QRVO","QVCA","RACE","RAI","RBS","RCI","RCL","RDS.A","RDY","RE","REG","REGN","RELX","RENX","RF","RGA","RHI","RHT","RIO","RJF","RL","RMD","RNR","ROK","ROL","ROP","ROST","RPM","RRC","RS","RSG","RSPP","RTN","RY","RYAAY","S","SABR","SAN","SAP","SATS","SBAC","SBNY","SBS","SBUX","SCCO","SCG","SCHW","SCI","SEE","SEIC","SEP","SERV","SGEN","SHG","SHLX","SHOP","SHPG","SHW","SINA","SIRI","SIVB","SIX","SJM","SJR","SKM","SLB","SLF","SLG","SLM","SLW","SMFG","SMG","SMI","SNA","SNAP","SNE","SNI","SNN","SNP","SNPS","SNV","SNY","SO","SON","SPB","SPG","SPGI","SPLK","SPLS","SPR","SQ","SRCL","SRE","SSL","SSNC","ST","STE","STI","STLD","STM","STO","STT","STWD","STX","STZ","SU","SUI","SWK","SWKS","SYF","SYK","SYMC","SYT","SYY","T","TAL","TAP","TD","TDG","TEAM","TECK","TEF","TEL","TER","TEVA","TFX","TGNA","TGT","TI","TIF","TJX","TKC","TLK","TLLP","TM","TMK","TMO","TMUS","TOL","TOT","TRGP","TRI","TRIP","TRMB","TROW","TRP","TRQ","TRU","TRV","TS","TSCO","TSLA","TSM","TSN","TSO","TSRO","TSS","TSU","TTC","TTM","TTWO","TU","TV","TWTR","TWX","TXN","TXT","TYL","UAL","UBS","UDR","UGI","UGP","UHAL","UHS","UL","ULTA","ULTI","UMC","UN","UNH","UNM","UNP","UPS","URI","USB","USFD","UTHR","UTX","V","VAL","VALE","VAR","VEDL","VEEV","VEON","VER","VFC","VIAB","VIPS","VIV","VLO","VMC","VMW","VNO","VNTV","VOD","VOYA","VRSK","VRSN","VRTX","VTR","VZ","W","WAB","WAL","WAT","WB","WBA","WBC","WBK","WCG","WCN","WDAY","WDC","WEC","WES","WF","WFC","WFM","WFT","WGP","WHR","WIT","WLK","WLTW","WM","WMB","WMT","WOOF","WPC","WPPGY","WPZ","WR","WRB","WRK","WST","WTR","WU","WUBA","WY","WYN","WYNN","XEC","XEL","XL","XLNX","XOM","XPO","XRAY","XRX","XYL","Y","YHOO","YNDX","YPF","YUM","YUMC","ZAYO","ZBH","ZBRA","ZION","ZTO","ZTS"]

len(simbols)

DELTA = 45 ## delay in days 
start_date = (datetime.date.today() - datetime.timedelta(DELTA)).strftime("%m-%d-%Y")
print("start_date",start_date)

end_date = (datetime.date.today()).strftime("%m-%d-%Y")
print("end_date",end_date)

df = fill_missing_values(get_data(symbols=simbols,
                             start=start_date, 
                             end=end_date))

df.shape

df.head()

pd.isnull(df).sum().sum()

to_remove = pd.isnull(df).sum()[pd.isnull(df).sum() > 0]
to_remove

to_remove.index

df = df.drop(to_remove.index,axis=1)

pd.isnull(df).sum().sum()

df.shape

from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

## let's start with an example 
series = df["AAPL"]
series = series.tolist()

pd.Series(series).plot()
plt.show()

p_tr = 0.66

tr_size = int(len(series) * p_tr)
train, test = series[0:tr_size], series[tr_size:len(series)]
print("tr_size:",tr_size,"xval_size:",(len(series)-tr_size))

from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot

# fit model
model = ARIMA(train, order=(5,0,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

model_fit.aic

from numpy.linalg import LinAlgError

def get_best_arima(sr,maxord = [6,6,6]):
    best_aic = 1000000
    best_ord = maxord
    best_model = None 
    for p in range(maxord[0]):
        for q in range(maxord[1]):
            for d in range(maxord[2]):
                try: 
                    model = ARIMA(sr, order=(p,q,d))
                    model_fit = model.fit(disp=0)
                    if (best_aic > model_fit.aic):
                        best_aic = model_fit.aic
                        best_ord = [p,q,d]
                        best_model = model_fit 
                except: 
                    pass
    return best_aic, best_ord , best_model

# supress warnings 
import warnings
warnings.filterwarnings('ignore')

best_aic, best_ord , best_model = get_best_arima(train)
print("best_aic:",best_aic)
print("best_ord:",best_ord)

def predict_arima(train,steps,best_order,verbose=True,do_plot=True):
    history = [x for x in train]
    predictions = list()
    yhat = train[len(train)-1]
    for t in range(steps):
        model = ARIMA(history, order=(best_order[0],best_order[1],best_order[2]))
        try:
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
        except:
            pass
        if not(np.isnan(output[0] )) and not(np.isinf(output[0])): 
            yhat = np.asscalar(output[0])
        predictions.append(yhat)
        history.append(yhat)
    return predictions

predictions = predict_arima(train,len(test),best_ord,verbose=True,do_plot=True)
MSE = mean_squared_error(test, predictions)
print("MSE:",MSE)

pyplot.plot(test , label="test data")    
pyplot.plot(predictions, color='red' , label="prediction")
pyplot.legend()
pyplot.show()

def predict_arima_worse(train,steps,best_order,verbose=True,do_plot=True):
    history = [x for x in train]
    predictions = list()
    model = ARIMA(history, order=(best_order[0],best_order[1],best_order[2]))
    try:
        model_fit = model.fit(disp=0)
        output = model_fit.forecast(steps=steps)
    except:
        pass
    yhat = history[len(history)-1]
    for i in range(steps): 
        if not(np.isnan(output[0][i] )) and not(np.isinf(output[0][i])): 
            yhat = np.asscalar(output[0][i] )
        predictions.append(yhat)
    return predictions

predictions = predict_arima_worse(train,len(test),best_ord,verbose=True,do_plot=True)
MSE = mean_squared_error(test, predictions)
print("MSE:",MSE)

pyplot.plot(test , label="test data")    
pyplot.plot(predictions, color='red' , label="prediction")
pyplot.legend()
pyplot.show()

arima_perf = DataFrame({'security':  df.columns.tolist(), 
                        
                        'cumulative_return_test': 0, 
                        'ARIMA_pred_cumulative_return_test': 0, 
                        'ARIMA_MSE_cumulative_return_test': 0, 
                        'ARIMA_R2_cumulative_return_test': 0, 
                        'ARIMA_MSE_price_test': 0 ,
                        'ARIMA_R2_price_test': 0 ,
                        
                        'cumulative_return_xval': 0, 
                        'ARIMA_pred_cumulative_return_xval': 0, 
                        'ARIMA_MSE_cumulative_return_xval': 0, 
                        'ARIMA_R2_cumulative_return_xval': 0, 
                        'ARIMA_MSE_price_xval': 0 ,
                        'ARIMA_R2_price_xval': 0 , 
                        
                        'ARIMA_best_order_0': 0, 
                        'ARIMA_best_order_1': 0, 
                        'ARIMA_best_order_2': 0} )
arima_perf = arima_perf.set_index(['security'])

p_tr = 0.60
p_xval = 0.20
p_test = 1-p_tr-p_xval  

for i,sec in enumerate(arima_perf.index.tolist()):
    if (i % 100 == 0) or (i ==10):
        print(i,sec)
    
    ## data 
    series = df[sec]
    series = series.tolist()
    tr_size = int(len(series) * p_tr)
    xval_size = int(len(series) * p_xval)
    train, txval, test = series[0:tr_size], series[tr_size:(tr_size+xval_size)] , series[(tr_size+xval_size):len(series)]
    
    ## fit model 
    best_aic, best_ord , best_model = get_best_arima(train)
    
    ## predict, assess 
    predictions = predict_arima(train,(len(txval)+len(test)),best_ord,verbose=False,do_plot=False)
    
    ## store
    arima_perf.loc[sec,'ARIMA_best_order_0'] = best_ord[0]
    arima_perf.loc[sec,'ARIMA_best_order_1'] = best_ord[1]
    arima_perf.loc[sec,'ARIMA_best_order_2'] = best_ord[2]
    
    # xval 
    pred_cumulative_returns_xval = cumulative_returns(pd.Series(predictions[0:xval_size]))
    cumulative_returns_xval = cumulative_returns(pd.Series(txval))
    
    arima_perf.loc[sec,'cumulative_return_xval'] = cumulative_returns_xval[len(cumulative_returns_xval)-1]
    arima_perf.loc[sec,'ARIMA_pred_cumulative_return_xval'] = pred_cumulative_returns_xval[len(pred_cumulative_returns_xval)-1]
    arima_perf.loc[sec,'ARIMA_MSE_cumulative_return_xval'] = mean_squared_error(cumulative_returns_xval, pred_cumulative_returns_xval)
    arima_perf.loc[sec,'ARIMA_R2_cumulative_return_xval'] = r2_score(cumulative_returns_xval, pred_cumulative_returns_xval)
    arima_perf.loc[sec,'ARIMA_MSE_price_xval'] = mean_squared_error(txval, predictions[0:xval_size])
    arima_perf.loc[sec,'ARIMA_R2_price_xval'] = r2_score(txval, predictions[0:xval_size])
    
    # test 
    pred_cumulative_returns_test = cumulative_returns(pd.Series(predictions[xval_size:]))
    cumulative_returns_test = cumulative_returns(pd.Series(test))
    
    arima_perf.loc[sec,'cumulative_return_test'] = cumulative_returns_test[len(cumulative_returns_test)-1]
    arima_perf.loc[sec,'ARIMA_pred_cumulative_return_test'] = pred_cumulative_returns_test[len(pred_cumulative_returns_test)-1]
    arima_perf.loc[sec,'ARIMA_MSE_cumulative_return_test'] = mean_squared_error(cumulative_returns_test, pred_cumulative_returns_test)
    arima_perf.loc[sec,'ARIMA_R2_cumulative_return_test'] = r2_score(cumulative_returns_test, pred_cumulative_returns_test)
    arima_perf.loc[sec,'ARIMA_MSE_price_test'] = mean_squared_error(test, predictions[xval_size:])
    arima_perf.loc[sec,'ARIMA_R2_price_test'] = r2_score(test, predictions[xval_size:])

arima_perf.sort_values(by='ARIMA_MSE_price_test' , ascending=False).head(5)

#arima_perf.to_csv('data/arima_perf.csv')

arima_perf = pd.read_csv('data/arima_perf.csv')

arima_perf = arima_perf.set_index('security')
arima_perf.sort_values(by='ARIMA_R2_cumulative_return_xval' , ascending=False).head(10)

arima_perf['ACTION'] = ''
for i,sec in enumerate(arima_perf.index.tolist()):
    if arima_perf.loc[sec,'ARIMA_R2_cumulative_return_xval'] > 0.65:
        
        if arima_perf.loc[sec,'ARIMA_pred_cumulative_return_xval'] > 0 and arima_perf.loc[sec,'ARIMA_pred_cumulative_return_xval']/arima_perf.loc[sec,'cumulative_return_xval'] >= 0.7 and arima_perf.loc[sec,'ARIMA_pred_cumulative_return_xval']/arima_perf.loc[sec,'cumulative_return_xval'] <= 2:
            arima_perf.loc[sec,'ACTION'] = 'BUY'
            
        if arima_perf.loc[sec,'ARIMA_pred_cumulative_return_xval'] < 0 and arima_perf.loc[sec,'ARIMA_pred_cumulative_return_xval']/arima_perf.loc[sec,'cumulative_return_xval'] >= 0.7 and arima_perf.loc[sec,'ARIMA_pred_cumulative_return_xval']/arima_perf.loc[sec,'cumulative_return_xval'] <= 2:
            arima_perf.loc[sec,'ACTION'] = 'SELL'    

arima_perf = arima_perf.sort_values(by='ARIMA_R2_cumulative_return_xval' , ascending=False)
arima_perf[['ACTION','ARIMA_R2_cumulative_return_xval','ARIMA_pred_cumulative_return_xval','cumulative_return_xval']][:20]

arima_perf = arima_perf.sort_values(by='ARIMA_pred_cumulative_return_xval' , ascending=False)
loss = arima_perf[['ACTION','ARIMA_R2_cumulative_return_xval','ARIMA_pred_cumulative_return_xval','cumulative_return_xval']][:20]
loss['UNEXPECTED_LOSS'] = loss['cumulative_return_xval'] - loss['ARIMA_pred_cumulative_return_xval']
loss

arima_perf = arima_perf.sort_values(by='ARIMA_MSE_cumulative_return_xval' , ascending=True)
flat = arima_perf[['ACTION','ARIMA_R2_cumulative_return_xval','ARIMA_MSE_cumulative_return_xval','ARIMA_pred_cumulative_return_xval','cumulative_return_xval']][:20]
flat

arima_perf = arima_perf.sort_values(by='ARIMA_R2_cumulative_return_xval' , ascending=False)
gain_loss = arima_perf[arima_perf['ACTION'] != ''][['ACTION','ARIMA_R2_cumulative_return_test','ARIMA_pred_cumulative_return_test','cumulative_return_test']]
gain_loss['EX_POST'] = ''
for i,sec in enumerate(gain_loss.index.tolist()):
    if gain_loss.loc[sec,'cumulative_return_test'] * gain_loss.loc[sec,'ARIMA_pred_cumulative_return_test'] < 0:
        gain_loss.loc[sec,'EX_POST'] = 'WRONG'
    elif np.absolute(gain_loss.loc[sec,'cumulative_return_test']) <  np.absolute(gain_loss.loc[sec,'ARIMA_pred_cumulative_return_test']):
        gain_loss.loc[sec,'EX_POST'] = 'ALMOST CORRECT'
    else:
        gain_loss.loc[sec,'EX_POST'] = 'CORRECT'
gain_loss

gain_loss[gain_loss['EX_POST'] == 'WRONG'].shape[0]/gain_loss.shape[0]

