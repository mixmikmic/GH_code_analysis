# from importlib import reload
from tsmom import *
import tsmom
# init_notebook_mode(connected=True)
import os
os.chdir('/Users/Ravi/Desktop/ETFMomentum')
from importlib import reload
# reload(tsmom)

url = 'https://github.com/quantopian/research_public/tree/master/advanced_sample_analyses/TSMOM/data'
factors = 'factors.csv'
futures = 'futures.csv'
fut_list = 'futures_list.csv'
book = pd.read_csv(futures, parse_dates= True, index_col= [0])
fut_info = pd.read_csv(fut_list)
# fut_info.head()

daily_cum = book.apply(lambda x: get_eq_line(x, dtime = 'daily'))
mnth_cum = book.apply(lambda x: get_eq_line(x))
mnth_vol = book.apply(lambda x: get_exante_vol(x, com = None))
# daily_cum.plot(logy = True)

daily_cum.plot(figsize = (12, 5), 
               logy = True,
               legend = False, 
               grid = True, 
               title = 'Cumulative Performance')
plt.ylabel('Cumulative Returns')

futures_exc_rets = get_excess_rets(daily_cum, kind = 'arth', freq = 'm', data_type= 'returns' )

tpose = futures_exc_rets.T
tpose.index.name = 'Asset'
tpose.reset_index(inplace = True)
merged = pd.merge(tpose, fut_info, on= 'Asset')
merged.sort_values(by = ['asset_class'.upper(), 'futures'.upper()], inplace = True)
merged.set_index(['asset_class'.upper(), 'futures'.upper()], inplace = True)
del merged['Asset']
straight = merged.T

straight.tail()

bonds = (straight.BOND)
comm = (straight.COMMODITIES)
currncs = (straight.CURRENCIES)
eq_idx = (straight['EQUITY INDEXES'])
df_ts_df = get_ts(bonds)

fig_lag, axs = plt.subplots(3, 3, 
                            figsize = (16, 10), 
                            facecolor = 'white', 
                            edgecolor = 'k', 
                            )

plt.suptitle('Lagged t-stats for Bonds', size = 25)
axs = axs.ravel()
fig_lag.subplots_adjust(hspace = 0.5)
fig_lag.text(0.01, 0.5, 
             't-stats', 
             rotation = 90, 
             va = 'center', 
             ha = 'center',
             fontdict = {'fontsize':20}
            )

fig_lag.text(0.5, 0.05, 
             'Number of Lags', 
             rotation = 0, 
             va = 'center', 
             ha = 'center',
             fontdict = {'fontsize':20}
            )
for i in range(len(df_ts_df.columns[:9])):
    _ax = axs[i]
    ser = df_ts_df.iloc[:, i]
    _ax.set_title('{}'.format(ser.name), fontdict={'fontsize': 12})
    _ax.bar(ser.index, ser.values, width = 0.5, color = 'black')

kurt = book.kurt()
kurt.name = 'Kurtosis'
ann_av, ann_vol, sr_ = get_stats(book, dtime= 'daily')
ann_av.name = 'Ann Mean'
ann_vol.name = 'Ann Volatility'
sr_.name = 'Sharpe Ratio'
frst_dates = daily_cum.apply(lambda x: x.first_valid_index().strftime('%Y-%m-%d'))
frst_dates.name = 'Start'
sk = book.skew()
sk.name = 'Skewness'
sumstats = pd.concat([frst_dates, ann_av, ann_vol, sk, kurt, sr_], axis = 1)
sumstats.index.name = 'Asset'
sumstats.reset_index(inplace= True)
summerged = sumstats.merge(fut_info)

summerged.sort_values(by=["ASSET_CLASS", "FUTURES"], inplace=True)
summerged.set_index(['asset_class'.upper(), 'futures'.upper()], inplace = True)
del summerged['Asset']
summerged.iloc[:, 1:] = summerged.iloc[:, 1:].applymap(lambda x: np.round(x, 3)) 

summerged.style.set_properties(**{'background-color' : 'black', 
                             'color': 'white', 
                             'text-align': 'center', 
                             'font-size': '9pt'})\
         .set_caption('Summary Statistics for Futures Contracts')

'Maximum Sharpe ratio is {0:,.2} of {1}'.format(summerged.iloc[:,-1].max(), 
                                                summerged.iloc[:,-1].idxmax(axis = 0)[1])

unlevered = get_tsmom_port(mnth_vol, mnth_cum)
levered = get_tsmom_port(mnth_vol, mnth_cum, flag = True, scale = 0.20)

unlevered.loc[:, 'Cumulative'] = (1 + unlevered.loc[:, 'TSMOM']).cumprod()
levered.loc[:, 'Cumulative VolScale'] = (1 + levered.loc[:, 'TSMOM VolScale']).cumprod()

strat = unlevered.iloc[:, 0]
strat_vol = levered.iloc[:,0]

def get_backtest_res(series, freq = 'monthly'):
    port_mean, port_std, port_sr = get_stats(series, freq)
    nmonths = series.shape[0]
    if freq == 'daily':
        nmonths = series.resample('BM').last().shape[0]
    
    print('Back test period for {0} is from {1} to {2} with a total of {3} months \n\n'.format(series.name, 
                                        series.index[0].date().strftime('%b-%Y'), 
                                        series.index[-1].date().strftime('%b-%Y'), 
                                        nmonths))
    print(('Annualized returns for {0} is: {1:,.2%}\nAnnualized volatility for {0} is: {2:,.2%}\nSharpe Ratio for {0} is {3:,.3f}\n\n').format(series.name, 
                                              port_mean, 
                                              port_std, 
                                              port_sr, 
                                               ))
    
get_backtest_res(strat)
get_backtest_res(strat_vol)


spy_prices = pd.read_csv('spy_1985.csv', 
                         index_col = [0], 
                         parse_dates= True, 
                         dtype= {'Adj Close**': np.float64}, 
                         thousands = ',')
spy_prices = spy_prices.loc[:, 'Adj Close**']
spy_prices.name = 'SPY'

spy_mnth_cum = get_eq_line(spy_prices, data = 'prices', dtime = 'monthly')
# spy_mnth_cum = spy_mnth_cum.reindex(unlevered.index)
spy_mnth_cum.iloc[0] = 1
spy_rets = get_rets(spy_prices).dropna()
spy_rets = spy_rets.reindex(unlevered.index)
spy_rets_d = get_rets(spy_prices, freq= 'd')

spy_mean, spy_std, spy_sr = get_stats(spy_rets_d, dtime= 'daily')

print('Back test period for {0} is from {1} to {2} with a total of {3} days and {4} months \n\n'.format(spy_rets_d.name, 
                                                     spy_rets.index[0].date().strftime('%b-%Y'), 
                                                     spy_rets.index[-1].date().strftime('%b-%Y'), 
                                                     np.busday_count(spy_rets.index[0], spy_rets.index[-1]),
                                                     spy_rets.shape[0], 
                                                     ))
print(('Annualized Mean for {3} is: {0:,.2%}\nAnnualized volatility for {3} is: {1:,.2%}\nSharpe Ratio for {3} is {2:,.3f}').format(spy_mean, 
                                          spy_std, 
                                          spy_sr, 
                                          spy_rets.name))

pd.concat([get_perf_att(strat, spy_rets), 
           get_perf_att(strat_vol, spy_rets), 
           get_perf_att(spy_rets_d, spy_rets_d, freq='daily')], axis = 1)

unlevered.loc[:, 'SPY'] = spy_mnth_cum
levered.loc[:, 'SPY'] = spy_mnth_cum
# unlevered.iloc[0, -2:] = 1
# levered.iloc[0, -2:] = 1

x = spy_rets.fillna(0)
y = strat
from scipy.interpolate import UnivariateSpline
ax, fig = plt.subplots(figsize = (7, 5))
plt.scatter(x * 100, 
            y * 100, 
            c = '#284c87', 
            marker= 'o', 
            s = 10,
            alpha = 0.8)
model = np.polyfit(x = x * 100, 
                   y = y * 100, 
                   deg = 2)
f = np.poly1d(model)

xs = np.linspace(np.min(x) * 100, 
                 np.max(x) * 100, 
                 250)
ys = f(xs)
plt.plot(xs, ys, ls = '--', c = 'black')

plt.axhline(y = 0, c = 'k', ls = '-.')
plt.axvline(x = 0, c = 'k', ls = '-.')
plt.title('Monthly Returns Scatter Plot', 
          dict(fontsize = 16))
plt.xlabel('S&P 500 Returns%', 
          dict(fontsize = 11, 
                color = 'k'))
plt.ylabel('TSMOM VolScale Returns %', 
           dict(fontsize = 11, 
                color = 'k'))

ts_idx = levered.index
init_cap = 100000
trace_10 = Scatter(x = ts_idx,
                   y = unlevered.Cumulative * init_cap,
                   visible = False, 
                   name = 'Unlevered TSMOM',
                   yaxis = 'y2',
                   line = dict(dash = 'line',
                               color = 'black', 
                               width = 3)
                 )
trace_20 = Scatter(x = ts_idx,
                   y = unlevered.Leverage,
                   visible = False,
                   name = 'Leverage',
                   line = dict(dash = 'dot',
                               color = '#aaa9a9', 
                               width = 1.5, 
                               )
                 )
                  
trace_30 =  Scatter(x = ts_idx,
                    y = unlevered.SPY * init_cap,
                    visible = False,
                    name = 'SPY',
                    yaxis = 'y2',
                    line = dict(dash = 'line', 
                                color = 'blue', 
                                width = 3)    
                   )

trace_11 = Scatter(x = ts_idx,
                   y = (levered['Cumulative VolScale'] * init_cap),
                   visible = True,
                   name = 'Levered TSMOM',
                   yaxis = 'y2',
                   line = dict(dash = 'line',
                               color = 'black', 
                               width = 3)
                 )
trace_12 = Scatter(x = ts_idx,
                   y = (levered.Leverage),
                   visible = True,
                   name = 'Leverage',
                   line = dict(dash = 'dot',
                               color = '#aaa9a9', 
                               width = 1.5),
                 )
trace_13 = Scatter(x = ts_idx,
                   y = (levered.SPY) * init_cap,
                   visible = True,
                   name = 'SPY',
                   yaxis = 'y2',
                   line = dict(dash = 'line',
                               color = 'blue', 
                               width = 3,
                              )
                  )
data_ttl = [trace_10, trace_20, trace_30, trace_11, trace_12, trace_13]
updatemenus = list([dict(#type="buttons", 
                         active= 1, 
                         
                         buttons=list([dict(label = 'Unlevered',  
                                            method = 'update', 
                                            args = [{'visible': [True, True, True, False, False, False]},]),
                       
                                       dict(label = 'Levered', 
                                            method = 'update', 
                                            args = [{'visible': [False, False, False, True, True, True]},]), 
                                       
                                      ]),  
                         x = 0.15,
                         y = 1.2
                        ),
                   ], 
                   
                  )

lay = Layout(title = 'TSMOM scaled for volatility',
             legend = dict(x = 0.75, y = 1.22),

             xaxis = dict(title = 'Dates', 
                          showgrid = True, 
                          showticklabels = True,
                          linecolor = 'black',
                          tickformat = '%b %Y',
                          hoverformat = '%b %Y'
                         ),
              yaxis = dict(title = 'Leverage', 
                          showgrid = False,
                          showticklabels = True,
                          linecolor = 'black',
                          range = [0, 6],   
                          ),
             yaxis2 = dict(title = 'Cumulative Returns', 
                          showgrid = False,
                          showticklabels = True,
                          linecolor = 'black',
                          range = [4.8922, 7.65863],
                          nticks = 5,
                          type = 'log',
                           side = 'right',
                           overlaying = 'y'
                          ),
             
             paper_bgcolor = 'white',
             plot_bgcolor = 'white',
             autosize = False,
             height = 500,
             width = 950,
             showlegend = True,
             updatemenus = updatemenus,

             shapes = [
                 {
                     'type' : 'line',
                     'xref' : 'paper',
                     'x0' : 0,
                     'y0' : init_cap,
                     'x1' : 1,
                     'y1' : init_cap,
                     'line' : {
                         
                         'color': 'black',
                         'width': 1,
                         'dash': 'dash'
                             },
                 },
                     {
                      'x0':'2000-03-15', 
                      'x1':'2000-09-15', 
                      'yref': 'paper',
                      'y0':0,
                      'y1':1,
                      'fillcolor':'rgba(30,30,30,0.3)',  
                      'opacity':.2, 
                      'line' : {'width': 0
                               },
                      },
                     {
                      
                      'x0':'2007-08-01', 
                      'x1':'2009-06-01', 
                      'yref': 'paper',
                      'y0':0,
                      'y1':1,
                      'fillcolor':'rgba(30,30,30,0.3)',  
                      'opacity':.2, 
                      'line' : {'width': 0
                               },
                      },
             ]
            )
annot = []

annot.extend([dict(xref = 'paper',
                  yref = 'paper',
                  x= 0.82, 
                  y= 0.98,
                  xanchor ='right', 
                  yanchor='right',
                  text= 'Global Financial Crisis',
                  font=dict(family='<b>Arial<b>',
                            size= 12, 
                            color= 'black',
                           ),
                  showarrow=False), 
            dict(xref = 'paper',
                  yref = 'paper',
                  x= 0.515, 
                  y= 0.98,
                  xanchor ='right', 
                  yanchor='right',
                  text= 'Dotcom',
                  font=dict(family='<b>Arial<b>',
                            size= 12, 
                            color= 'black',
                           ),
                  showarrow=False)
             ]
            )

lay['annotations'] = annot
fig_ttl = Figure(data = data_ttl, layout = lay)
iplot(fig_ttl, show_link = False, filename = '/Users/Ravi/Desktop/Git/india-famafrench/docs/_static/CTAPort.html')

tsmom.get_monthly_heatmap(strat, 
                    'RdYlGn', 
                    yr_from='2000', 
                    filename = '/Users/Ravi/Desktop/Git/india-famafrench/docs/_static/CTAHeatmap.html',
                    width = 600,
                    height = 500,
                    colors = ['black', 'black'],
                    plt_type= 'iplot', 
                    vmin = -50,
                    show_scale = False)
tsmom.get_monthly_heatmap(strat_vol, 
                    'RdYlGn', 
                    yr_from = '2000', 
                    width = 600,
                    height= 500,
                    vmin = -50,
                    colors = ['black', 'black'],
                    show_scale = False,
                    filename = '/Users/Ravi/Desktop/Git/india-famafrench/docs/_static/CTAVolScaleHeatmap.html', 
                    plt_type = 'iplot')

tsmom.get_monthly_heatmap(spy_rets, 
                    'RdYlGn', 
                    yr_from = '2000', 
                    width = 600,
                    height= 500,
                    vmin = 50,
                    show_scale = False,
                    filename = '/Users/Ravi/Desktop/Git/india-famafrench/docs/_static/CTASPYHeatmap.html', 
                    plt_type = 'iplot')


# reload(tsmom)
tsmom.underwater(strat, spy_rets,
           range = [-60, 0], 
           filename = '/Users/Ravi/Desktop/Git/india-famafrench/docs/_static/CTADrawdown.html', 
           plt_type = 'iplot', width = 550, height = 320, online= False
           
           )
tsmom.underwater(strat_vol, spy_rets,
           range = [-60, 0], 
           filename = '/Users/Ravi/Desktop/Git/india-famafrench/docs/_static/CTAVolDrawdown.html', 
           plt_type = 'iplot', 
           width = 550, height = 320)

iplot(tsmom.get_ann_ret_plot(strat, 
                       height = 500, 
                       width = 500, 
                       x2range = [-40, 40], 
                       ), 
     filename = '/Users/Ravi/Desktop/Git/india-famafrench/docs/_static/CTAAnnRet.html',
     show_link = False)
iplot(tsmom.get_ann_ret_plot(strat_vol, 
                       height = 500, 
                       width = 500, 
                       x2range = [-40, 40], 
                      ), 
    show_link = False,
    filename = '/Users/Ravi/Desktop/Git/india-famafrench/docs/_static/CTAVolAnnRet.html')

iplot(tsmom.get_ann_ret_plot(spy_rets_d, 
                      dtime = 'daily',
                       height = 500, 
                       width = 500, 
#                        x2range = [40-, 40], 
                      ), 
    show_link = False,
    filename = '/Users/Ravi/Desktop/Git/india-famafrench/docs/_static/CTASPYAnnRet.html')

periods = pf.interesting_periods.PERIODS
p_df = pd.DataFrame(periods)
p_df.index = ['From', 'To']
p_df = p_df.transpose()
p_df = p_df.applymap(lambda x: x.to_period(freq = 'd'))
p_df

# def css():
#     style = open('css/custom.css', 'r').read()
#     return HTML(style)
# css()
reload(tsmom)

ff_facs = tsmom.get_ff_rolling_factors(strat= strat_vol)

tsmom.plot_rolling_ff(strat_vol, width = 700, rng = [-2,3], rolling_window= 72)
# iplot()

