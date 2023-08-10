import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize

#needs blpapi to be installed
import pdblp

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 10)

con = pdblp.BCon(port=8194)
con.start()

# bloomberg indices
historical_tickers = [
    'G0O1 Index',
    'LUATTRUU Index',
    'LBUTTRUU Index',
    'BCOMTR Index',
    'SPX Index',
    'BCOMGCTR Index',
    'JPEIGLBL Index',
    'LF98TRUU Index',
    'LT08TRUU Index'
]

names = [
    'RF',
    'US Tsy',
    'US Tips',
    'Commodities',
    'US Equities',
    'Gold',
    'EM Bonds',
    'HY Bonds',
    'Intermediate US Tsy'
]

flds = ['PX_LAST']
startDate = 19701231
rdf = con.bdh(historical_tickers, flds, startDate, '')

# remove multiindex
rdf.columns = rdf.columns.droplevel(1)
rdf = rdf.rename(columns=dict(zip(historical_tickers,names)))

# convert to monthly
rdf = rdf.resample('M').last()

# % change
rdf = rdf.pct_change()

con.stop()
rdf.tail()

print("Start Dates")
for i,n in enumerate(names):
    null_mask = pd.isnull(rdf[n])
    print("%s : %s" % (n, str(rdf.loc[null_mask,n].index[-1].date())))

gidf = pd.read_csv('macrodf.csv')
gidf = gidf.set_index('date')
gidf.head(15).tail(5)

rdf2 = pd.merge(
    rdf,
    gidf.loc[:,
             [
                 'INDPRO Forecast Surprise Normalized',
                 'INDPRO Forecast Surprise Signal',
                 'INDPRO Trend Surprise Normalized',
                 'INDPRO Trend Surprise Signal',
                 'INDPRO Combined Surprise Normalized',
                 'INDPRO Combined Surprise Signal',
                 'CPI Forecast Surprise Signal',
                 'CPI Forecast Surprise Normalized',
                 'CPI Trend Surprise Signal',
                 'CPI Trend Surprise Normalized',
                 'CPI Combined Surprise Signal',
                 'CPI Combined Surprise Normalized'
             ]
            ],
    how='left',
    left_index=True,
    right_index=True
)

# start at 73
start73_mask = rdf2.index >= pd.datetime(1973,1,1)

# end in 2017
end17_mask = rdf2.index <= pd.datetime(2017,12,31)

rdf2 = rdf2[(start73_mask & end17_mask)]
rdf2.head(12)

# convert to spread indices to remove Tsy component
rdf2['HY Spread'] = rdf2['HY Bonds'] - rdf2['Intermediate US Tsy']
rdf2['EM Spread'] = rdf2['EM Bonds'] - rdf2['Intermediate US Tsy']

signal_cols_dict = {
    'Forecast' : ('INDPRO Forecast Surprise Signal','CPI Forecast Surprise Signal'),
    'Trend' : ('INDPRO Trend Surprise Signal','CPI Trend Surprise Signal'),
    'Combined' : ('INDPRO Combined Surprise Signal','CPI Combined Surprise Signal')
}

cash_ticker = 'RF'

tickers = np.array(
    [
        'US Tsy',
        'US Tips',
        'US Equities',
        'Commodities',
        'Gold',
        'HY Spread',
        'EM Spread'
    ]
)

macro_environments = np.array([
                'All',
                'Growth Up',
                'Growth Down',
                'Inflation Up',
                'Inflation Down',
                'Growth Up + Inflation Up',
                'Growth Up + Inflation Down',
                'Growth Down + Inflation Up',
                'Growth Down + Inflation Down'
            ])

macro_envs_ordered = np.array([
    'Growth Down + Inflation Down',
    'Growth Up + Inflation Down',
    'Growth Down + Inflation Up',
    'Growth Up + Inflation Up'
])

def create_macro_environment_masks(
    rdf:pd.DataFrame,
    signal_type:str,
    sc_dict:dict=signal_cols_dict
) -> pd.DataFrame:
    
    mdf = pd.DataFrame(index=rdf.index)
    
    mdf['Growth Up'] = rdf[sc_dict[signal_type][0]] > 0.9
    mdf['Growth Down'] = rdf[sc_dict[signal_type][0]] < 0.1
    mdf['Inflation Up'] = rdf[sc_dict[signal_type][1]] > 0.9
    mdf['Inflation Down'] = rdf[sc_dict[signal_type][1]] < 0.1
    
    mdf['Growth Up + Inflation Up'] = mdf['Growth Up'] & mdf['Inflation Up']
    mdf['Growth Up + Inflation Down'] = mdf['Growth Up'] & mdf['Inflation Down']
    mdf['Growth Down + Inflation Up'] = mdf['Growth Down'] & mdf['Inflation Up']
    mdf['Growth Down + Inflation Down'] = mdf['Growth Down'] & mdf['Inflation Down']
    return mdf
    
def print_macro_environment_masks(mdf:pd.DataFrame) -> None:
    print("Rising Growth number of Months: %i" % np.nansum(mdf['Growth Up']))
    print("Rising Growth %% of Months: %.2f%%" % (np.nanmean(mdf['Growth Up'])*100))
    print("Falling Growth number of Months: %i" % np.nansum(mdf['Growth Down']))
    print("Falling Growth %% of Months: %.2f%%" % (np.nanmean(mdf['Growth Down'])*100))
    print("Rising Inflation number of Months: %i" % np.nansum(mdf['Inflation Up']))
    print("Rising Inflation %% of Months: %.2f%%" % (np.nanmean(mdf['Inflation Up'])*100))
    print("Falling Inflation number of Months: %i" % np.nansum(mdf['Inflation Down']))
    print("Falling Inflation %% of Months: %.2f%%" % (np.nanmean(mdf['Inflation Down'])*100))
    print("Rising Growth + Rising Inflation number of Months: %i" % np.nansum(mdf['Growth Up + Inflation Up']))
    print("Rising Growth + Rising Inflation %% of Months: %.2f%%" % (np.nanmean(mdf['Growth Up + Inflation Up'])*100))
    print("Rising Growth + Falling Inflation number of Months: %i" % np.nansum(mdf['Growth Up + Inflation Down']))
    print("Rising Growth + Falling Inflation %% of Months: %.2f%%" % (np.nanmean(mdf['Growth Up + Inflation Down'])*100))
    print("Falling Growth + Rising Inflation number of Months: %i" % np.nansum(mdf['Growth Down + Inflation Up']))
    print("Falling Growth + Rising Inflation %% of Months: %.2f%%" % (np.nanmean(mdf['Growth Down + Inflation Up'])*100))
    print("Falling Growth + Falling Inflation number of Months: %i" % np.nansum(mdf['Growth Down + Inflation Down']))
    print("Falling Growth + Falling Inflation %% of Months: %.2f%%" % (np.nanmean(mdf['Growth Down + Inflation Down'])*100))
    return

def calculate_sharpes_in_macro_envs(
    rdf:pd.DataFrame,
    mdf:pd.DataFrame,
    macro_environments:np.ndarray,
    tickers:np.ndarray,
    cash_ticker:str=cash_ticker) -> pd.DataFrame:
    
    resdf = pd.DataFrame(
        np.zeros((len(tickers),len(macro_environments))),
        index=tickers,
        columns=macro_environments
    )
    
    for t in tickers:
        #no need to subtract the risk free in self financing portfolios
        #this assumes going long HY and short duration matched treasuries is dollar neutral
        if t in ("HY Spread","EMD Spread"):
            for m in macro_environments:
                if m == 'All':
                    resdf.loc[t,m] = (
                        np.mean(rdf.loc[:,t])/np.std(rdf.loc[:,t],ddof=1)*np.sqrt(12)
                    )
                else:
                    mask = mdf[m]
                    resdf.loc[t,m] = (
                        np.mean(rdf.loc[mask,t])/np.std(rdf.loc[mask,t],ddof=1)*np.sqrt(12)
                    )
        else:
            for m in macro_environments:
                if m == 'All':
                    resdf.loc[t,m] = (
                        np.mean(rdf.loc[:,t] - rdf.loc[:,cash_ticker])/np.std(rdf.loc[:,t],ddof=1)*np.sqrt(12)
                    )
                else:
                    mask = mdf[m]
                    resdf.loc[t,m] = (
                        np.mean(rdf.loc[mask,t] - rdf.loc[mask,cash_ticker])/np.std(rdf.loc[mask,t])*np.sqrt(12)
                    )
    return resdf

signal_type = 'Forecast'
print(signal_type)

mdf_fc = create_macro_environment_masks(rdf2,signal_type,sc_dict=signal_cols_dict)
print_macro_environment_masks(mdf_fc)

signal_type = 'Trend'
print()
print(signal_type)

mdf_trend = create_macro_environment_masks(rdf2,signal_type,sc_dict=signal_cols_dict)
print_macro_environment_masks(mdf_trend)

signal_type = 'Combined'
print()
print(signal_type)

mdf_combined = create_macro_environment_masks(rdf2,signal_type,sc_dict=signal_cols_dict)
print_macro_environment_masks(mdf_combined)

mdf_fc.head()

sdf_fc = calculate_sharpes_in_macro_envs(
    rdf2,
    mdf_fc,
    macro_environments,
    tickers,
    cash_ticker
)
print("Sharpes using Forecasts to determine economic surprises")
sdf_fc

sdf_trend = calculate_sharpes_in_macro_envs(
    rdf2,
    mdf_trend,
    macro_environments,
    tickers,
    cash_ticker
)
print("Sharpes using Trend to determine economic surprises")
sdf_trend

sdf_combined = calculate_sharpes_in_macro_envs(
    rdf2,
    mdf_combined,
    macro_environments,
    tickers,
    cash_ticker
)
print("Sharpes using a combination of Forecasts and Trend to determine economic surprises")
sdf_combined

def plot_sharpes_across_macroenvironments(
    resdf:pd.DataFrame,
    macroenvironments:list,
    type_of_macro_forecast=None
):
    colors = ['blue','gray','red','orange','gold','purple','pink']
    for i,t in enumerate(resdf.index):
        plt.scatter(
            macroenvironments,
            resdf.loc[t,macroenvironments],
            label=t,
            color=colors[i]
        )
        plt.axhline(y=resdf.loc[t,'All'],color=colors[i])
    plt.axhline(y=0,color='gray',linestyle='dashed')
    plt.title("Sharpes Across Macroenvironments")
    plt.legend()
    
    if type_of_macro_forecast is not None:
        plt.annotate(
            "*Macroenvironment Surprises calculated using %s" % type_of_macro_forecast,
            (0, 0),
            (0, -40),
            xycoords='axes fraction',
            textcoords='offset points',
            va='top'
        )
    
    plt.show()

plot_sharpes_across_macroenvironments(
    sdf_fc,
    macro_environments[5:9],
    "Survey of Professional Forecasters Forecast"
)

plot_sharpes_across_macroenvironments(
    sdf_trend,
    macro_environments[5:9],
    "Trend"
)

plot_sharpes_across_macroenvironments(
    sdf_combined,
    macro_environments[5:9],
    "Trend"
)

def plot_relationship(
    rdf:pd.DataFrame,
    ticker:str,
    g_or_i:str,
    fc_or_trend:str
):
    type_of_macro_surprise = 'CPI'
    if g_or_i == 'Growth':
        type_of_macro_surprise = 'INDPRO'
    
    type_of_forecast = 'Forecast'
    if fc_or_trend == 'Trend':
        type_of_forecast = 'Trend'
    
    surprise_col_name = (
        "%s %s Surprise Normalized" % 
        (
            type_of_macro_surprise,
            type_of_forecast
        )
    )
    
    tempdf = rdf.loc[:,[
        ticker,
        surprise_col_name
    ]].dropna()
    
    X = sm.add_constant(tempdf[surprise_col_name])
    y = tempdf[ticker]
    model = sm.OLS(y, X)
    results = model.fit()
    print("Beta of %s to %s: %0.4f" %
         (
             ticker,
             surprise_col_name,
             results.params[1]
         )
    )
    print("T statistic of Beta of %s to %s: %0.4f" %
         (
             ticker,
             surprise_col_name,
             results.tvalues[1]
         )
    )
    
    plt.scatter(
        tempdf[surprise_col_name],
        tempdf[ticker]
    )
    plt.plot(
        tempdf[surprise_col_name],
        results.fittedvalues,
        color='red',
        label='OLS'
    )
    plt.xlabel(
        "%s Suprise using %s" % 
        (
            g_or_i,
            type_of_forecast
        )
    )
    plt.ylabel('Return')
    plt.legend()
    plt.show()
    
    return

for i,t in enumerate(tickers):
    print(t)
    plot_relationship(
        rdf2,
        t,
        'Growth',
        'Forecast'
    )

    plot_relationship(
        rdf2,
        t,
        'Inflation',
        'Forecast'
    )
    print('\n\n\n\n\n')

