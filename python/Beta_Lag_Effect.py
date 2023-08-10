import wmcm
import wmcm.blfunctions as wmbl

import pandas as pd
import numpy as np
import pickle
import seaborn as sns
get_ipython().magic('matplotlib inline')

# Single stock Universe object instance created with LULU

sstock = wmcm.Universe(['LULU'], 'SPY', start='2010-11-01', end='2015-12-31', earn=False)

sstock = wmbl.run_analysis(sstock, excess=True)

sstock['LULU'].model_sr.summary()

sstock['LULU'].model_mr.summary()

sstock['LULU'].model_exc_sr.summary()

sstock['LULU'].model_exc_mr.summary()

wmbl.backtest_stock(sstock, 'LULU', full_results=False, verbose=True)

nyse = pd.DataFrame(finsymbols.get_nyse_symbols())
nyse['exchange'] = 'nyse'
nasdaq = pd.DataFrame(finsymbols.get_nasdaq_symbols())
nasdaq['exchange'] = 'nasdaq'
universe = pd.concat([nyse, nasdaq])
universe['symbol'] = universe['symbol'].replace('.', '-')
universe.head()

# Uncomment these lines the first time notebook run on computer. Full price data isn't uploaded to github.
# stocks = wmcm.Universe(universe['symbol'], 'SPY', start='2010-11-01', end='2015-12-31', earn=False)
# output = open('pickle/full_history_monthly.p', 'wb')
# pickle.dump(stocks, output)
# output.close()

pkl_file = open('pickle/full_history_monthly.p', 'rb')
stocks = pickle.load(pkl_file)
pkl_file.close()

stocks = wmbl.run_analysis(stocks, excess=True)

plot_data = wmbl.get_plot_data(stocks)
plot_data = plot_data.loc[plot_data['market_cap'].notnull()]

sns.distplot(plot_data['t_val'])

sns.regplot(x='market_cap', y='t_val', data=plot_data.loc[(plot_data['market_cap']<1e11) & (plot_data['market_cap']>1e10)], lowess=True)

plot_data['bic_likes_mr'] = (plot_data['mr_bic'] < plot_data['sr_bic'])
plot_data['bic_likes_mr'].mean()

plot_data['95_signif'] = (plot_data['mr_p_values'] < 0.05)
plot_data['95_signif'].mean()

plot_data['95_signif_exc'] = (plot_data['mr_p_values_exc'] < 0.05)
plot_data['95_signif_exc'].mean()

plot_data.sort_values('mr_p_values',inplace=False,ascending=True).head()

sns.regplot(x='lag_ret_cc_market', y='ret_cc', data=stocks['MAGS'].analysis_df)

sns.regplot(x='lag_ret_cc_market', y='ret_cc', data=stocks['PFBI'].analysis_df)

sns.regplot(x='lag_ret_cc_market', y='ret_cc', data=stocks['LNC'].analysis_df)

