import calcbench as cb #https://github.com/calcbench/python_api_client
get_ipython().magic('pylab inline')

tickers = cb.tickers(index="SP500")

quarterly_data = cb.normalized_data(tickers, ['revenue', 'accountsreceivable'], 2008, 1, 2014, 3)
quarter_corr = quarterly_data['accountsreceivable'].corrwith(quarterly_data['revenue'])
quarter_corr.hist(bins=50, figsize=(15, 10))

quarterly_data.columns

annual_data = cb.normalized_data(tickers, ['revenue', 'accountsreceivable'], 2008, 0, 2014, 0)
annual_corr = annual_data['accountsreceivable'].corrwith(annual_data['revenue'])
annual_corr.hist(bins=50, figsize=(15, 10))

quarterly_data['revenue']['YUM'].plot(figsize=(10, 6))
quarterly_data['accountsreceivable']['YUM'].plot(secondary_y=True)

annual_data['revenue']['YUM'].plot(figsize=(10, 6))
annual_data['accountsreceivable']['YUM'].plot(secondary_y=True)

