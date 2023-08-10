import calcbench as cb
import pandas as pd
pd.options.display.max_rows = 1000
get_ipython().magic('pylab inline')

z_score_metrics = ['CurrentAssets',
                   'CurrentLiabilities', 
                   'Assets', 
                   'RetainedEarnings', 
                   'EBIT', 
                   'MarketCapAtEndOfPeriod',
                   'Liabilities',
                   'Revenue']

SIC_codes = {
    "Oil And Gas Extraction" : 1300,
    "Metal Mining" : 1000,
    "Coal Mining" : 1200,
    "Mining Nonmetallic Minerals" : 1400,
}

def get_z_score_inputs(tickers):
    z_score_data = cb.normalized_dataframe(company_identifiers=list(tickers), 
                                           metrics=z_score_metrics, 
                                           start_year=2008, start_period=0, 
                                           end_year=2015, end_period=0)
    return z_score_data

def peer_group_z_score(peer_group):
    peer_group = peer_group[(peer_group.ticker != 'GMC') & (peer_group.ticker != 'PPI')] #GMC's marketvalue is off
    z_score_data = get_z_score_inputs(peer_group.ticker)
    aggregate_data = z_score_data.sum(level=[0], axis=1)
    return compute_z_score(aggregate_data), z_score_data

def compute_z_score(inputs):
    #from https://github.com/calcbench/notebooks/blob/master/z-score.ipynb
    working_capital = inputs['CurrentAssets'] - inputs['CurrentLiabilities']
    

    z_score = (1.2 * (working_capital / inputs['Assets']) + 
              1.4 * (inputs['RetainedEarnings'] / inputs['Assets']) +
              3.3 * (inputs['EBIT'] / inputs['Assets']) +
              0.6 * (inputs['MarketCapAtEndOfPeriod'] / inputs['Liabilities']) +
              .99 * (inputs['Revenue'] / inputs['Assets']))
    
    return z_score

peer_groups = [(industry, cb.companies(SIC_codes=[SIC_code])) for industry, SIC_code in SIC_codes.items()]
sp500 = cb.companies(index="SP500")
sp500_no_financials = sp500[sp500.sic_code & ((sp500.sic_code < 6000) | (sp500.sic_code >= 7000))] # There is a different z-score formulas for financials.
peer_groups.append(("SP500 (no financials)", sp500_no_financials))
industry_z_scores = [(industry, peer_group_z_score(peer_group)[0]) for industry, peer_group in peer_groups]
z_scores = pd.DataFrame.from_items(industry_z_scores)

coal_peer_group = cb.companies(SIC_codes=[1200])

z_score_inputs = get_z_score_inputs(coal_peer_group.ticker)

compute_z_score(z_score_inputs)

z_scores.plot(figsize=(18, 10))

z_score, data = peer_group_z_score(peer_groups[0][1])

compute_z_score(data.swaplevel(0, 1, 1).ACI)



