import pandas as pd
import censusdata
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 2)

county65plus = censusdata.download('acs5', '2015', censusdata.censusgeo([('county', '*')]),
                                   ['B01001_001E', 'B01001_020E', 'B01001_021E', 'B01001_022E', 'B01001_023E',
                                    'B01001_024E', 'B01001_025E', 'B01001_044E', 'B01001_045E', 'B01001_046E',
                                    'B01001_047E', 'B01001_048E', 'B01001_049E'])
county65plus.describe()

county65plus['percent_65plus'] = (county65plus.B01001_020E + county65plus.B01001_021E + county65plus.B01001_022E
                                  + county65plus.B01001_023E + county65plus.B01001_024E + county65plus.B01001_025E
                                  + county65plus.B01001_044E + county65plus.B01001_045E + county65plus.B01001_046E
                                  + county65plus.B01001_047E + county65plus.B01001_048E
                                  + county65plus.B01001_049E) / county65plus.B01001_001E * 100
county65plus = county65plus[['B01001_001E', 'percent_65plus']]
county65plus = county65plus.rename(columns={'B01001_001E': 'population_size'})
county65plus.describe()

county65plus.sort_values('percent_65plus', ascending=False, inplace=True)
county65plus.head(30)

censusdata.search('acs5', '2015', 'label', '65', tabletype='profile')[-25:]

censusdata.printtable(censusdata.censustable('acs5', '2015', 'DP05'))

county65plus = censusdata.download('acs5', '2015', censusdata.censusgeo([('county', '*')]),
                                   ['DP05_0001E', 'DP05_0014PE', 'DP05_0015PE', 'DP05_0016PE',],
                                   tabletype='profile')
county65plus.describe()

county65plus['percent_65plus'] = (county65plus['DP05_0014PE'] + county65plus['DP05_0015PE']
                                  + county65plus['DP05_0016PE'])
county65plus = county65plus[['DP05_0001E', 'percent_65plus']]
county65plus = county65plus.rename(columns={'DP05_0001E': 'population_size'})
county65plus.describe()

county65plus.sort_values('percent_65plus', ascending=False, inplace=True)
county65plus.head(30)

censusdata.exportcsv('county65plus.csv', county65plus)

