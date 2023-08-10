import pandas as pd
import censusdata
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 2)

censusdata.search('acs5', '2015', 'label', 'unemploy')[160:170]

censusdata.search('acs5', '2015', 'concept', 'education')[730:790]

censusdata.printtable(censusdata.censustable('acs5', '2015', 'B23025'))

censusdata.printtable(censusdata.censustable('acs5', '2015', 'B15003'))

censusdata.geographies(censusdata.censusgeo([('state', '*')]), 'acs5', '2015')

censusdata.geographies(censusdata.censusgeo([('state', '17'), ('county', '*')]), 'acs5', '2015')

cookbg = censusdata.download('acs5', '2015',
                             censusdata.censusgeo([('state', '17'), ('county', '031'), ('block group', '*')]),
                             ['B23025_003E', 'B23025_005E', 'B15003_001E', 'B15003_002E', 'B15003_003E',
                              'B15003_004E', 'B15003_005E', 'B15003_006E', 'B15003_007E', 'B15003_008E',
                              'B15003_009E', 'B15003_010E', 'B15003_011E', 'B15003_012E', 'B15003_013E',
                              'B15003_014E', 'B15003_015E', 'B15003_016E'])
cookbg['percent_unemployed'] = cookbg.B23025_005E / cookbg.B23025_003E * 100
cookbg['percent_nohs'] = (cookbg.B15003_002E + cookbg.B15003_003E + cookbg.B15003_004E
                          + cookbg.B15003_005E + cookbg.B15003_006E + cookbg.B15003_007E + cookbg.B15003_008E
                          + cookbg.B15003_009E + cookbg.B15003_010E + cookbg.B15003_011E + cookbg.B15003_012E
                          + cookbg.B15003_013E + cookbg.B15003_014E +
                          cookbg.B15003_015E + cookbg.B15003_016E) / cookbg.B15003_001E * 100
cookbg = cookbg[['percent_unemployed', 'percent_nohs']]
cookbg.describe()

cookbg.sort_values('percent_unemployed', ascending=False).head(30)

cookbg.corr()

