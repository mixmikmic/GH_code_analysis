# load the rpy2 extension
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')

import pandas as pd

d = {'STATE_NAME':[
     'alabama',
     'alaska',
     'arkansas',
     'arizona',
     'california',
     'colorado',
     'connecticut',
     'delaware',
     'florida',
     'georgia',
     'hawaii',
     'iowa',
     'idaho',
     'illinois',
     'indiana',
     'kansas',
     'kentucky',
     'louisiana',
     'maine',
     'massachusetts',
     'maryland',
     'michigan',
     'minnesota',
     'missouri',
     'mississippi',
     'montana',
     'north carolina',
     'north dakota',
     'nebraska',
     'new hampshire',
     'new jersey',
     'new mexico',
     'nevada',
     'new york',
     'ohio',
     'oklahoma',
     'oregon',
     'pennsylvania',
     'rhode island',
     'south carolina',
     'south dakota',
     'tennessee',
     'texas',
     'utah',
     'virginia',
     'vermont',
     'washington',
     'wisconsin',
     'west virginia',
     'wyoming'],
     
     'STATE_CODE':[
     'AL',
     'AK',
     'AR',
     'AZ',
     'CA',
     'CO',
     'CT',
     'DE',
     'FL',
     'GA',
     'HI',
     'IA',
     'ID',
     'IL',
     'IN',
     'KS',
     'KY',
     'LA',
     'ME',
     'MA',
     'MD',
     'MI',
     'MN',
     'MO',
     'MS',
     'MT',
     'NC',
     'ND',
     'NE',
     'NH',
     'NJ',
     'NM',
     'NV',
     'NY',
     'OH',
     'OK',
     'OR',
     'PA',
     'RI',
     'SC',
     'SD',
     'TN',
     'TX',
     'UT',
     'VA',
     'VT',
     'WA',
     'WI',
     'WV',
     'WY']
    }

state_code_table = pd.DataFrame(d)
state_code_table

df = pd.read_excel(r'D:\jupyter\miar\92Ki\HNA-MIAR-150429-01\13G_RDX_TPMS_Sensor_Leaks.xlsx', 'State_Defect_Rates')

df.head()

states_data = pd.merge(state_code_table, df, how='left', left_on='STATE_CODE', right_on='STATE_CODE')

states_data.head()

states_data = states_data.fillna(value=0)
states_data

states_data.drop('STATE_CODE', axis='columns', inplace=True)
states_data

get_ipython().run_line_magic('R', '-i states_data')

get_ipython().run_cell_magic('R', '', 'df <- as.data.frame(states_data)\nlibrary(choroplethr)\n\nchoropleths = list()\nfor (i in 2:ncol(states_data)) {\n  df           = states_data[, c(1, i)]\n  colnames(df) = c("region", "value")\n  title        = paste0(colnames(states_data)[i])\n  choropleths[[i-1]] = state_choropleth(df, title=title, legend="Defect Rate (%)", buckets=7)\n}')

get_ipython().run_cell_magic('R', '-w 900 -h 500 -u px', 'choropleths[3]')

