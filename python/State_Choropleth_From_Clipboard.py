import pandas as pd

df = pd.read_clipboard().applymap(lambda x:str(x).replace('%',''))

df.head()

df.dtypes

df.value = df.value.astype(float)

df.dtypes

state_code_2_name = {
    'AK':'alaska',
    'AL':'alabama',
    'AR':'arkansas',
    'AZ':'arizona',
    'CA':'california',
    'CO':'colorado',
    'CT':'connecticut',
    'DC':'district of colunbia',
    'DE':'delaware',
    'FL':'florida',
    'GA':'georgia',
    'HI':'hawaii',
    'IA':'iowa',
    'ID':'idaho',
    'IL':'illinois',
    'IN':'indiana',
    'KS':'kansas',
    'KY':'kentucky',
    'LA':'louisiana',
    'MA':'massachusetts',
    'MD':'maryland',
    'ME':'maine',
    'MI':'michigan',
    'MN':'minnesota',
    'MO':'missouri',
    'MS':'mississippi',
    'MT':'montana',
    'NC':'north carolina',
    'ND':'north dakota',
    'NE':'nebraska',
    'NH':'new hampshire',
    'NJ':'new jersey',
    'NM':'new mexico',
    'NV':'nevada',
    'NY':'new york',
    'OH':'ohio',
    'OK':'oklahoma',
    'OR':'oregon',
    'PA':'pennsylvania',
    'RI':'rhode island',
    'SC':'south carolina',
    'SD':'south dakota',
    'TN':'tennessee',
    'TX':'texas',
    'UT':'utah',
    'VA':'virginia',
    'VT':'vermont',
    'WA':'washington',
    'WI':'wisconsin',
    'WV':'west virginia',
    'WY':'wyoming'
}

df.region = df.region.map(state_code_2_name)

df

get_ipython().run_line_magic('load_ext', 'rpy2.ipython')

get_ipython().run_line_magic('R', '-i df')

get_ipython().run_cell_magic('R', '-w 800 -h 800 -u px', 'library(choroplethr)\nchoropleth_data = as.data.frame(df)\nstate_choropleth(choropleth_data,\n                 title      = "State Defect Rates (%)",\n                 legend     = "Defect Rate %",\n                 num_colors = 1)')

