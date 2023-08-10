# import modules
import pandas as pd
import patsy

# Create dataframe
raw_data = {'countrycode': [1,2,3,2,1]}
df = pd.DataFrame(raw_data, columns=['countrycode'])
df

# Convert the countrycode variable into three binary variables
patsy.dmatrix('C(countrycode)-1', df, return_type='dataframe')

