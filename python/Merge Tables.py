# Click the Blue Plane to preview this notebook as a CrossCompute Tool
a_table_path = 'usa-temperature-by-state.csv'
b_table_path = 'usa-precipitation-by-state.csv'
key_column_name = 'State'
target_folder = '/tmp'

import pandas as pd
a_table = pd.read_csv(a_table_path)
b_table = pd.read_csv(b_table_path)
c_table = pd.merge(a_table, b_table, on=key_column_name)

from os.path import join
target_path = join(target_folder, 'table.csv')
c_table.to_csv(target_path, index=False)
print('c_table_path = ' + target_path)

