import os
import sys

sys.path.append("../")

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import pandas
import cubesext

CSV_FILE = 'fakedata-product.csv'

df = pandas.read_csv(CSV_FILE)
df['product_price'] = df['product_price'].apply(lambda x: float(x.replace(",", ".")))
#df['product_price'] = df['product_price'].astype(float)
df[:10]

(db_url, model_path) = cubesext.pandas2cubes(df)
print(db_url, model_path)

cubesext.cubes_serve(db_url, model_path)

cubesext.cubesviewer_jupyter()

pandas.DataFrame([[1, 2, "Yes"], [3, 4, "No"]])

