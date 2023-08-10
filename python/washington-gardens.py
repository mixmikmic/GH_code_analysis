import pandas as pd
import numpy as np
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
df = pd.read_csv('../../data/processed/complaints-3-25-scrape.csv')

move_in_date = '2015-05-01'

df[(df['facility_id']=='50R382') & (df['incident_date']<move_in_date)].count()[0]

