import pandas as pd
import numpy as np
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

old = pd.read_csv('../../data/scraped/scraped_complaints_3_25.csv')

new = pd.read_csv('../../data/scraped/scraped_complaints_4_14.csv')

new.count()[0]

old.count()[0]

merged = new.merge(old,how = 'left',on='abuse_number')

merged2 = new.merge(old,how = 'right',on='abuse_number')

merged[merged['online_incident_date_y'].isnull()].count()[0]



