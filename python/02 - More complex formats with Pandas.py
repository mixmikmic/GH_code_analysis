import pandas as pd

import pdftables

get_ipython().magic('pinfo pdftables.get_tables')

get_ipython().magic('pinfo pdftables.page_to_tables')

my_pdf = open('../../data/WEF_GlobalCompetitivenessReport_2014-15.pdf', 'rb')

chart_page = pdftables.get_pdf_page(my_pdf, 29)

chart_page

table = pdftables.page_to_tables(chart_page)

table

table[0]

titles = zip(table[0][0], table[0][1])[:5]

titles

titles = [''.join([title[0], title[1]]) for title in titles]
print(titles)

all_rows = []
for row_data in table[0][2:]:
    all_rows.extend([row_data[:5], row_data[5:]])

all_rows

df = pd.DataFrame(all_rows, columns=titles)

df.head()

new_chart_page = pdftables.get_pdf_page(my_pdf, 30)

table = pdftables.page_to_tables(new_chart_page)

table[0]



