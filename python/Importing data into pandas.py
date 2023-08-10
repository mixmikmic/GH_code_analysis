import pandas as pd

csv_df = pd.read_csv('../data/mlb.csv')

csv_df.head()

csvi_df = pd.read_csv('http://electionresults.sos.ne.gov/resultsCSV.aspx?text=All')

csvi_df.head()

xl_df = pd.read_excel('../data/homicides2014.xlsx', sheet_name='Murders')

xl_df.head()

test_data = [
    {'name': 'Cody Winchester', 'job': 'Training director', 'location': 'Colorado Springs, CO'},
    {'name': 'Matt Wynn', 'job': 'Data reporter', 'location': 'Omaha, NE'},
    {'name': 'Guy Fieri', 'job': 'Gourmand', 'location': 'Flavortown'},
    {'name': 'Sarah Huckabee Sanders', 'job': 'Spokeswoman', 'location': 'Washington, D.C.'}
]

py_df = pd.DataFrame(test_data)

py_df.head()

html_df = pd.read_html('https://www.tdcj.state.tx.us/death_row/dr_media_witness_list.html',
                       flavor='bs4',
                       attrs={'class': 'tdcj_table'},
                       header=0)[0]

html_df.head()

json_df = pd.read_json('https://data.sunshinecoast.qld.gov.au/resource/44qj-t4fr.json')

json_df.head()

