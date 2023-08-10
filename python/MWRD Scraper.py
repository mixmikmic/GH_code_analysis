import requests
from bs4 import BeautifulSoup
from robobrowser import RoboBrowser
import robobrowser
import csv 

mwrd_url = "http://apps.mwrd.org/csoreports/CSO_Synopisis_Report"
mwrd_page = requests.get(mwrd_url)
print(mwrd_page.content)

mwrd_soup = BeautifulSoup(mwrd_page.content, 'html.parser')
input_vals = mwrd_soup.findAll('input')
for input_val in input_vals:
    print(input_val['name'])

s = requests.Session()
asp_headers = {
#     'Host': 'apps.mwrd.org',
#     'Origin': 'http://apps.mwrd.org',
#     'Referer': 'http://apps.mwrd.org/csoreports/CSO_Synopisis_Report',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36',
    'Connection': 'keep-alive',
    'Accept-Encoding': 'gzip, deflate'
}

mwrd_url = "http://apps.mwrd.org/csoreports/CSO_Synopisis_Report"
mwrd_page = s.get(mwrd_url)
mwrd_soup = BeautifulSoup(mwrd_page.content, 'html.parser')

all_inputs = mwrd_soup.find_all('input')

payload = {}

for inval in all_inputs:
    if inval.get('value'):
        payload[inval['name']] = inval['value']
    else:
        payload[inval['name']] = ''

payload['__EVENTTARGET'] = 'bttSearch'
payload['txtStartDateSearch'] = '08/13/2016'
payload['txtEndDateSearch'] = '08/27/2016'

payload['ReportViewer1$ctl11'] = 'standards'
#payload['ddlCSODates'] = '0'
#payload['ReportViewer1$ctl07$img'] = ''
# payload['__EVENTARGUMENT'] = ''
# payload['ReportViewer1$ctl09$VisibilityState$ctl00'] = 'ReportPage'

drop_keys = [
    'bttSearchDay',
    'bttResetSearch2',
    'bttSearch',
    'bttResetSearch',
    'ReportViewer1$ToggleParam$img'
]

for key in drop_keys:
    if payload.get(key):
        payload.pop(key)
# print('cookies: {} \n\n'.format(s.cookies))

for key in payload.keys():
    print(key)
# print(payload['ReportViewer1$ctl11'])
# print(payload['ReportViewer1$ctl09$VisibilityState$ctl00'])
mwrd_post_response = requests.post(mwrd_url, data=payload, headers=asp_headers, cookies=s.cookies)
mwrd_response_soup = BeautifulSoup(mwrd_post_response.content, 'html.parser')
# mwrd_response_soup = BeautifulSoup(resp.content, 'html.parser')
# print(mwrd_response_soup.find(id='ReportViewer1_ctl09'))

# Table with results has 9 cols, otherwise difficult to access
main_table_div = mwrd_response_soup.select("table[cols=9]")
# Get all table rows, other than the first which doesn't have the valign attribute, only searching in this subgroup
table_rows = main_table_div[0].select("tr[valign='top']")
# Create an empty list to add values to
table_list = []
# Go through each row, select the div in it, then grab the text from each div, add it to the row for the csv
for row in table_rows:
    row_cells = row.select('div')
    row_list = []
    # Get the text with the .text soup attribute, use .strip() to remove unnecessary whitespaces
    for cell in row_cells:
        row_list.append(cell.text.strip())
    table_list.append(row_list)

print(table_list)

import csv 

with open('example_mwrd.csv', 'w') as mwrd_csv:
    mwrd_writer = csv.writer(mwrd_csv)
    for row in table_list:
        mwrd_writer.writerow(row)

# Create RoboBrowser, make sure to set parser to avoid errors later
browser = RoboBrowser(
    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36',
    parser='html.parser'
)
# Open a session
browser.open('http://apps.mwrd.org/csoreports/CSO_Synopisis_Report')
# Identify the form by action, manually set important values
form = browser.get_form(action='./CSO_Synopisis_Report')
form['txtStartDateSearch'].value = '08/13/2016'
form['txtEndDateSearch'].value = '08/27/2016'
form['__EVENTTARGET'].value = 'bttSearch'
form['ReportViewer1$ctl11'].value = 'standards'
# form['ReportViewer1$ctl09$VisibilityState$ctl00'].value = 'ReportPage'

drop_keys = [
    'bttSearchDay',
    'bttResetSearch2',
    'bttSearch',
    'bttResetSearch',
    'ReportViewer1$ToggleParam$img'
]

# Remove extra keys
for key in drop_keys:
    form.fields.pop(key)

# Drop keys identified by printing all keys, then comparing to those posted in browser in Dev Tools
# for key in form.keys():
#     print('{}: {}'.format(key, form[key].value))

# Submit the form
browser.submit_form(form)

# Table with results has 9 cols, otherwise difficult to access
main_table_div = browser.select("table[cols=9]")
# Get all table rows, other than the first which doesn't have the valign attribute, only searching in this subgroup
table_rows = main_table_div[0].select("tr[valign='top']")
# Create an empty list to add values to
table_list = []
# Go through each row, select the div in it, then grab the text from each div, add it to the row for the csv
for row in table_rows:
    row_cells = row.select('div')
    row_list = []
    # Get the text with the .text soup attribute, use .strip() to remove unnecessary whitespaces
    for cell in row_cells:
        row_list.append(cell.text.strip())
    table_list.append(row_list)

print(table_list)



