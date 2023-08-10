import os, sys
import decimal
from decimal import Decimal

import pandas as pd

# get the current directory and files inside 
print(os.getcwd()); print(os.listdir( os.getcwd() ));

from NISTFund import retrieve_file, scraping_allascii, init_FundConst, make_pd_alphabeticalconv_lst, make_pd_conv_lst

urladdr = retrieve_file(); print(urladdr);

os.path.isfile('./rawdata/allascii.txt') 

lines,title,src,header,rawtbl,tbl=scraping_allascii()

FundConst = pd.DataFrame(tbl, columns=header)

FundConst = init_FundConst()

DF_conv=make_pd_conv_lst()

from NISTFund import scraped_BS, NISTCONValpha
convBS = scraped_BS(NISTCONValpha)

convBS.soup.find_all("table",{"class":'texttable'})

convBS = scraped_BS("https://www.nist.gov/physical-measurement-laboratory/nist-guide-si-appendix-b8")

convBS.soup.find_all("div",{"class":"table-inner"})

convBS.soup.find_all("table"); 
print(len( convBS.soup.find_all("table")) )  # there are 26 letters in the alphabet; NIST has entries for 21 of them

convBS.convtbls = convBS.soup.find_all("table")
convdata=[]
convdata2=[]
headers = convBS.convtbls[0].find_all('tr')[1].find_all('th')
headers = [ele.text.replace(' ','') for ele in headers]

headers

for tbl in convBS.convtbls:
    for row in tbl.find_all('tr'):
        if row.find_all('td') != []:
            if row.text != '':
                rowsplit = row.text.replace("\n",'',1).split('\n')
                try:
                    rowsplit = [pt.replace(u'\xa0',u' ').strip() for pt in rowsplit]
                except UnicodeDecodeError as err:
                    print rowsplit
                    Break
                    raise err
                convdata.append( rowsplit )
                if len(row.find_all('td')) == (len(headers)+1):
                    convdata2.append( row.find_all('td'))

print(len(convdata));
print(len(convdata2))

convdata3 = []
for row in convdata2:
    rowout = []
    rowout.append( row[0].text.strip())
    rowout.append( row[1].text.strip())
    value = (row[2].text+row[3].text).strip().replace(u'\xa0',' ').replace(u'\n',' ').replace(' ','')
        
    rowout.append(Decimal( value ))
    convdata3.append(rowout)

print(len(convdata3))

pd.DataFrame(convdata3,columns=headers).head()

print(len(convdata))

convBS = scraped_BS("https://www.nist.gov/pml/nist-guide-si-appendix-b9-factors-units-listed-kind-quantity-or-field-science")

convBS.convtbls = convBS.soup.find_all("table")
print(len(convBS.convtbls))

headers = convBS.convtbls[1].find_all('tr')[1].find_all('th')
headers = [ele.text.replace(' ','') for ele in headers]

headers

convBS.convtbls[1].find_all('tr')

for rows in convBS.convtbls[1].find_all('tr'):
    print rows.find_all('td')

for row in convBS.convtbls[1].find_all('tr'):
    if row.find_all('td') != []:
        if row.text != '':
            rowsplit = row.text.split('\n')
#            print rowsplit
            if u'' in rowsplit:
                rowsplit.remove(u'')
            print rowsplit

test_convdata=[]
field_of_science = ""
for row in convBS.convtbls[1].find_all('tr'):
    if row.find_all('td') != []:
        if row.text != '':
            rowsplit = row.text.split('\n')
            if u'' in rowsplit:
                rowsplit.remove(u'')
            if len(rowsplit) is 1:
                field_of_science = rowsplit[0]
                print field_of_science
            elif field_of_science is not "":
                rowsplit.append(field_of_science)
                print rowsplit
#            print len(row.find_all('td')) 
                if len(row.find_all('td')) is (len(headers)+1):
                    test_convdata.append( rowsplit )

test_convdata

from NISTFund import make_conv_lst

test_conv=make_conv_lst()

test_conv[2]

DF_alphaconv=make_pd_alphabeticalconv_lst()

DF_alphaconv.head()

DF_conv= make_pd_conv_lst()

print( DF_conv.head() )
DF_conv.describe()

FundConst = init_FundConst()

conv = pd.read_pickle('./rawdata/DF_conv')
alphaconv = pd.read_pickle('./rawdata/DF_alphabeticalconv')

alphaconv



