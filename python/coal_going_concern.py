import calcbench as cb
import itertools
from operator import itemgetter

coal_companies = cb.tickers(SIC_codes=[1200])

search_term = '"going concern"'  #Use Lucene query syntax, http://lucene.apache.org/core/2_9_4/queryparsersyntax.html

for year in range(2010, 2016):
    print("YEAR ", year)
    going_concern_footnotes = cb.document_search(company_identifiers=coal_companies, 
                                             full_text_search_term=search_term, 
                                             year=year, period=0)
    going_concern_footnotes = sorted(going_concern_footnotes, key=itemgetter('entity_name'))
    for company_name, _ in itertools.groupby(going_concern_footnotes, itemgetter('entity_name')):
        print(company_name)



