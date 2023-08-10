import calcbench as cb
import itertools
from operator import itemgetter
from IPython.core.display import display, HTML
import csv

motion_picture_companies = cb.tickers(SIC_codes=[7800])

revenue_recognition_results = []
for year in range(2009, 2016):
    revenue_recognition_results.extend(cb.document_search(company_identifiers=motion_picture_companies,
                                             block_tag_name="RevenueRecognitionPolicyTextBlock",                                              
                                             year=year, 
                                             period=0))

for result in revenue_recognition_results:
    display(HTML('<h1>{entity_name} - Fiscal Year {fiscal_year}</h1>'.format(**result) + cb.tag_contents(accession_id=result['accession_id'], block_tag_name="RevenueRecognitionPolicyTextBlock")))

with open('cinema_revenue_recognition.csv', 'w') as f:
    writer = csv.writer(f)
    for result in revenue_recognition_results:
        contents = cb.tag_contents(accession_id=result['accession_id'], block_tag_name="RevenueRecognitionPolicyTextBlock")
        writer.writerow([result['entity_name'], result['fiscal_year'], contents])



