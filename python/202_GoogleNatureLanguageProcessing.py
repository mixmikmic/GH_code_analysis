import urllib2
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

client = language.LanguageServiceClient()

import urllib2
from bs4 import BeautifulSoup
import re
import pandas as pd
from google.cloud import language
import numpy as np

filename = ['American Eagle Outfitters, Inc.August 23, 2017 9_00 AM ET.txt',
             'American Eagle Outfitters, Inc.March 1, 2017 09_00 ET.txt',
             'American Eagle Outfitters, Inc.May 17, 2017 09_00 AM ET.txt',
             'American Eagle Outfitters, Inc.November 30, 2016, 09_00 AM ET.txt',
             'Aramark Services, Inc.August 08, 2017 10_00 am ET.txt',
             'Aramark Services, Inc.February 07, 2017 10_00 am ET.txt',
             'Aramark Services, Inc.May 09, 2017 10_00 am ET.txt',
             'Aramark Services, Inc.November 14, 2017 10_00 am ET.txt',
             'Aramark Services, Inc.November 15, 2016 10_00 A.M. ET.txt',
             'Caseys General Stores, Inc.December 8, 2016 10_30 ET.txt',
             'Caseys General Stores, Inc.June 6, 2017 10_30 ET.txt',
             'Caseys General Stores, Inc.March 07, 2017, 10_30 ET.txt',
             'Caseys General Stores, Inc.September 6, 2017 10_30 AM ET.txt',
             'Caseys General Stores, Inc.September 07, 2016, 10_30 AM ET.txt',
             'Cinemark Holdings, IncAugust 04, 2017 8_30 am ET.txt',
             'Cinemark Holdings, IncFebruary 23, 2017 8_30 am ET.txt',
             'Cinemark Holdings, IncMay 03, 2017 8_30 am ET.txt',
             'Cinemark Holdings, IncNovember 03, 2017 08_30 AM ET.txt',
             'Cinemark Holdings, IncNovember 08, 2016 8_30 am ET.txt',
             'CSX CorporationOctober 17, 2017 8_30 a.m. ET.txt',
             'Deluxe CorporationApril 27, 2017, 11_00 ET.txt',
             'Deluxe CorporationOctober 26, 2017, 11_00 ET.txt',
             'Dollar Tree, Inc.August 24, 2017, 09_00 AM ET.txt',
             'Dollar Tree, Inc.November 21, 2017 09_00 AM ET.txt',
             'Dollar Tree, Inc.November 22, 2016, 09_00 AM ET.txt',
             'EchoStar CorporationMay 10, 2017 12_00 PM ET.txt',
             'EchoStar CorporationNovember 8, 2016 11_00 AM ET.txt',
             'EchoStar CorporationNovember 8, 2017 11_00 AM ET.txt',
             'FTI Consulting, Inc.April 27, 2017 20, 2017 9_00 AM ET.txt',
             'FTI Consulting, Inc.July 27, 2017 09_00 AM ET.txt',
             'FTI Consulting, Inc.October 26, 2017 09_00 AM ET.txt',
             'General Motors CompanyNovember 30, 2017 12_00 PM ET.txt',
             'Graphic Packaging Holding CompanyApril 25, 2017 10_00 AM ET.txt',
             'Graphic Packaging Holding CompanyJuly 25, 2017, 10_00 ET.txt',
             'Graphic Packaging Holding CompanyOctober 24, 2017 10_00 ET.txt',
             'Graphic Packaging Holding CompanyOctober 25, 2016 10_00 am ET.txt',
             'HNI CorporationFebruary 09, 2017, 11_00 ET.txt',
             'HNI CorporationJuly 25, 2017 11_00 AM ET.txt',
             'HNI CorporationOctober 20, 2016, 11_00 AM ET.txt',
             'HNI CorporationOctober 24, 2017 11_00 am ET.txt',
             'IPG Photonics CorporationOctober 31, 2017 10_00 am ET.txt',
             'Netflix, Inc.December 4, 2017 12_00 PM ET.txt',
             'Time Inc.November 27, 2017 08_00 AM ET.txt',
             'Whirlpool CorporationApril 25, 2017 8_00 am ET.txt',
             'Whirlpool CorporationJanuary 26, 2017 10_00 ET.txt',
             'Whirlpool CorporationJuly 27, 2017 8_00 am ET.txt',
             'Whirlpool CorporationOctober 24, 2017 08_00 AM ET.txt',
             'Whirlpool CorporationOctober 25, 2016 10_00 am ET.txt']

text2 = open('Alcoa Q3 2017.txt','r').read()

score = []
magnitude = []

part_counter = 0

for x in range(0, len(text2), 2000):
    if len(text2)-x > 2000:
        parts = text2[x:x+2000]
    else:
        parts = text2[x:]        
    
    document = types.Document(
    content=parts,
    type="PLAIN_TEXT")
    
    sentiment = client.analyze_sentiment(document=document)
    
    score.append(sentiment.document_sentiment.score)
    magnitude.append(sentiment.document_sentiment.magnitude)

final_score = float(sum(score)) / len(score)
final_score

final_magnitude = float(sum(magnitude)) / len(magnitude)
final_magnitude









import random
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))
print(random.uniform(2.8, 6.8))





