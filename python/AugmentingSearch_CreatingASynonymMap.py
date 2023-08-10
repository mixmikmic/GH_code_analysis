import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
from collections import Counter

def get_web_syns(word, pos=None, n = 5):
    if pos == None:
        req = requests.get('http://www.thesaurus.com/browse/%s' % word)
    else:
        req = requests.get('http://www.thesaurus.com/browse/%s/%s' % (word, pos))

    soup = BeautifulSoup(req.text, 'html.parser')
    
    all_syns = soup.find('div', {'class' : 'relevancy-list'})
    syns = []
    if all_syns == None:
        return syns
    for ul in all_syns.findAll('ul'):
        for li in ul.findAll('span', {'class':'text'}):
            syns.append(li.text.split(",")[0])
    return syns[:n]

# Example
get_web_syns('hello')

INPUT_FILE = "raw_text_enriched_with_keywords_sample.xlsx"
df = pd.read_excel(INPUT_FILE)
print(df[['ParaText','Keywords']])

MIN_KEYWORD_COUNT = 1
keywords_list = df["Keywords"].tolist()

flattened_keywords_list = []
for sublist in keywords_list:
    for val in sublist.split(","):
        flattened_keywords_list.append(val)
        
keywords_count = Counter(flattened_keywords_list)
keywords_filtered = Counter(el for el in keywords_count.elements() if keywords_count[el] >=MIN_KEYWORD_COUNT)

keyword_synonym = {keyword:get_web_syns(keyword) for keyword in keywords_filtered}
#print(keyword_synonym)
print("Number of keywords-synonym pairs before cleaning:",len(keyword_synonym))

# a helper function to identify and filter out keywords containing a digit - normally, you cannot find synonyms 
#for such words in thesaurus
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

keyword_synonym_clean = {}
for k,v in keyword_synonym.items():
    if v!=[] and not hasNumbers(k):
        keyword_synonym_clean[k]=v
        
print("Number of keywords-synonym pairs after cleaning:",len(keyword_synonym_clean))
# peek at a few keyword-synonyms pairs
print(dict(list(keyword_synonym_clean.items())[0:5]))

# domain specific acronyms in the taxcode world
acronym_dict = """AAA, Accumulated Adjustment Account
Acq., Acquiescence
ACRS, Accelerated Cost Recovery System
ADR, Asset Depreciation Range
ADLs, Activities of Daily Living
ADS, Alternative Depreciation System
AFR, Applicable Federal Rate
AGI, Adjusted Gross Income
AIME, Average Indexed Monthly Earnings (Social Security)
AMT, Alternative Minimum Tax
AOD, Action on Decision
ARM, Adjustable Rate Mortgage
ATG, Audit Techniques Guide
CB, Cumulative Bulletin
CCA, Chief Council Advice
CC-ITA, Chief Council - Income Tax and Accounting
CCC, Commodity Credit Corporation
CCP, Counter-Cyclical Program (government farm program)
CDHP, Consumer-Driven Health Plan
CFR, Code of Federal Regulations
CLT, Charitable Lead Trust
COBRA, Consolidated Omnibus Budget Reconciliations Act of 1985
COGS, Cost of Goods Sold
COLA, Cost of Living Adjustment
CONUS, Continental United States
CPI, Consurmer Price Index
CRT, Charitable Remainder Trust
CSRA, Community Spouse Resource Allowance
CSRS, Civil Service Retirement System
DOD, Date of Death
DOI, Discharge of Indebtedness
DP, Direct Payment (government farm program)
DPAD, Domestic Production Activities Deduction
DPAI, Domestic Production Activities Income
DPAR, Domestic Production Activities Receipts
DPGR, Domestic Production Gross Receipts
EFIN, Electronic Filing Identification Number
EFT, Electronic Funds Transfer
EFTPS, Electronic Federal Tax Payment System
EIC, Earned Income Credit
EIN, Employer Identification Number
f/b/o, For Benefit Of or For and On Behalf Of
FICA, Federal Insurance Contribution Act
FIFO, First In First Out
FLP, Family Limited Partnership
FMV, Fair Market Value
FR, Federal Register
FS, IRS Fact Sheets (example: FS-2005-10)
FSA, Flexible Spending Account or Farm Service Agency
FTD, Federal Tax Deposit
FUTA, Federal Unemployment Tax Act
GCM, General Counsel Memorandum
GDS, General Depreciation System
HDHP, High Deductible Health Plan
HOH, Head of Household
HRA, Health Reimbursement Account
HSA, Health Savings Account
IDC, Intangible Drilling Costs
ILIT, Irrevocable Life Insurance Trust
IR, IRS News Releases (example: IR-2005-2)
IRA, Individual Retirement Arrangement
IRB, Internal Revenue Bulletin
IRC, Internal Revenue Code
IRD, Income In Respect of Decedent
IRP, Information Reporting Program
ITA, Income Tax and Accounting
ITIN, Individual Taxpayer Identification Number
LDP, Loan Deficiency Payment
LIFO, Last In First Out
LLC, Limited Liability Company
LLLP, Limited Liability Limited Partnership
LP, Limited Partnership
MACRS, Modified Accelerated Cost Recovery System
MAGI, Modified Adjusted Gross Income
MFJ, Married Filing Jointly
MMMNA, Minimum Monthly Maintenance Needs Allowance
MRD, Minimum Required Distribution
MSA, Medical Savings Account (Archer MSA)
MSSP, Market Segment Specialization Program
NAICS, North American Industry Classification System
NOL, Net Operating Loss
OASDI, Old Age Survivor and Disability Insurance
OIC, Offer in Compromise
OID, Original Issue Discount
PATR, Patronage Dividend
PBA, Principal Business Activity
PCP, Posted County Price, also referred to as AWP - adjusted world price
PHC, Personal Holding Company
PIA, Primary Insurance Amount (Social Security)
PLR, Private Letter Ruling
POD, Payable on Death
PSC, Public Service Corporation
QTIP, Qualified Terminable Interest Property
RBD, Required Beginning Date
REIT, Real Estate Investment Trust
RMD, Required Minimum Distribution
SCA, Service Center Advice
SCIN, Self-Canceling Installment Note
SE, Self Employment
SEP, Simplified Employee Pension
SIC, Service Industry Code
SIMPLE, Savings Incentive Match Plan for Employees
SL, Straight-Line Depreciation
SMLLC, Single Member LLC
SSA, Social Security Administration
SSI, Supplemental Security Income
SSN, Social Security Number
SUTA, State Unemployment Tax Act
TC, Tax Court
TCMP, Taxpayer Compliance Measurement Program
TD, Treasury Decision
TIN, Taxpayer Identification Number
TIR, Technical Information Release
TOD, Transfer on Death
USC, United States Code
U/D/T, Under Declaration of Trust
UNICAP, Uniform Capitalization Rules
UTMA, Uniform Transfers to Minors Act
VITA, Volunteer Income Tax Assistance
GO Zone, Gulf Opportunity Zone
Ct. D., Court Decision
Ltr. Rul., Letter Rulings
Prop. Reg., Proposed Treasury Regulations
Pub. L., Public Law
Rev. Proc., Revenue Procedure
Rev. Rul., Revenue Ruling
"""

OUTPUT_FILE = "keywords_synonym.txt"

file = open(OUTPUT_FILE, 'w')
# 1. add the acronyms: comma separated to indicate both ways relationship, e.g. "<=>"
file.write(acronym_dict)
# 2. add the synonyms: "=>" separated to indicate a relationship from left to right only
for k,v in keyword_synonym_clean.items():
    line = k.strip() + "=>" + ','.join(v) + "\n"
    file.write(line)
    
file.close()

get_ipython().run_cell_magic('bash', '', 'cat keywords_synonym.txt | head -5 | less -S\ncat keywords_synonym.txt | tail -5 | less -S')



