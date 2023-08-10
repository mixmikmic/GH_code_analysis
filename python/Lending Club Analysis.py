import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
get_ipython().magic('matplotlib inline')
pd.set_option('display.max_rows', 200)

def read_loan_data():
    """Read in data from 21 files and concatentate into one dataframe"""
    dataframes = []
    for i in range(21):
        filename = './data/LoanStats_2007_to_2015_' + str(i) + '.csv'
        df = pd.read_csv(filename, low_memory=False)
        dataframes.append(df)
    loans = pd.concat(dataframes)
    return loans

loans = read_loan_data()

loans.shape

loans.head()

# Remove 9 loans with missing interest rates. These were never funded
loans = loans[loans.int_rate.notnull()]

# Strip out % and convert to float
loans.int_rate = loans.int_rate.apply(lambda x: float(x.rstrip('%')))

loans.int_rate.describe()

sns.distplot(loans.int_rate, axlabel='Interest Rate');

loans.term.value_counts(1)

loans.term.value_counts().plot(kind='pie', fontsize=16);

loans.loan_amnt.describe()

loans.loan_amnt.plot(kind='hist', bins=20, x='Loan Amount');

purposes = loans.purpose.value_counts()
purposes

purposes.plot.barh(figsize=(15, 5));

titles = loans.title.str.cat(sep=',')
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', max_font_size=40, relative_scaling=0.5)
wordcloud.generate(titles)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

loans.home_ownership.value_counts()

loans.loc[(loans.home_ownership == 'ANY') | (loans.home_ownership == 'NONE'), 'home_ownership'] = 'OTHER'

loans.home_ownership.value_counts().plot(kind='pie', fontsize=16);

loans.grade.value_counts().sort_index(ascending=False).plot(kind='barh', fontsize=16);

loans.sub_grade.value_counts().sort_index().plot(kind='bar', fontsize=16, figsize=(12, 4));

from datetime import datetime
loans['issue_date'] = loans.issue_d.apply(lambda x: datetime.strptime(x, '%b-%Y').date())

loans.issue_date.apply(lambda x: x.year).value_counts().sort_index().plot(kind='bar');

loans.addr_state.value_counts() #  51 states including DC as it's own

loans.loan_status.value_counts()

loans['defaulted'] = loans.loan_status.map({'Fully Paid': 0,
                                            'Current': 0,
                                            'Charged Off': 1,
                                            'Late (31-120 days)': 1,
                                            'In Grace Period': 0,
                                            'Late (16-30 days)': 0,
                                            'Does not meet the credit policy. Status:Fully Paid': 0,
                                            'Does not meet the credit policy. Status:Charged Off': 1,
                                            'Default': 1
                                           })

loans = loans.drop('loan_status', axis=1)

# Member id and loan URL can be removed
loans = loans.drop(['url', 'member_id'], axis=1)

loans.application_type.value_counts()

# Since there are only 511 Joint accounts, let's remove them to simplify our analysis
loans = loans[loans.application_type == 'INDIVIDUAL']
joint_app_columns = ['revol_bal_joint', 'annual_inc_joint', 'dti_joint', 'verification_status_joint',
                     'sec_app_num_rev_accts', 'sec_app_mths_since_last_major_derog', 'sec_app_fico_range_high',
                     'sec_app_earliest_cr_line', 'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc',
                     'sec_app_revol_util', 'sec_app_open_il_6m', 'sec_app_chargeoff_within_12_mths',
                     'sec_app_collections_12_mths_ex_med', 'sec_app_fico_range_low']
loans = loans.drop(joint_app_columns, axis=1)

# Remove policy code since it is 1 for all rows
loans = loans.drop('policy_code', axis=1)

# last_credit_pull_d => The most recent month LC pulled credit for this loan. Remove since it's irrelevant
loans = loans.drop('last_credit_pull_d', axis=1)

# Show that max range is small, so that it's safe to take mean
#loans.loc[:, ['fico_range_low', 'fico_range_high', 'fico']].head(20)

# Fico is pulled multiple times while loan is being invested. Take the mean of high and low
loans['fico'] = (loans.fico_range_low + loans.fico_range_high) / 2
loans = loans.drop(['fico_range_low', 'fico_range_high'], axis=1)

# Hardship program allows borrowers who had an unexpected life event make interest only payments. Since it was
# introduced recently in May, 2017, it only affects some current loans. So it's best to remove those columns.
hardship_columns = ['orig_projected_additional_accrued_interest', 'hardship_status', 'payment_plan_start_date',
                   'hardship_type', 'hardship_reason', 'deferral_term', 'hardship_amount', 'hardship_end_date',
                   'hardship_start_date', 'hardship_length', 'hardship_dpd', 'hardship_loan_status',
                   'hardship_payoff_balance_amount', 'hardship_last_payment_amount', 'hardship_flag']
loans = loans.drop(hardship_columns, axis=1)

# These fields were added on December 2015 and only exist for new loans
dec_2015_columns = ['il_util', 'mths_since_rcnt_il', 'open_acc_6m', 'inq_last_12m', 'open_il_6m', 'open_il_12m',
                   'open_il_24m', 'total_bal_il', 'open_rv_12m', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc',
                   'inq_fi', 'total_cu_tl', 'all_util']
loans = loans.drop(dec_2015_columns, axis=1)

# Description is missing for most loans
loans = loans.drop('desc', axis=1)

# We can drop last payment date and next payment date since they are irrelevant
loans = loans.drop(['next_pymnt_d', 'last_pymnt_d'], axis=1)

# We can drop title since we already have purpose as a categorical feature
loans = loans.drop('title', axis=1)

loans.shape

# Safe to delete rows where these columns are null
# annual_inc, 

loans[loans.annual_inc.isnull()].issue_date

loans[loans.acc_now_delinq.isnull() | loans.total_acc.isnull() | loans.pub_rec.isnull() | loans.open_acc.isnull()
      | loans.delinq_amnt.isnull() | loans.inq_last_6mths.isnull() | loans.earliest_cr_line.isnull() 
      | loans.delinq_2yrs.isnull()].shape[0]

# Remove these loans from the summer of 2007
missing_summer_2007 = ['delinq_2yrs', 'acc_now_delinq', 'earliest_cr_line', 'inq_last_6mths', 'open_acc', 'pub_rec',
                      'delinq_amnt', 'total_acc']
loans = loans.dropna(subset=missing_summer_2007)

loans[loans.tax_liens.isnull()].issue_d.value_counts()

# Remove loans where tax liens are not present since they were issued early on
loans = loans[loans.tax_liens.notnull()]
loans.shape

# chargeoff_within_12_mths and collections_12_mths_ex_med
loans[loans.chargeoff_within_12_mths.isnull() | loans.collections_12_mths_ex_med.isnull()].issue_d.value_counts()

# Drop these two since they are from earlier loans
loans = loans.dropna(subset=['chargeoff_within_12_mths', 'collections_12_mths_ex_med'])

loans.pub_rec_bankruptcies.value_counts()

# Public records are derogatory records such as bankruptcy, civil judgment, and tax liens
loans[loans.mths_since_last_record.isnull() & (loans.pub_rec > 0) & (loans.tax_liens > 0)
      & (loans.pub_rec_bankruptcies > 0)].shape

# Since the other three columns for public records report 0, the months since last record should be 0
loans.mths_since_last_record.fillna(value=0, inplace=True)



# Find all the columns with null values
loans.isnull().sum().sort_values(ascending=False)

#loans[loans.mths_since_last_record.isnull()][['mths_since_last_record','delinq_amnt','delinq_2yrs', 'issue_d', 'pub_rec']]
loans[loans.mths_since_last_record.notnull() & (loans.pub_rec == 0)][['mths_since_last_record', 'pub_rec']]



# revol_util => Revolving line utilization rate, or the amount of credit the borrower is using relative to all available
# revolving credit.
# It's missing for 502 of the rows and not at any particular point in time.
#loans[loans.revol_util.isnull()].issue_date.value_counts().sort_index()
# revol_bal => Total credit revolving balance.
# num_rev_accts => Number of revolving accounts. Includes closed accounts
# open_acc => Number of open credit lines in borrower's credit file. Does this included both revolving and fixed installment?
# num_actv_rev_tl => Number of currently active revolving accounts
# num_actv_bc_tl => Number of currently active bank card accounts
# avg_cur_balance => Avg current balance of all accounts


# def impute_revol_util(df):
    # Calculate 
    

loans[['revol_bal', 'num_rev_accts', 'open_acc', 'num_actv_rev_tl', 'avg_cur_bal', 'tot_cur_bal', 'dti', 'annual_inc']].head(20)

