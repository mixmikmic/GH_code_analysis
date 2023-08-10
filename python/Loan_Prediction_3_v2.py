import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

dtrain = pd.read_csv('data/cleaned_train.csv').iloc[:, 2:]
dtest = pd.read_csv('data/cleaned_test.csv').iloc[:, 1:]

labels = dtrain.Loan_Status
dtrain.drop(['Loan_Status'], axis=1, inplace=True)

dtrain['total_income_to_loan_amt'] = (dtrain.ApplicantIncome + 
                                      dtrain.CoapplicantIncome) / dtrain.LoanAmount
dtrain['income_to_loan_amt'] = dtrain.ApplicantIncome / dtrain.LoanAmount

# at interest rate of 8.70%
interest_rate = 8.70 / (12 * 100)
dtrain['EMI'] = (dtrain.LoanAmount * interest_rate * (1 + interest_rate)**dtrain.Loan_Amount_Term) / (1 + interest_rate)**(dtrain.Loan_Amount_Term - 1)

dtest['total_income_to_loan_amt'] = (dtest.ApplicantIncome + 
                                      dtest.CoapplicantIncome) / dtest.LoanAmount
dtest['income_to_loan_amt'] = dtest.ApplicantIncome / dtest.LoanAmount

# at interest rate of 8.70%
interest_rate = 8.70 / (12 * 100)
dtest['EMI'] = (dtest.LoanAmount * interest_rate * (1 + interest_rate)**dtest.Loan_Amount_Term) / (1 + interest_rate)**(dtest.Loan_Amount_Term - 1)

dtrain.drop(['Loan_Amount_Term', 'ApplicantIncome',
            'CoapplicantIncome', 'LoanAmount'], axis=1, inplace=True)
dtest.drop(['Loan_Amount_Term', 'ApplicantIncome',
            'CoapplicantIncome', 'LoanAmount'], axis=1, inplace=True)

dtrain['Gender'] = np.log10(dtrain.Gender)
dtest['Gender'] = np.log10(dtest.Gender)

scalar = MinMaxScaler()
dtrain[['total_income_to_loan_amt', 'income_to_loan_amt']] = scalar.fit_transform(dtrain[
    ['total_income_to_loan_amt', 'income_to_loan_amt']])
dtest[['total_income_to_loan_amt', 'income_to_loan_amt']] = scalar.fit_transform(dtest[
    ['total_income_to_loan_amt', 'income_to_loan_amt']])

dtrain['Loan_Status'] = labels

dtrain.shape

dtest.shape

dtrain.to_csv('data/cleaned_train_v2.csv')
dtest.to_csv('data/cleaned_test_v2.csv')

get_ipython().system('ls data')



dte

dtest.shape



