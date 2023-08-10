import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

dataset= pd.read_csv('Market_Basket_Optimisation.csv', header=None)

### preprocess
# apriori expects the input as a list of list, which contains all the diff transactions. --> One big list that contains all the transactions that are in lists
# dataset is now just a dataframe, i.e. Neither each transaction nor whole transactions is structured as list
transactions = []
for i in range(len(dataset)):
    transactions.append([str(dataset.values[i, j]) for j in range(len(dataset.columns))])

# just to confirm the contents of transactions. `transactions` is a big list which contains 7.5k small lists of  20 names}
print(type(transactions))
print(type(transactions[1]))
print(type(transactions[1][1]))
print('\n')
print(len(transactions))
print(len(transactions[1]))
print(transactions[1][1])

### training apriori on the dataset
# import file instead of using library
from apyori import apriori # apriori is defined in apyori file that is in the same working directory
rules=apriori(transactions=transactions, 
              min_support=0.003,     # e.g. items bought at least three times a day 3*7/7500, 7500 is the nr of purchases a week in the market
              min_confidence=0.2,    # these arguments depend on the business prob we face
              min_lift=3,                      #  normally, we need to adjust the arguments several times to get a meaningful result
              min_length=2)
results= list(rules)  # the result is sorted according to its criterion (most relevant to least wrt sup, conf, and lift)

# Vidualise the result 
def inspect(results):
    rh          = [tuple(result[2][0][0]) for result in results]
    lh          = [tuple(result[2][0][1]) for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(rh, lh, supports, confidences, lifts))
# this command creates a data frame to view
resultDataFrame=pd.DataFrame(inspect(results),
                columns=['rhs','lhs','support','confidence','lift'])

resultDataFrame



