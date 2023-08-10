import numpy as np

yrs_work = 35              # how many more years will you work
yrs_retr = 35              # how many years of retirement do you anticipate
int_rate = 0.05            # what interest rate do you expect to get
dep_401k = 8000            # how much do you plan to invest in 401k (including employer match)
dep_ira = 5500             # how much do you plan to invest in an IRA
yrly_ret_spnd = 80000      # how much do you expect to spend in retirement
ira_first = True           # do you plan to spend your IRA first?
res_401k = 0               # your current 401k balance
res_ira = 0                # your current IRA balance
tax_rate  = 0.28

for i in range(yrs_work + yrs_retr):
    res_401k *= (1 + int_rate)
    res_ira *= (1 + int_rate)
    
    if i < yrs_work:
        res_401k += dep_401k
        res_ira += dep_ira
    else:
        
        if ira_first:
            
            if res_ira > 0:
                res_ira -= yrly_ret_spnd
            else:
                res_401k -= yrly_ret_spnd /(1-tax_rate)
        else:
            
            if res_401k > 0:
                res_401k -= yrly_ret_spnd /(1-tax_rate)
            else:
                res_ira -= yrly_ret_spnd


    if (res_ira + res_401k) < 0:
        print("Oh no! You ran out of money {0} years into retirement (out of your desired {1})".format(i-yrs_work, yrs_retr))
        break
    elif i == (yrs_work + yrs_retr - 1):
        print("Success! You'll die with ${0:,.2f} unspent".format(res_401k + res_ira))

yrs_work = 35              # how many more years will you work
yrs_retr = 35              # how many years of retirement do you anticipate
int_rate = 0.055            # what interest rate do you expect to get

pct_to_401k = 0.05         # your contribution
employer_match = 0.05      # percent of your salary your employer contributes
pct_to_ira = 0.05          # pre-tax percentage (actual amount you put in)

'''
For example, if your salary is $100,000 before taxes and you contribute 5% to your 401k with a 5% match 
from your employer, the actual amount going into your 401k is $10,000 per year. If you also want to contribute
5% of your salary to a Roth IRA, you must do so after paying taxes on that money. So you will still be expected
to contribute $5,000 to your account but it will be after-tax money.
'''

start_sal = 80000
end_sal = 150000

yrly_ret_spnd = 80000      # how much do you expect to spend in retirement
ira_first = True          # do you plan to spend your IRA first?
res_401k = 0               # your current 401k balance
res_ira = 0                # your current IRA balance
tax_rate  = 0.28


pct_increase = np.power(end_sal/start_sal, 1/(yrs_work-1))
sals = start_sal*pct_increase**range(yrs_work)

for i in range(yrs_work + yrs_retr):
    res_401k *= (1 + int_rate)
    res_ira *= (1 + int_rate)
    
    if i < yrs_work:
        res_401k += sals[i] * (pct_to_401k + employer_match)
        res_ira += sals[i] * pct_to_ira
    else:
        
        if ira_first:
            
            if res_ira > 0:
                res_ira -= yrly_ret_spnd
            else:
                res_401k -= yrly_ret_spnd /(1-tax_rate)
        else:
            
            if res_401k > 0:
                res_401k -= yrly_ret_spnd /(1-tax_rate)
            else:
                res_ira -= yrly_ret_spnd


    if (res_ira + res_401k) < 0:
        print("Oh no! You ran out of money {0} years into retirement (out of your desired {1})".format(i-yrs_work, yrs_retr))
        break
    elif i == (yrs_work + yrs_retr - 1):
        print("Success! You'll die with ${0:,.2f} unspent".format(res_401k + res_ira))



