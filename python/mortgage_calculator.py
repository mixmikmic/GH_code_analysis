get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import numpy as np

class own_home:
    def __init__(self, value, down, interest, term, tax_rate, 
                 maintenance_rate, insurance_rate, HOA_fees):
        
        if type(down) == float:
            down = down * value
        
        self.P = value - down
        
        self.value = value
        self.down = down
            
        self.N = term * 12
        self.i = interest/12
        
        self.mortgage_payment = self.calculate_mortgage_payment()
        
        self.monthly_tax = tax_rate * value / 12
        self.monthly_insurance = insurance_rate * value / 12
        self.monthly_upkeep = maintenance_rate * value / 12
        self.monthly_hoa = HOA_fees
        
        self.total_monthly = self.calculate_monthly_cost()
        
        self.total_cost = self.total_monthly * self.N + down
        
    def calculate_mortgage_payment(self):
        
        m_payment = self.i * (self.P*(1+self.i)**self.N)/( (1+self.i)**self.N - 1)
        
        return m_payment
    
    def calculate_monthly_cost(self):
        
        total = sum([self.mortgage_payment, self.monthly_tax, self.monthly_insurance,
                   self.monthly_upkeep, self.monthly_hoa])
        
        return total
        
    def __str__(self):
        
        fmt = '{0:30} $ {1:,.2f}'
        long_string = [
            'Monthly Breakdown',
            '-'*50,
            fmt.format('Mortgage Payment', o.mortgage_payment),
            fmt.format('Property Taxes', o.monthly_tax),
            fmt.format('Insurance', o.monthly_insurance),
            fmt.format('Maintenance Cost', o.monthly_upkeep),
            fmt.format('HOA Fees', o.monthly_hoa),
            "",
            fmt.format('Total Monthly Payment', self.total_monthly),
            "",
            "Total Costs",
            "-"*50,
            fmt.format('Total Cost of Home', self.total_cost),
            fmt.format('Total Cost minus Value', self.total_cost - self.value)
        ]

        return '\n'.join(long_string)
    
    def compare_with_rental(self, monthly_rent):
        
        print(self)
        
        rental_cost = monthly_rent * self.N
        print('\n')
        print('Comparison with Renting')
        print('-'*50)
        print('{0:30} $ {1:,.2f}'.format("Comparable Monthly Rent", monthly_rent))
        print('{0:30} $ {1:,.2f}'.format("Total Cost of Rental", rental_cost))
        # total savings
        print('{0:30} $ {1:,.2f}'.format("Savings over Renting", rental_cost - self.total_cost + self.value))
        
    def create_burndown(self):
        
        P = [self.P]
        I = [0]
        princ_contribution = [0]
        
        for i in range(self.N):
            interest = self.i * P[i]
            amt_to_principal = self.mortgage_payment - interest         
            P.append(P[i]-amt_to_principal)
            I.append(interest)
            princ_contribution.append(amt_to_principal)
            
        return princ_contribution, I
        
        
def inverse_payment_calculator(payment, 
                               down=.20, 
                               interest=0.04, 
                               term=30, 
                               tax_rate=0.02, 
                               maintenance_rate=0.01, 
                               insurance_rate=0.0035, 
                               HOA_fees=0):
    i = interest/12
    N = term * 12
    
    numerator = payment - HOA_fees
    denom = i*(1-down)*(1+i)**N/((1+i)**N - 1) + (tax_rate + insurance_rate + maintenance_rate)/12
    
    return numerator/denom
        

o = own_home(value=475000, 
             down=.20, 
             interest=0.035, 
             term=15, 
             tax_rate=0.02, 
             maintenance_rate=0.01, 
             insurance_rate=0.0035, 
             HOA_fees=0)

print(o)

o.compare_with_rental(2000)

fig, ax = plt.subplots(figsize=(15, 9))

ind = range(o.N + 1)[::6]
width=5

p1 = ax.bar(ind, o.create_burndown()[0][::6], width, color='r')
p2 = ax.bar(ind, o.create_burndown()[1][::6], width, color='y',
             bottom=o.create_burndown()[0][::6])

# plt.ylabel('Scores')
plt.title('Principal and Interest Contributions', size=22)
plt.xticks([x + width/2 for x in ind][::2], [int(x/12) for x in ind][::2])
plt.legend((p1[0], p2[0]), ('Principal', 'Interest'))

plt.show()

fig, ax = plt.subplots(figsize=(15, 9))

ind = range(o.N + 1)


p1 = ax.plot(ind, o.down + np.cumsum(o.create_burndown()[0]))
plt.xticks(ind[::12], [int(x/12) for x in ind][::12])
ax.grid(b=True, which='major', color='r', linestyle='--')
plt.title('Equity Built Over Time', size=22)


plt.show()

home_values = [1000 * x for x in range(100,500)]
payments = [own_home(value=x, 
             down=.20, 
             interest=0.04, 
             term=15, 
             tax_rate=0.02, 
             maintenance_rate=0.01, 
             insurance_rate=0.0035, 
             HOA_fees=0).total_monthly for x in home_values]

fig, ax = plt.subplots(figsize=(15, 9))

ax.plot(payments, home_values)

ax.grid(b=True, which='major', color='r', linestyle='--')

ax.set_ylabel('Home Price', fontsize=18)

ax.set_xlabel('Monthly Payment', fontsize=18)

plt.suptitle('Home Price vs. Monthly Payments', fontsize=22)
plt.title('Assumes 20% Down, 4% interest, 15 year term, 1% Maintenance, .35% Insurance', fontsize=14)

ax.tick_params(labelsize=14)

plt.show()

pct_to_housing = .3 # should be between 1/5 and 1/3, loosely


home_values = [1000 * x for x in range(100,500)]
salaries = [own_home(value=x, 
             down=.20, 
             interest=0.04, 
             term=15, 
             tax_rate=0.02, 
             maintenance_rate=0.01, 
             insurance_rate=0.0035, 
             HOA_fees=0).total_monthly / pct_to_housing * 12 for x in home_values]


fig, ax = plt.subplots(figsize=(15, 9))

ax.plot(salaries, home_values)

ax.grid(b=True, which='major', color='r', linestyle='--')

ax.set_ylabel('Home Price', fontsize=18)

ax.set_xlabel('Minimum Salary Required', fontsize=18)

plt.suptitle('Home Price vs. Salary', fontsize=22)
plt.title('Assumes 20% Down, 4% interest, 15 year term, 1% Maintenance, .35% Insurance, 2% Taxes', fontsize=14)

ax.tick_params(labelsize=14)

plt.show()

def norm(x, lower, upper):
    # normalize a number between .2 and .4
    return (x-lower)/(upper-lower) * (.4 - .2) + .2


salaries = [x*1000 for x in range(50, 150)]
payment_amt = [x * norm(x, 50000, 150000) / 12 for x in salaries]
home_values = [inverse_payment_calculator(payment=x, 
                                          down=.20, 
                                          interest=0.04,
                                          term=15,
                                          tax_rate=0.02,
                                          maintenance_rate=0.01,
                                          insurance_rate=0.0035,
                                          HOA_fees=0) for x in payment_amt]


fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

plt.suptitle('Home Price vs. Salary', fontsize=22, y=1.1)

ax1.plot(salaries, home_values)
ax1.grid(b=True, which='major', color='r', linestyle='--')
ax1.set_ylabel('Home Price', fontsize=18)
ax1.set_xlabel('Minimum Salary Required', fontsize=18)
ax1.set_title('15 year term', fontsize=14, y=1.12)
ax2 = ax1.twiny()
ax2.set_xticks([x/10 for x in range(11)])
ticks = [norm(x, 50000, 150000) for x in salaries]
ax2.set_xticklabels([round(x, 2) for x in ticks[::10] + [ticks[-1]]])
ax2.set_xlabel("Home Expenditure Rate")
ax1.tick_params(labelsize=14)

salaries = [x*1000 for x in range(50, 150)]
payment_amt = [x * norm(x, 50000, 150000) / 12 for x in salaries]
home_values = [inverse_payment_calculator(payment=x, 
                                          down=.20, 
                                          interest=0.04,
                                          term=30,
                                          tax_rate=0.02,
                                          maintenance_rate=0.01,
                                          insurance_rate=0.0035,
                                          HOA_fees=0) for x in payment_amt]

ax3.plot(salaries, home_values)
ax3.grid(b=True, which='major', color='r', linestyle='--')
# ax3.set_ylabel('Home Price', fontsize=18)
ax3.set_xlabel('Minimum Salary Required', fontsize=18)
ax3.set_title('30 year term', fontsize=14, y=1.12)
ax4 = ax3.twiny()
ax4.set_xticks([x/10 for x in range(11)])
ticks = [norm(x, 50000, 150000) for x in salaries]
ax4.set_xticklabels([round(x, 2) for x in ticks[::10] + [ticks[-1]]])
ax4.set_xlabel("Home Expenditure Rate")
ax3.tick_params(labelsize=14)

plt.show()

def norm(x, lower, upper):
    # normalize a number between .2 and .4
    return (x-lower)/(upper-lower) * (.4 - .2) + .2


salaries = [x*1000 for x in range(50, 150)]
payment_amt = [x * norm(x, 50000, 150000) / 12 for x in salaries]
home_values = [inverse_payment_calculator(payment=x, 
                                          down=.20, 
                                          interest=0.04,
                                          term=30,
                                          tax_rate=0.02,
                                          maintenance_rate=0.01,
                                          insurance_rate=0.0035,
                                          HOA_fees=0) for x in payment_amt]


fig, ax = plt.subplots(figsize=(15, 9))

ax.plot(salaries, home_values)

ax.grid(b=True, which='major', color='r', linestyle='--')

ax.set_ylabel('Home Price', fontsize=18)

ax.set_xlabel('Minimum Salary Required', fontsize=18)

plt.suptitle('Home Price vs. Salary', fontsize=22, y=1.04)
plt.title('Assumes 20% Down, 4% interest, 30 year term, 1% Maintenance, .35% Insurance, 2% Taxes', fontsize=14, y=1.08)

ax2 = ax.twiny()
ax2.set_xticks([x/10 for x in range(11)])
ticks = [norm(x, 50000, 150000) for x in salaries]
ax2.set_xticklabels([round(x, 2) for x in ticks[::10] + [ticks[-1]]])
ax2.set_xlabel("Home Expenditure Rate")

ax.tick_params(labelsize=14)

plt.show()

o = own_home(value=300000, 
             down=.20, 
             interest=0.03, 
             term=15, 
             tax_rate=0.02, 
             maintenance_rate=0.01, 
             insurance_rate=0.0035, 
             HOA_fees=0)
o.total_monthly
o.mortgage_payment

o = own_home(value=300000, 
             down=.20, 
             interest=0.04, 
             term=30, 
             tax_rate=0.02, 
             maintenance_rate=0.01, 
             insurance_rate=0.0035, 
             HOA_fees=0)
o.total_monthly

sum([(700*12) * (1.03)**x for x in range(1,31)]) # 30 yrs investment income

sum([(1700*12) * (1.03)**x for x in range(1,16)]) # 15 yrs investment income

o = own_home(value=300000, 
             down=.20, 
             interest=0.04, 
             term=30, 
             tax_rate=0.02, 
             maintenance_rate=0.01, 
             insurance_rate=0.0035, 
             HOA_fees=0)

print(o)



