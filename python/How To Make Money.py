get_ipython().magic('matplotlib notebook')

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.utils.extmath import cartesian

interest_rate = 0.1
reserve_rate = 0.2

class Bank(object):
    def __init__(self, interest_rate, reserve_rate):
        self.interest_rate = interest_rate
        self.reserve_rate = reserve_rate
        self.loans = []
        self.unrecovered = 0.0
        self.holding = 0.0
    def can_lend(self, amount):
        # check if there is enough money in holdings
        if self.unrecovered+amount > 1/self.reserve_rate*self.holding:
            return False # cannot loan any more money
        return True
    def lend(self, loan):
        self.unrecovered += loan.amount
    def repay(self, loan):
        self.unrecovered -= loan.repay_amount()


class Loan(object):
    def __init__(self, lender, borrower, amount, repay_at, interest_rate):
        self.lender = lender
        self.borrower = borrower
        self.amount = amount
        self.repay_at = repay_at
        self.interest_rate = interest_rate
    def repay_amount(self):
        return self.amount * (1 + self.interest_rate)
    def __repr__(self):
        return "{} | {}".format(self.repay_amount(), self.repay_at)




central_bank = Bank(interest_rate, reserve_rate)

class Investment(object):
    def __init__(self, principle, return_rate, return_at):
        self.principle = principle
        self.return_rate = return_rate
        self.return_at = return_at
        self.p = abs(random.gauss(0, 0.01)) # half normal distribution
    def __repr__(self):
        return "{} | {}".format(self.principle, self.return_at)

DEFAULT_THRESHOLD = 4
class Person(object):
    def __init__(self, bank, endowment, productivities, preferences, investment_pref, savings_pref):
        '''
        :param bank: which Bank does this Person banks with?
        :param endowment: how much savings does a person initially start with? rich families = high endowment
        :param productivities: a list of numbers the size of `basket`. 
                               This represents how many goods Person can produce per period.
        :param preferences: a list of floats the size of `basket`
        :param investment_pref: a float between [0,1). 0 = person doesn't care about his/her future earnings
        '''
        self.bank = bank
        self.savings = endowment
        self.bank.holding += self.savings
        self.productivities = productivities
        self.preferences = preferences
        self.investment_pref = investment_pref
        self.savings_pref = savings_pref
        self.loans = []
        self.investments = []
        self.last_income = endowment
        self.nonpayment = 0
        self.defaulted = False
        self.consumerism = np.random.beta(3,5)

    def utility(self, q, investment):
        '''
        utility represents a utility a Person has when presented with an allocation of goods and investment.
        
        :param q: a list of quantities
        :param investment: a number
        '''
        u = 1
        for g, p in zip(q, self.preferences):
            u *= math.pow(g, p)
        u *= math.pow(investment, self.investment_pref)
        return u

    def normalized_preferences(self):
        return [pref/(sum(self.preferences)+self.investment_pref+self.savings_pref) for pref in self.preferences]
    
    def normalized_productivities(self):
        return [prod/sum(self.productivities) for prod in self.productivities]
    
    def produce(self, priceMatrix, current):
        '''
        produce simulates the production capactity of a Person. 
        '''
        normal_prod = np.array(self.normalized_productivities())
        budget = self.budget(current)
        quantities = normal_prod * priceMatrix
        return quantities

    def consume(self, priceMatrix, current):
        '''
        Returns the demand given the prices. Demand is inelastick, because these Persons are turdballs.
        '''
        normal_pref = np.array(self.normalized_preferences())
        budget = self.budget(current)
        quantities = normal_pref * budget / priceMatrix
        return quantities
    
    def budget(self, current):
        '''
        The budget forms the constraint for a Person in his/her consumption
        '''
        repayments = 0.0
        for loan in self.loans:
            if current == loan.repay_at:
                repayments += loan.repay_amount()
        return self.creditworthiness()*self.consumerism + self.savings - (1-self.consumerism) * repayments

    def creditworthiness(self):
        '''
        creditworthiness determines how much a Person can borrow from the bank
        '''
        if self.savings > 0:
            return self.last_income + (self.last_income * interest_rate) + self.savings
        elif self.last_income > 0:
            return self.last_income + (self.last_income * interest_rate)
        return 0 # NO CREDIT FOR YOU!
    
    def debt(self):
        '''
        How much debt does a Person have?
        '''
        amount = 0.0
        for loan in self.loans:
            amount += loan.repay_amount()
        return amount
    
    def spend(self, amount, current):
        if amount < self.savings:
            self.savings -= amount
            return
        elif amount > self.savings:
            if self.bank.can_lend(amount):
                loan = Loan(self.bank, self, self.creditworthiness(), current+5, interest_rate)
                self.loans.append(loan)
                self.bank.lend(loan)
        else:
            self.savings = 0
            
    def earn(self, income):
        self.savings += income
        self.last_income = income

    def invest(self, amount, current):
        investment = Investment(amount, interest_rate, current+7)
        self.investments.append(investment)

    def step(self, prices, current):
        production = self.produce(prices, current)
        income = sum([p*q for p, q in zip(prices, production)])
        consumption = self.consume(prices, current)
        
        inv_pref = self.investment_pref/(sum(self.preferences)+self.investment_pref)
        inv_amt = inv_pref * self.budget(current)
        
        norm_prod = self.normalized_productivities()
        prod_cost = [p*q for p, q in zip(prices, production)]
        prod_cost = sum(n*c for n, c in zip(norm_prod, prod_cost))
        
        cost = sum([p*q for p, q in zip(prices, consumption)]) + inv_amt + prod_cost
        self.earn(income)
        self.spend(cost, current)
        if self.savings > inv_amt:
            self.invest(inv_amt, current)
        
        # remove loans that are up
        remove = []
        cantpay = False
        for loan in self.loans:
            if loan.repay_at <= current:
                if self.savings >= loan.repay_amount():
                    self.savings -= loan.repay_amount()
                    self.bank.repay(loan)
                    remove.append(loan)
                else:
                    cantpay = True
        for loan in remove:
            self.loans.remove(loan)
            
        if cantpay:
            self.nonpayment += 1
        
        # realize investments
        remove = []
        for investment in self.investments:
            if investment.return_at == current:
                self.savings += investment.principle*(1+investment.return_rate)
                self.productivities = [prod *(1+investment.p) for prod in self.productivities]
                remove.append(investment)
        for investment in remove:
            self.investments.remove(investment)
            
        # check for defaults
        if self.nonpayment >= DEFAULT_THRESHOLD:
            self.defaulted = True
            default_amt = 0.0
            for loan in self.loans:
                default_amt += loan.amount
            self.bank.unrecovered -= default_amt
            

basket = 3
endowment = random.gauss(50,10)
preferences = np.random.randint(0,10, basket)
skills = np.random.randint(0,10, basket)
investment_pref = np.random.randint(0,10)
savings_pref = np.random.randint(0,10)
A = Person(central_bank,endowment, skills, preferences, investment_pref, savings_pref)

for i in range(2):
    prices = np.random.beta(3,2, (3,basket)) * 10
    production = A.produce(prices,i)
    income = sum([p*q for p, q in zip(prices, production)])
    consumption = A.consume(prices,i)
    consumes = sum([p*q for p,q in zip(prices, consumption)])
    A.step(prices[0], i)

    print("Step {}".format(i))
    print("Price: \n{}".format(prices))
    print("Budget {}".format(A.budget(i)))
    print("Productivity \n{}".format(production))
    print("Consumption \n{}".format(consumption))
    print("Income: {} {}".format(A.last_income, income))
    print("Savings: {}".format(A.savings))
    print("Cost: {}".format(consumes))
    print("Debt: {}".format(A.debt()))
    print("Creditworthiness {}".format(A.creditworthiness()))
    print("Loans: {}".format(A.loans))
    print("Investments: {}".format(A.investments))
    print("\n\n")

class Market(object):
    current = 0
    def __init__(self, thickness, basket, bank, max_price, steps):
        self.basket = basket
        self.participants = []
        self.bank = bank
        self.max_price = max_price # government intervention! This is the price ceiling that the government sets
        self.steps = steps         # how many steps to break down from 0 to max price
        for i in range(thickness):
            endowment = random.gauss(50,10)
            preferences = np.random.randint(1,10, self.basket)
            skills = np.random.randint(1,10, self.basket)
            investment_pref = np.random.randint(0, 10)
            savings_pref = np.random.randint(0,10)
            participant = Person(bank,endowment, skills, preferences, investment_pref, savings_pref)
            self.participants.append(participant)
    def demand(self, priceMatrix):
        quantities = np.zeros((priceMatrix.shape[0], priceMatrix.shape[1]))
        for participant in self.participants:
            quantities += participant.consume(priceMatrix, self.current)
        return quantities
    
    def supply(self, priceMatrix):
        quantities = np.zeros((priceMatrix.shape[0], priceMatrix.shape[1]))
        for participant in self.participants:
            q = participant.produce(priceMatrix, self.current)
            quantities += q
        return quantities 

    
    def market_price(self):
        prices = (np.arange(self.steps) + 1) * self.max_price / float(self.steps)
        priceMatrix = np.tile(prices.reshape(self.steps,1), (1, self.basket))
#         priceMatrix = cartesian([prices for i in range(m.basket)])
        
        supply = self.supply(priceMatrix)
        demand = self.demand(priceMatrix)
        surplus = supply - demand
        min_surplus = np.argmin(np.abs(surplus),0)
        eq_prices = []
        for i in range(self.basket):
            eq_prices.append(priceMatrix[min_surplus[i]][i])
        return np.array(eq_prices)

    def step(self):
        eq_prices = self.market_price()
        for participant in self.participants:
            if not participant.defaulted:
                central_bank.holding -= participant.savings
                participant.step(eq_prices, m.current)
                central_bank.holding += participant.savings
                
        m.current += 1
    
    def m1(self):
        money = 0.0
        for participant in self.participants:
            if not participant.defaulted and participant.savings > 0:
                money += participant.savings
        return money
    def m2(self):
        money = 0.0
        for participant in self.participants:
            if not participant.defaulted:
                for loan in participant.loans:
                    money += loan.amount * (1+loan.interest_rate)
        return money

    def actives(self):
        return sum([1 for p in self.participants if not p.defaulted])

basket = 4
m = Market(100, basket, central_bank, 20, 20)
prices = (np.arange(m.steps) + 1) * m.max_price / float(m.steps)
# priceMatrix = cartesian([prices for i in range(m.basket)])
priceMatrix = np.tile(prices.reshape(m.steps,1), ( 1, basket))
supply = m.supply(priceMatrix)
demand = m.demand(priceMatrix)
market_price = m.market_price()
print("Market Price: {}".format(market_price))

fig, ax = plt.subplots(1,1)
ax.set_xlabel("Quantity")
ax.set_ylabel("Price")
ax.plot(demand.T[0],priceMatrix.T[0], c='r')
ax.plot(supply.T[0],priceMatrix.T[0], c='b')
fig.canvas.draw()

def plotmoney(ax, steps, m1s, m2s):
    ax.lines[0].set_xdata(steps)
    ax.lines[1].set_xdata(steps)
    ax.lines[0].set_ydata(m1s)
    ax.lines[1].set_ydata(m2s)
    
def plotpeople(ax, steps, people):
    ax.lines[0].set_xdata(steps)
    ax.lines[0].set_ydata(people)

def plotprices(ax, steps, goods):
    for i, g in enumerate(goods):
        ax.lines[i].set_xdata(steps)
        ax.lines[i].set_ydata(g)

#reset in case the below case "no loans allowed" is run previously
def budget(self, current):
    '''
    The budget forms the constraint for a Person in his/her consumption
    '''
    repayments = 0.0
    for loan in self.loans:
        if current == loan.repay_at:
            repayments += loan.repay_amount()
    return self.creditworthiness()*self.consumerism + self.savings  - (1-self.consumerism) * repayments
Person.budget = budget
central_bank = Bank(interest_rate, reserve_rate)

n_steps = 60
participant_count = 1000
max_price = 100
price_steps = 200
basket = 7
m = Market(participant_count, basket, central_bank, max_price, price_steps)
print(central_bank.holding)
print(len(m.participants))
mp = m.market_price()

fig,(ax1,ax2, ax3) = plt.subplots(3,1, sharex=True)
ax1.set_xlabel('Time')
ax1.set_ylabel('Money Supply')
ax1.set_xlim(0,n_steps)
ax1.set_ylim(0,700000)
ax2.set_xlabel('Time')
ax2.set_ylabel('Prices')
ax2.set_xlim(0,n_steps)
ax2.set_ylim(0,35)
ax3.set_ylabel("Participants")
ax3.set_ylim(0,participant_count*1.1)


steps =[0]
m1s = [m.m1()]
m2s = [m.m2()]
participants = [participant_count]
prices = [[p] for p in mp]
colors = ['r', 'b', 'g', 'c', 'm', 'y','k']

ax1.plot(steps, m1s, c='r')
ax1.plot(steps, m2s, c='b')
ax3.plot(steps, participants)
for i, p in enumerate(prices):
    ax2.plot(steps, p, c=colors[i])
fig.canvas.draw()

for i in range(n_steps):
    mp = m.market_price()
    m.step()

    steps.append(i)
    m1 = m.m1()
    m2 = m.m2()
    
    # plotting shit
    m1s.append(m1)
    m2s.append(m2)
    for i, p in enumerate(mp):
        prices[i].append(p)

    participants.append(m.actives())
    
    maxY1 = max(m1s+m2s)*1.1
    maxY2 = max([p for pl in prices for p in pl ])*1.1
    ax1.set_ylim(0, maxY1)
    ax2.set_ylim(0, maxY2)
    plotmoney(ax1, steps, m1s, m2s)
    plotprices(ax2, steps, prices)
    plotpeople(ax3, steps, participants)
    
    fig.canvas.draw()
#     print("Time {}. M1: {}, M2: {}, Total: {}".format(i, m1, m2, m1+m2))
    

import types
central_bank2 = Bank(interest_rate, reserve_rate)
central_bank2.can_lend = types.MethodType(lambda bank, amount: False, central_bank2) # never do this on prod servers!
Person.budget = lambda self, cuurent: self.savings

m = Market(participant_count, basket, central_bank2, max_price, price_steps)
print(central_bank2.holding)
print(len(m.participants))
mp = m.market_price()

fig,(ax1,ax2, ax3) = plt.subplots(3,1, sharex=True)
ax1.set_xlabel('Time')
ax1.set_ylabel('Money Supply')
ax1.set_xlim(0,n_steps)
ax1.set_ylim(0,700000)
ax2.set_xlabel('Time')
ax2.set_ylabel('Prices')
ax2.set_xlim(0,n_steps)
ax2.set_ylim(0,35)
ax3.set_ylabel("Participants")
ax3.set_ylim(0,participant_count*1.1)


steps =[0]
m1s = [m.m1()]
m2s = [m.m2()]
participants = [participant_count]
prices = [[p] for p in mp]
colors = ['r', 'b', 'g', 'c', 'm', 'y','k']

ax1.plot(steps, m1s, c='r')
ax1.plot(steps, m2s, c='b')
ax3.plot(steps, participants)
for i, p in enumerate(prices):
    ax2.plot(steps, p, c=colors[i])
fig.canvas.draw()

for i in range(n_steps):
    mp = m.market_price()
    m.step()

    steps.append(i)
    m1 = m.m1()
    m2 = m.m2()
    
    # plotting shit
    m1s.append(m1)
    m2s.append(m2)
    for i, p in enumerate(mp):
        prices[i].append(p)

    participants.append(m.actives())
    
    maxY1 = max(m1s+m2s)*1.1
    maxY2 = max([p for pl in prices for p in pl ])*1.1
    ax1.set_ylim(0, maxY1)
    ax2.set_ylim(0, maxY2)
    plotmoney(ax1, steps, m1s, m2s)
    plotprices(ax2, steps, prices)
    plotpeople(ax3, steps, participants)
    
    fig.canvas.draw()
#     print("Time {}. M1: {}, M2: {}, Total: {}".format(i, m1, m2, m1+m2))

