get_ipython().magic('matplotlib inline')

import datetime
import pandas as pd
import numpy as np

class loan:
    def __init__(self, name, princ, i_rate, i_start_date, today=datetime.date.today()):
        self.name = name
        self.princ = princ
        self.i_rate = i_rate
        self.i_start_date = i_start_date
        self.today = today
        
        days = (today-i_start_date).days
        if days <= 0:
            days = 0
            
        self.i_accrued = self.calc_interest(i_start_date, today)
        self.total = self.princ + self.i_accrued
        
    def calc_interest(self, start, end):
        if start < self.i_start_date:
            return 0
        days = (end - start).days
        if days <= 0:
            days = 0
        return days * self.i_rate * self.princ / 365        
    
    def pass_month(self):
        days_per_month = 365.25/12
        next_time = self.today + datetime.timedelta(days = days_per_month)
        
        self.i_accrued += self.calc_interest(self.today, next_time)
        
        self.today = next_time
        self.total = self.princ + self.i_accrued        
        
    
    def make_payment(self, amt):
        
        if amt > self.i_accrued:
            amt -= self.i_accrued
            self.i_accrued = 0
            if amt > self.princ:
                amt -= self.princ
                self.princ = 0
            else:
                self.princ -= amt
                amt = 0
        else:
            self.i_accrued -= amt
            amt = 0
        
        self.total = self.princ + self.i_accrued
        
            
        return amt
            
        
    def __str__(self):
        q = 'Name: {0}\nPrincipal: {1}\nInterest Rate: {2:0.2%}\nTotal: {3:0.2f}'
        return q.format(self.name, self.princ, self.i_rate, self.total)
    
    
class portfolio:
    def __init__(self):
        self.loans = []
        self.total = 0
        self.payments_made = 0
    
    def add_loan(self, one_loan):
        self.loans.append(one_loan)
        self.total += one_loan.total
        
    def __str__(self):
        for l in self.loans:
            print(l, "\n")
            
        total = sum([x.total for x in self.loans])
        return "Total Portfolio Size: {0:0.2f}".format(total)
    
    def pass_month(self):
        for x in self.loans:
            x.pass_month()
        self.total = sum([x.total for x in self.loans])
        
        return self
    
    def make_payment(self, amt):
        
        df = pd.DataFrame({
            'interest': [x.i_rate for x in p.loans],
            'principal': [x.princ for x in p.loans]
        })
        
        priorities = df.sort(['interest', 'principal'], ascending=False).index
        
        self.total = sum([x.total for x in self.loans])
        
        for l in priorities:
            amt = p.loans[l].make_payment(amt)
            self.total = sum([x.total for x in self.loans])
            if amt == 0 or self.total <= 0:
                break
                
        self.payments_made += 1

        return self
    
    def pay_loans(self, monthly):
        
        while self.total > 0:
            prev_tot = self.total
            self.make_payment(monthly).pass_month()

        print("Paid off in {0} months at ${1} per month".format(self.payments_made, monthly))
        print("Total paid ${0}".format(self.payments_made * monthly -(monthly - prev_tot)))

su = loan('institution', 3000, 0.050, datetime.date(2017, 5, 10))
s1 = loan('s1', 3500, 0.034, datetime.date(2017, 5, 10))
s3 = loan('s3', 4500, 0.034, datetime.date(2017, 5, 10))
s5 = loan('s5', 4500, .0386, datetime.date(2017, 5, 10))
s7 = loan('s7', 4500, .0466, datetime.date(2017, 5, 10))

u2 = loan('u2', 2000, 0.068, datetime.date(2011, 8, 14))
u4 = loan('u4', 2000, 0.068, datetime.date(2012, 8, 19))
u6 = loan('u6', 2000, .0386, datetime.date(2013, 8, 18))
u8 = loan('u8', 2000, .0466, datetime.date(2014, 8, 17))
u9 = loan('u9', 2000, .0584, datetime.date(2015, 8, 16))

p = portfolio()
p.add_loan(su); p.add_loan(s1);
p.add_loan(u2); p.add_loan(s3);
p.add_loan(u4); p.add_loan(s5);
p.add_loan(u6); p.add_loan(s7);
p.add_loan(u8); p.add_loan(u9);
print(p.total)

p.pay_loans(1200)

print(p)



