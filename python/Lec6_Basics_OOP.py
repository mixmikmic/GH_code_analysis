x = complex(1,-1)
print(type(x))

x.real

x.imag

x.conjugate()

y = {1:'a',2:'b'}
y.__doc__

y.keys()

dir(complex)

dir(dict)

import inspect
inspect.getmembers(x)

class Consumer:
    
    def __init__(self, w):
        "Initialize consumer with w dollars of wealth"
        self.wealth = w
        
    def earn(self, y):
        "The consumer earns y dollars" 
        self.wealth += y
        
    def spend(self, x):
        "The consumer spends x dollars if feasible"
        new_wealth = self.wealth - x
        if new_wealth < 0:
            print("Insufficent funds")
        else:
            self.wealth = new_wealth

C1 = Consumer(10)  # this calls __init__ with w = 10
C2 = Consumer(100) # this calls __init__ with w = 100

# Inspect the data attributes of our objects
print("C1.wealth = ",C1.wealth)
print("C2.wealth = ",C2.wealth)

C1.spend(5)
C2.spend(5)
print("C1.wealth = ",C1.wealth)
print("C2.wealth = ",C2.wealth)

C1.earn(7)
C2.earn(15)
print("C1.wealth = ",C1.wealth)
print("C2.wealth = ",C2.wealth)

TransactionValue = 9
C1.earn(TransactionValue)
C2.spend(TransactionValue)
print("C1.wealth = ",C1.wealth)
print("C2.wealth = ",C2.wealth)

Cons = [Consumer(w0) for w0 in range(11,21)]
for i in range(len(Cons)):
    print("The wealth of consumer %d is %f units."%(i,Cons[i].wealth))

GovTransfer = 5
for cons in Cons:
    cons.earn(GovTransfer)

for i in range(len(Cons)):
    print("The wealth of consumer %d is %f units."%(i,Cons[i].wealth))

