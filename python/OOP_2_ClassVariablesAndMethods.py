class Employee:

    def __init__(self, first, last, pay):
        self.first = first #these are instance variables
        self.last = last
        self.pay = pay
        self.email = self.first + "." + self.last + "@deerwalk.com"
    
    def fullname(self):
        return "{} {}".format(self.first, self.last)

emp1 = Employee("Sagar", "Giri", 50000)
print(emp1.pay)
print(emp1.fullname())

# But, first lets set a hardcoded raise percentage
class Employee:

    def __init__(self, first, last, pay):
        self.first = first #these are instance variables
        self.last = last
        self.pay = pay
        self.email = self.first + "." + self.last + "@deerwalk.com"
    
    def fullname(self):
        return "{} {}".format(self.first, self.last)
    
    def raise_salary(self):
        self.pay = int(self.pay * 1.04)

emp1 = Employee("Sagar", "Giri", 50000)
print(emp1.pay)
emp1.raise_salary()
print(emp1.pay)

class Employee:
    rasie_amount = 1.04
    
    def __init__(self, first, last, pay):
        self.first = first #these are instance variables
        self.last = last
        self.pay = pay
        self.email = self.first + "." + self.last + "@deerwalk.com"
    
    def fullname(self):
        return "{} {}".format(self.first, self.last)
    
    def raise_salary(self):
        self.pay = int(self.pay * Employee.rasie_amount)
        #self.pay = int(self.pay * self.rasie_amount)

emp1 = Employee("Sagar", "Giri", 50000)
print(emp1.pay)
emp1.raise_salary()
print(emp1.pay)

emp1 = Employee("Sagar", "Giri", 50000)
emp2 = Employee("Hari", "Bahadur", 40000)
print(Employee.rasie_amount)
print(emp1.rasie_amount)
print(emp2.rasie_amount)

print(Employee.__dict__)

class Employee(object):
    num_of_employee = 0
    
    def __init__(self):
        Employee.num_of_employee += 1


class Account:
    interest = 0.15
    def __init__(self, name, accNo, balance=0.0):
        self.name = name
        self.accNo = accNo
        self.balance = balance

    def withdraw(self, amount):
        if self.balance > amount:
            self.balance -= amount
            return self.balance
        else:
            return False

    def deposit(self, amount):
        self.balance += amount
        return self.balance

    def getInterestRate(self):
        return self.interest

    @classmethod
    def initializeFromString(cls, inp): #This is class method
        name, accNo = inp.split("-")
        return cls(name, accNo)

a1 = Account("Sagar", 525252)
a2 = Account.initializeFromString("Sagar-1234")
a2.interest = 0.20
print(a1.getInterestRate())
print(a2.getInterestRate())

