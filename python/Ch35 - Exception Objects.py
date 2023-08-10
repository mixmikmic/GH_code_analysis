import sys

class General(Exception): pass
class Specific1(General): pass
class Specific2(General): pass

def raiseGernerl(): raise General()
def raiseSpecific1(): raise Specific1()
def raiseSpecific2(): raise Specific2()
    
for func in (raiseGernerl, raiseSpecific1, raiseSpecific2):
    try:
        func()
    except General as X:
        print('Method 1: ' ,sys.exc_info()[0])
        print('Method 2: ', X.__class__)
        print()

# Without Hierarchies
class Divzero(Exception): pass
class Oflow(Exception): pass

def func():
    raise Divzero
    raise Oflow

try:
    func()
except (Divzero, Oflow) as e:
    print(e.__class__)

# With Hierarchies
class NumErr(Exception): pass
class Divzero(NumErr): pass
class Oflow(NumErr): pass

def func():
    raise Divzero
    raise Oflow

try:
    func()
except NumErr as e:
    print(e.__class__)

# This should be sufficient for most cases

class MyExc(Exception): pass

try:
    raise MyExc('This is my Message')
except MyExc as e:
    print(e)

# For More Detail Customization

class MyExc(Exception):
    def __str__(self):
        return 'detail'
    
try:
    raise MyExc()
except MyExc as e:
    print(e)

class MyLogExc(Exception):
    file_name = 'log.txt'
    
    
    def logerror(self):
        log = open(self.file_name, 'a')
        print('Error:', self.args, file=log)
        
try:
    raise MyLogExc('This is logging error')
except MyLogExc as exc:
    exc.logerror()

with open('./log.txt', 'r') as input_file:
    print(input_file.read())

