# Read input from console, input will read numeric values and throw error if a string value entered
num = input("Enter a number: ")
print type(num)

# Read input from console, raw_input will read input as string
num = raw_input("Enter a number: ")
print type(num)

# Set working directory
import os

# Set working directory
os.chdir('C:\\Users\\Manoh\\Documents')

# Below code will create a file named vechicles and add the items. \n is a newline character
vechicles = ['scooter\n', 'bike\n', 'car\n']
f = open('vechicles.txt', 'w')
f.writelines(vechicles)
f.close()

f = open('vechicles.txt')
print f.readlines()
f.close()

import sys

try:
    a = 1
    b = 1
    print "Result of a/b: ", a / b
except (ZeroDivisionError):
    print("Can't divide by zero")
except (TypeError):
    print("Wrong data type, division is allowed on numeric data type only")
except:
    print "Unexpected error occurred", '\n', "Error Type: ", sys.exc_info()[0], '\n', "Error Msg: ", sys.exc_info()[1]

try:
    a = 1
    b = 0
    print(a / b)
except (ZeroDivisionError):
    print("Can't divide by zero")
except (TypeError):
    print("Wrong data type, division is allowed on numeric data type only")
except:
    print "Unexpected error occurred", '\n', "Error Type: ", sys.exc_info()[0], '\n', "Error Msg: ", sys.exc_info()[1]

try:
    a = 1
    b = 0
    print(A / b)
except (ZeroDivisionError):
    print("Can't divide by zero")
except (TypeError):
    print("Wrong data type, division is allowed on numeric data type only")
except:
    print "Unexpected error occurred", '\n', "Error Type: ", sys.exc_info()[0], '\n', "Error Msg: ", sys.exc_info()[1]

try:
    f = open('C:\\Users\Manoh\\Documents\\vechicles.txt')
    print f.readline()
    i = int(s.strip())
except IOError as e:
    print "I/O error({0}): {1}".format(e.errno, e.strerror)
except ValueError:
    print "Could not convert data to an integer."
except:
    print "Unexpected error occurred", '\n', "Error Type: ", sys.exc_info()[0], '\n', "Error Msg: ", sys.exc_info()[1]
finally:
    f.close()
    print "file has been closed"

