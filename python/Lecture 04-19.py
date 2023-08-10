map, filter

def square(n):
    return n * n

a_list_of_numbers = list(range(2,12,2))
a_list_of_numbers

mapped = map(square, a_list_of_numbers)
mapped

mapped_list = list(mapped)
mapped_list

another_mapped_list = [square(n) for n in a_list_of_numbers]
another_mapped_list

list(map(square, a_list_of_numbers)), [square(n) for n in a_list_of_numbers] #eager/strict

#lazy, can iterate over once!
map(square, a_list_of_numbers), (square(n) for n in a_list_of_numbers) 

square2 = lambda n: n * n
list(map(square2, a_list_of_numbers)), list(map(lambda n: n * n, a_list_of_numbers))

list(map(lambda n: n * n, a_list_of_numbers)), [n * n for n in a_list_of_numbers]

list(filter(lambda n: not n % 2, range(1,11)))

[n for n in range(1, 11) if not n % 2]

# OOP
from functools import total_ordering
@total_ordering # get __le__, __gt__, __ge__ (<=, >, >=) for free with __lt__ (<)!
class Student:
    def _is_valid_operand(self, other):
        return (hasattr(other, "lastname") and
                hasattr(other, "firstname"))
    def __eq__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return ((self.lastname.lower(), self.firstname.lower()) ==
                (other.lastname.lower(), other.firstname.lower()))
    def __lt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return ((self.lastname.lower(), self.firstname.lower()) <
                (other.lastname.lower(), other.firstname.lower()))

# What is the @total_ordering?
class Student: # this class definition is equivalent to the above
    def _is_valid_operand(self, other):
        return (hasattr(other, "lastname") and
 hasattr(other, "firstname"))
    def __eq__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return ((self.lastname.lower(), self.firstname.lower()) ==
 (other.lastname.lower(), other.firstname.lower()))
    def __lt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return ((self.lastname.lower(), self.firstname.lower()) <
 (other.lastname.lower(), other.firstname.lower()))
    
Student = total_ordering(Student) # don't forget the class decorator

# Basic class definition:

class ActualStudent(Student): # we inherit from Student (above) - so totally ordered
    def __init__(self, firstname, lastname):
        """called when we instantiate a Student object"""
        self.firstname = firstname
        self.lastname = lastname

ActualStudent('Aaron', 'Hall')

ActualStudent.mro() #method resolution order

ActualStudent.__lt__

ActualStudent.__gt__

ActualStudent.__repr__

object.__ge__

ActualStudent # view __repr__ for class



