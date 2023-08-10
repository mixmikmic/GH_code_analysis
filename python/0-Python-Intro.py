import time

# The print function lets you print to the console. This is useful to learn about what your code is doing
print("Hello Python")
print(1 + 1)
print("Local time is " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

print("Howdy Python")
print('Cheers Python')

name = "Curtis Connors"
age = 32
number_of_books_owned = 42

print(name)
print(name, age, number_of_books_owned)

whole_number = 22
decimal_number = 22.22
negative_whole_number = -22

print(whole_number)
print(decimal_number)
print(negative_whole_number)

a, b = 2, 7  # Assigning variable can also be done like this. a = 2 and b = 7

print(a + b)  # Addition
print(a - b)  # Subtraction
print(a * b)  # Multiplication
print(a % b)  # Modulus - Remainder of a / b
print(a / b)  # Float Division
print(a // b)  # Integer Division
print(b ** 2)  # Square a number

# These are short-hand expressions for modifying the value of a variable
a += 1
b *= 2
print(a)
print(b)

a = 2
b = 7
c = 22

x = c / a * b
print(x)

y = c / (a + b)
print(y)

apple = 'Apple'
citrus = "Orange"

print(apple)
print(citrus)

full_name = "Curtis\nConnors"  # \n is a new line character

print(full_name)

some_tabular_header = "Name\tSurname\tAge"
some_tabular_data = "Curtis\tConnors\t32"

print(some_tabular_header)
print(some_tabular_data)

print("You're a wizzard")  # Notice the single quote in the string for you're

name = "Curtis"
surname = "Connors"

message = "Hello, my name is " + name + " " + surname
print(message)

name = "Curtis"
surname = "Connors"
age = 32

message = "Hello my name is {0} {1} and I am {2} years old".format(name, surname, age)
print(message)

age_as_string = "18"
age_as_integer = int(age_as_string)
print(age_as_integer)

# Now we can work with the value as a number
age_as_integer += 1
print(age_as_integer)

# We can also explictly define the type of a variable
typed_age:int = 18

person_name = "Curtis Connor"
person_age = 19
person_country = "ZA"

age_limit = 18

if person_age >= age_limit:
    message = person_name + " you're welcome"
    print(message)

if person_age >= 18 and person_country == "ZA":
    print("Welcome to responsibility")
    
if person_age <= 18 or True:
    print("We're forcing responsibility")
    

person_age = 17
if person_age >= age_limit:
    message = person_name + " you're welcome"
    print(message)
else:
    message = person_name + " please GTFO"
    print(message)

person_age = 9
if 0 <= person_age <= 18:
    print("Enjoy!")
elif 19 <= person_age <= 65:
    print("Welcome to responsibility")
else:
    print("Enjoy you reward!")
    

for i in range(0, 10):
    print(i)

# Guess a number between 0 and 10. 
correct_answer = 5
guess = 0
while guess != correct_answer:
    guess = 5#int(input("Guess a number until you get it right: "))  # This is how you can get input for the console
print("You guessed right!")

name = "Curtis Connors"
for c in name:
    print(c)

all_numbers_between_0_and_10 = range(0, 11)
for i in all_numbers_between_0_and_10:
    print(i)

twos = range(2, 100, 2)  # All the numbers that are multiples of 2 from 1 to 100
print(twos)  # The range
print(twos[1])  # The second multiple of 2 - remember we index from 0

fruits = ["apple", "orange", "pear", "banana", "melon"]
fruits.append("lime")

print("The second fruit is: {}".format(fruits[1]))

for fruit in fruits:
    print(fruit)
    
fruits.append("apple")  # Lists allow duplicates
print(fruits)

fruits.sort()
print(fruits)

even_numbers = list(range(0, 10, 2))
odd_numbers = list(range(1, 10, 2))
print(even_numbers)
print(odd_numbers)

all_numbers = even_numbers + odd_numbers  # Using the + operator creates a list that contains the contents of both
print(all_numbers)

all_numbers.sort()  # Order the list
print(all_numbers)

all_numbers.sort(reverse=True)  # Reverse order the list
print(all_numbers)

vegetables = {"potato", "corn", "cucumber", "olive"}
print(vegetables)

vegetables.add("potato")
print(vegetables)
vegetables.add("onion")  # Notice that order is not guarenteed
print(vegetables)

if "tomato" not in vegetables:
    print("Yep. It's technically a fruit")

even = set(range(0, 42, 2))
squares = {1, 4, 9, 16, 25, 36}

print(even)
print(squares)

# Union of two sets. I.e. Create one set with the values from both
print(even.union(squares))

# Intersection of two sets. I.e. Create one set that is the common values from both
print(even.intersection(squares))

# Difference of two sets. I.e. Create one set that is the uncommon values from both
print(even.difference(squares))

drinks = {"Coke": "Fizzy cola", 
          "Coffee": "Work juice",
          "Beer": "After work juice"}

print(drinks)

# Now any key can be referenced
print(drinks["Coke"])

# Here's a function definition that takes in two numbers and returns the result
def sum(a, b):
    return a + b

# Now we can call that function and reuse it with different parameters
print(sum(1, 1))
print(sum(2, 1))
print(sum(7, 5))

