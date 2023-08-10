# Printing something using print function

print ("hello world")

# Python script for describing data structure that has employee

SALARY_RAISE_FACTOR = .05
STATE_CODE_MAP = {'WA': 'Washington', 'TX': 'Texas'}

def update_employee_record(rec):
    old_sal = rec['salary']
    new_sal = old_sal * (1+SALARY_RAISE_FACTOR)
    rec['salary'] = new_sal
    state_code = rec['state_code']
    rec['state_name'] = STATE_CODE_MAP[state_code]
    
input_data = [ {'employee_name': 'Susan', 'salary': 100000.0, 'state_code': 'WA'},
                {'employee_name': 'Ellen', 'salary': 75000.0, 'state_code': 'TX'},]

for rec in input_data:
    update_employee_record(rec)
    name = rec['employee_name']
    salary = rec['salary']
    state = rec['state_name']
    print (name + ' now lives in ' + state)
    print ('    and make $' + str(salary))
    

# Atomic datatype

my_integer = 2
print (my_integer)
my_other_inter = 2+3
print (my_other_inter)
my_flot = 2.0
print (my_flot)
my_true_bool = True
print (my_true_bool)
my_false_bool = False
print (my_false_bool)
this_is_true = (0<100)
print (this_is_true)
this_is_false = (0>100)
print (this_is_false)

# String

a_string = "hello"
print (a_string)
same_as_previous = 'hello'
print (same_as_previous)
an_empty_string = ""
print (an_empty_string)
w_a_single_quote = "hello's"
print (w_a_single_quote)

# Multiline string

multi_line_string = """line 1
line 2"""

print (multi_line_string)

# Substring of a string

print ("ABCD"[0])
print ("ABCD"[0:2])
print ("ABCD"[1:3])

# Comments and Doc Strings

# this whole line is a comment
a = 4 # and last part of this line is too
# here again
print (a)

# Doc strings

def sqr(x):
    "This function just squares its intput "
    return x*x

# Complex datatype

my_list = ["a ", "b ", "c "]
print (my_list)
my_set = set(my_list)
print (my_set)
my_tuple = tuple(my_list)
print (my_tuple)


# List

my_list = ["a ", "b ", "c "]
print (my_list[0]) # print a

my_list[0] = "A " # changes that element of the list
print (my_list)
my_list.append("d ") # adds new element to the end of the list
print (my_list)

# list elements can be anything
mixed_list = ["A ", 5.7, "B ", [1,2,3]]
print (mixed_list)

# Special operation on list

original_list = [1,2,3,4,5,6,7,8,]
squares = [x*x for x in original_list]
squares_of_evens = [x*x for x in original_list if x%2 == 0]

print ("original - ", original_list)
print ("square - ", squares)
print ("square of the evens - ", squares_of_evens)


my_list = ["a", "b", "c"]
subset_of_list = my_list[1:2]
first_two_elements = my_list[:2]
last_two_elements = my_list[1:]
all_but_last_element = my_list[-1:]
print ("subset of my_list - ", my_list, "is - ", subset_of_list)
print ("first two elements of my_list - ", my_list, "is - ", first_two_elements)
print ("last two elements of my_list - ", my_list, "is - ", last_two_elements)
print ("all but last elements of my_list - ", my_list, "is - ", all_but_last_element)

## Substring manipulation

print ("ABC DEF".split())
print ("ABC \tDEF".split())
print ("ABC \tDEF".split(' '))
print ("ABCABD".split("AB"))
print (",".join(["A", "B", "C"]))

start, end, count_by = 1, 7, 2
print ("Slice example - ", "ABCDEFG"[start: end: count_by])

## Tuples example (this can not be modified like list)

my_tuple = (1, 2, "hello world")
print ("tuple first value - ", my_tuple[0])
print ("######Trying to modified first value of tuple - Expected error ####### ")
my_tuple[1] = 4

## Dictionaries

print ("Dictionary example:")
my_dict = {"January": 1, "February": 2}
print ("January: ", my_dict["January"])
my_dict["March"] = 3
print ("March: ", my_dict["March"])
print ("overriding the initial value of January")
my_dict["January"] = "Start of the year"
print ("January: ", my_dict["January"])

print ()
print ("Dictory using the list")
pairs = [("one", 1), ("two", 2)]
print ("list is: ", pairs)
as_dict = dict(pairs)
print ("dict is: ", as_dict)
same_as_pairs = as_dict.items()
print ("Back to list from the dict: ", same_as_pairs)

## set

s = set()
print ("An empty set s: ", s)
print ("check if has 5 in it: ", 5 in s)
s.add(5)
print ("after adding 5 in s: ", s)
print ("check if has 5 in it: ", 5 in s)
print ("adding another 5 - (it does nothing):", s.add(5))

## Functions

def my_function(x):
    y = x+1
    x_sqrd = x*x
    return x_sqrd

print ("Calling my_function to find the squr of 5 ")
print (my_function(5))
print ()

# function with optional arguments

def my_raise(x, n=2):
    return pow(x,n)

print ("Calling function raise which finds the power of given number to the given power. if the second arg not passed it will use 2 for power")
two_sqrd = my_raise(2)
print ("1. Without passing second argument - power of 2: ", two_sqrd)
two_cube = my_raise(2, n=3)
print ("2. 2 power 3: ", two_cube)
print ()


# example of lambda function
print ("Function defination using lambda (function containing one line function)")
squr = lambda x : x*x
five_sqrd = squr(5)
print ("Square of 5 using lambda function: ", five_sqrd)
print ()

#example of annonymus function
def apply_to_evens(a_list, a_func):
    return [a_func(x) for x in a_list if x%2==0]
print ("1. Defining the 'apply_to_evens' function which takes function as an second argument and passes the even number to that function")
my_list = [1,2,3,4,5]
print ("2. Now calling the 'apply_to_evens' function and passing an lambda function to sencond argument which calculates the square of given number")
sqrs_of_evens = apply_to_evens(my_list, lambda x:x*x)
print ("3. squars of the even number calculated suing 'apply_to_evens' function is: ", sqrs_of_evens)

## For loops and Control Structures

my_list = [1,2,3]
print ("For loop on my_list: ", my_list)
for x in my_list:
    print ("the number is ", x)
print ()

print ("For loop over my_dict.items: ")
for key, value in my_dict.items():
    print ("the value for ", key, " is ", value)
print ()

print ("If examples: ")
i = 4
if i<3:
    print ("i is less than three")
elif i<5: print ("i is between 3 and 5")
else: print ("i is greater than 5")

## Exception handling

print ("Exception handling ")
input_text = """first line
             second line"""
print ("input_text: ", input_text)
print ("Now trying to access 4th line: ")
try:
    lines = input_text.split("\n")
    print ("tenth line was: ", lines[4])
except:
    print ("########################################")
    print ("########################################")
    print ("Error while accessing line in input_text")
    print ("There were < 10 lines")

## Libraries import

print ("Importing sys and displaying the system path")
import sys
print (sys.path)
print ()
print ("Other import statement using from")

## classes and objects

class Dog:
    def __init__(self, name):
        self.name = name
    def respond_to_command(self, command):
        if command == self.name: self.speak()
    def speak(self):
        print ("bark bark!!")

fido = Dog("fido")
print ("Dog object fido: ", fido)
print ("Calling function of fido:")
fido.respond_to_command("spot") # does nothing
fido.respond_to_command("fido")

## Hashable and UnHashable types

a = 5 # a is a hashable int
b = a # b points to a copy of a
print ("Example of hashable type")
print ("Vaule of hashable int (a): ", a)
print ("Vaule of b (b=a): ", b)
a = a+1
print ("Vaule of b after modifying a to 6: ", b)
print ()

print ("Example of unhashable type")
A = [] # A is an unhashable list
print ("Vaule of unhashable list (A): ", A)
B = A  # B points to the same list as A
print ("Vaule of B (B=A): ", B)
A.append(5)
print ("Vaule of B after modifying A to have 5: ", B)
print ()

print ("Example of unhashable type made hashable ")
# And true copy can be maid like this for unhashable list
C = [] # C is an unhashable list
print ("Vaule of unhashable list (C): ", C)
D = [x for x in C] # D has copy of list C
print ("Vaule of D (D= copy of C): ", D)
C.append(5)
print ("Vaule of D after modifying C to have 5: ", D)
print ()

# Example where elements of unhashable list is unhashable 
print ("Example where elements of unhashable list is unhashable")
A = [{}, {}] # list of dicts
print ("A is: ", A)
B = [x for x in A]
print ("B is: ", B)
A[0]["name"] = "bob"
print ("A after modifying: ", A)
B[0]["name"]
print ("B after modifying A: ", B)

## Dataframes of panda

import pandas as pd
print ("Making data frame from a dictionary that maps column names to their values")
df = pd.DataFrame({
    "name": ["Bob", "Alex", "Janice"],
    "age": [60, 25, 33]
})
print ("Data frame df: \n", df)
print ()

print ("Reading a DataFrame from a file ")
other_df = pd.read_csv("data_files/myfile.csv")
print ("Loaded data frame other_df: \n", other_df)
print ()

print ("Making new columns from old ones is really easy")
df["age_plus_one"] = df["age"] + 1
df["age_times_two"] = 2 * df["age"]
df["age_squared"] = df["age"] * df["age"]
df["over_30"] = (df["age"] > 30) # this col is boolean
print ("Modified data frame df with new columns: \n", df)
print ()

print ("The col has various built in functions ")
total_age = df["age"].sum()
print ("Total age ", total_age)
median_age = df["age"].quantile(0.5)
print ("Median age: ", median_age)
print ()

print ("Select several rows of the DataFrame and make a new DataFrame out of them")
df_below50 = df[df["age"] < 50]
print ("Rows with age less than fifty")
print (df)
print ()

print ("Apply a custom function to a column")
df["age_squared"] = df["age"].apply(lambda x:x*x)
print ("Modified df with squared age: ")
print (df)
print ()

print ("---------------------------")
print ("Data frame indexing")
df = pd.DataFrame({
    "name": ["Bob", "Alex", "Janice"],
    "age": [60, 25, 33]
})
print (df.index)
print ()

print ("Creaet a DataFrame containing the same data but where name is the index")
df_w_name_as_ind = df.set_index("name")
print (df_w_name_as_ind.index) # print their names name
print ()

print ("Get the row for Bob")
bobs_row = df_w_name_as_ind.ix["Bob"]
print (bobs_row)
print ("Bobs age ", bobs_row["age"])

## Series of panda

print ("Example of Series structure of panda")
import pandas as pd

print ("Make series from the given list")
s = pd.Series([1,2,3])
print (s)

print()

print ("Add number to each element")
print (s+2)

print ()

print ("Access the index directly")
print (s.index)


print ()

print ("Adding two series - adds corresponding element of series")
s2 = s + pd.Series([4,5,5])
print (s2)

print ()

print ("Checking the type of Dataframe (it should be of the same series type)")
bobs_row = df_w_name_as_ind.ix["Bob"]
print (type(bobs_row))

## joining and grouping

print ("Create a new data frame ")
df_w_age = pd.DataFrame({
    "name": ["Tom", "Tyrell", "Claire"],
    "age": [60, 25, 33]
})
print("data frame df_w_age")
print (df_w_age)
df_w_height = pd.DataFrame({
    "name": ["Tom", "Tyrell", "Claire"],
    "height": [6.2, 4.0, 5.5]
})
print ("data frame df_w_height")
print (df_w_height)
print ()

joined = df_w_age.set_index("name").join(df_w_height.set_index("name"))
print ("Joined dataframe by name")
print (joined)
print ()

print ("reset index")
print (joined.reset_index())
print ()
print ("---------------------------------------------")
print ("Group the rows and agg each row of dataframe")
df_w_age = pd.DataFrame({
    "name": ["Tom", "Tyrell", "Claire"],
    "age": [60, 25, 33],
    "height": [6.2, 4.0, 5.5],
    "gender": ["M", "M", "F"]
})
print ("Dataframe")
print (df_w_age)
print ()

print ("Using built in aggregates")
print (df_w_age.groupby("gender").mean())
print ()
print ("Calculating median ")
print (df_w_age.groupby("gender").quantile(0.5))
print ()
print ("Using custome aggregate function")
def myaggs(ddf):
    return pd.Series({
        "name": max(ddf["name"]),
        "oldest": max(ddf["age"]),
        "mean_height": ddf["height"].mean()
    })
print (df_w_age.groupby("gender").apply(myaggs))

