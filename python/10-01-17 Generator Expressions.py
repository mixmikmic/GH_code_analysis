# Creat a list of students
students = ['Abe', 'Bob', 'Christina', 'Derek', 'Eleanor']

# Create a generator expressions that yields lower-case version's of the students names
lowercase_names = (student.lower() for student in students)

# View the generator object
lowercase_names

# Get the next name lower-cased
next(lowercase_names)

# Get the next name lower-cased
next(lowercase_names)

# Get the next name lower-cased
next(lowercase_names)

# Get the remaining names lower-cased
list(lowercase_names)

