# First code
# This code works, but the flag variable is terribly ugly.
# Also, the end='' argument in the print functions is ugly.
# Also, there are many print functions. Fewer would be prettier.

def fizzbuzz(n):
    for i in range(1, n + 1):
        is_a_multiple = False
        if i % 3 == 0:
            print('Fizz', end='')
            is_a_multiple = True
        if i % 5 == 0:
            print('Buzz', end='')
            is_a_multiple = True
        if is_a_multiple:
            print()
        else:
            print(i)
            
# The range is limited for ease of development.
# Needed to include 3 * 5 in that range.

n = 20
fizzbuzz(n)

# Simplified much.
# The ugly flag variable is gone,
# replaced by a list, which accretes 'Fizz', 'Buzz', and str(i) as needed.
# There is only one print function and it is at the end.

def fizzbuzz(n):
    for i in range(1, n + 1):
        s = []
        if i % 3 == 0:
            s.append('Fizz')
        if i % 5 == 0:
            s.append('Buzz')
        if not s:
            s.append(str(i))
        print(''.join(s))
        
n = 20
fizzbuzz(n)

# This plays with directly printing the number,
# instead of appending str(i) to s.
# I prefer the single print of the earlier cell.

def fizzbuzz(n):
    for i in range(1, n + 1):
        s = []
        if i % 3 == 0:
            s.append('Fizz')
        if i % 5 == 0:
            s.append('Buzz')
        if not s:
            print(i)
        else:
            print(''.join(s))
            
n = 20
fizzbuzz(n)

# Make the function, a pure function.
# That is:
#     The function has no I/O.
#     The function's return value depends only in the input value.
# A drawback is that it requires much RAM for large n.

def fizzbuzz(n):
    t = []
    for i in range(1, n + 1):
        s = []
        if i % 3 == 0:
            s.append('Fizz')
        if i % 5 == 0:
            s.append('Buzz')
        if not s:
            s.append(str(i))
        t.append(''.join(s))
    return '\n'.join(t)

n = 20
print(fizzbuzz(n))

# Give s and t meaningful names.

def fizzbuzz(n):
    lines = []
    for i in range(1, n + 1):
        terms = []
        if i % 3 == 0:
            terms.append('Fizz')
        if i % 5 == 0:
            terms.append('Buzz')
        if not terms:
            terms.append(str(i))
        lines.append(''.join(terms))
    return '\n'.join(lines)

n = 20
print(fizzbuzz(n))

# Should lines have a new line for each line?
# Let's try it.
# It works, but the code got a little uglier, so I am against this code.

def fizzbuzz(n):
    lines = []
    for i in range(1, n + 1):
        terms = []
        if i % 3 == 0:
            terms.append('Fizz')
        if i % 5 == 0:
            terms.append('Buzz')
        if not terms:
            terms.append(str(i))
        terms.append('\n')
        lines.append(''.join(terms))
    return ''.join(lines)

n = 20
print(fizzbuzz(n), end='')

# Let's convert an earlier version to a generator.

def fizzbuzz(n):
    for i in range(1, n + 1):
        terms = []
        if i % 3 == 0:
            terms.append('Fizz')
        if i % 5 == 0:
            terms.append('Buzz')
        if not terms:
            terms.append(str(i))
        yield ''.join(terms)

n = 20
for line in fizzbuzz(n):
    print(line)

# I can not think of more improvements,
# so run it for n specified by Jeff Atwood.

def fizzbuzz(n):
    lines = []
    for i in range(1, n + 1):
        terms = []
        if i % 3 == 0:
            terms.append('Fizz')
        if i % 5 == 0:
            terms.append('Buzz')
        if not terms:
            terms.append(str(i))
        lines.append(''.join(terms))
    return '\n'.join(lines)

n = 100
print(fizzbuzz(n))

