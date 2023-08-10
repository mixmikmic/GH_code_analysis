range('a', 'z')

ord('A')

ord('a')

characters = []
for ord_value in range(65, 123):
    if 90 < ord_value < 97:
        continue
    characters.append(chr(ord_value))
characters

characters = ''.join(characters)
characters

# simple way to make list of numbers as list of strings
numbers = []
for num in range(0, 10):
    numbers.append(str(num))
numbers = ''.join(numbers)
numbers

special_chars = '!@#$%^&*_?{}[]()'

import random

help(random.randint)

random.randint(0, 99)

password_seed = characters + numbers + special_chars
password_seed

len(password_seed)

password = []
while len(password) < 6:
    index = random.randint(0, len(password_seed))
    password.append(password_seed[index])
password

password = ''.join(password)
password

has_num = False
for num in numbers:
    if num in password:
        has_num = True
        break
print(has_num)





def password_generator():
    password = []
    while len(password) < 6:
        index = random.randint(0, len(password_seed))
        password.append(password_seed[index])
    return ''.join(password)

password_generator()

password_generator()

password_generator(6)

def password_generator(length):
    password = []
    while len(password) < length:
        index = random.randint(0, len(password_seed))
        password.append(password_seed[index])
    return ''.join(password)

password_generator(6)

password_generator(10)

password_generator()

def password_generator(length=6):
    password = []
    while len(password) < length:
        index = random.randint(0, len(password_seed))
        password.append(password_seed[index])
    return ''.join(password)

password_generator()

password_generator(12)

password_generator(length=10)



def verify_password(password, verify_special=False, verify_case=False):
    internal_variable = 555
    global numbers
    has_num = False
    for num in numbers:
        if num in password:
            has_num = True
            break
            
    has_special = True
    if verify_special:
        has_special = False
        pass
        # TODO
    
    has_valid_case = True
    if verify_case:
        has_valid_case = False
        pass
        # TODO
        
    return has_num and has_special and has_valid_case

internal_variable

verify_password(password_generator(6))

pass3 = password_generator(10)
pass3

verify_password(pass3)

verify_password(pass3, True, True)

verify_password(pass3, verify_case=True, verify_special=True)





