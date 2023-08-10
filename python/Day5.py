arr_len = 3
arr1 = [100, 15, 1945]
arr2 = [8, 245, 54]

result = []
for idx in range(0, arr_len):
    _sum = arr1[idx] + arr2[idx]
    print(idx, _sum)

arr1[2] + arr2[2]

idx= 0
while idx < len(arr1):
    _sum = arr1[idx] + arr2[idx]
    print(idx, _sum)
    idx += 1

chr(92)

ord('a')

ord('z')

ord('A')

ord('Z')

ord('क')

chr(2327)

characters = ''
for ch in range(97, 123):
    characters += chr(ch)
characters

for ch in range(65, 91):
    characters += chr(ch)
characters

for ch in range(0, 10):
    characters += str(ch)
characters

'a' + 1

'a' + str(1)

str(1)

characters += '`~!@#$%^&*(_-)[]{}:;.,/?|'
characters

length = 6

import random

random.randint(0, 100)

len(characters)

characters[random.randint(0, len(characters) - 1)]

characters[-1]

password = ''
while length > 0:
    random_num = random.randint(0, len(characters) - 1)
    password += characters[random_num]
    length -= 1
password

def password_generator(password_length):
    password = ''
    while password_length > 0:
        random_num = random.randint(0, len(characters) - 1)
        password += characters[random_num]
        password_length -= 1
    return password

password_generator(8)

password_generator(12)



characters[0:62]

captcha_seed = characters[0:62] 

characters

captcha_seed

captcha_seed[7:9]

from random import randint as ri

ri(0, 7)

captcha_seed[8:12]

captcha_seed[32:21]

captcha_seed[-32:-21]



city = 'kathmandu'

city[2:5]

city[-2:-5]

city[-5:-2]

city[5:2]

city[5:2:1]

city[5:2:-1]

city[::-1]

city[2:7:3]

city



def captcha_generator():
    chr_length = ri(3, 5)
    start = ri(0, len(captcha_seed) - chr_length)
    end = start + chr_length
    return captcha_seed[start:end]

captcha_generator()

def pair_captcha(pairs=2):
    captchas = []
    for i in range(0, pairs):
        captchas.append(captcha_generator())
    print(captchas)
    return ' '.join(captchas)

pair_captcha()

pair_captcha(3)

