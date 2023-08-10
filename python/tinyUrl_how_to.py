_alphabet = '23456789bcdfghjkmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ-_'
_base = len(_alphabet)

def encode(number):
    string = ''
    while(number > 0):
        string = _alphabet[number % _base] + string
        number //= _base
    return string

def decode(string):
    number = 0
    for c in string:
        number = number * _base + _alphabet.index(c)
    return number

n = 456788
encode(n)

decode('ckHb')



