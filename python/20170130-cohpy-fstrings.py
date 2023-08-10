stuff = {
    'apple': 1.97,
    'banana': 2.99,
    'cherry': 3.99,
}

# Common pattern of .format use:

for name, price in stuff.items():
    print(
        'The price of {name} is {price}'.
        format(name=name, price=price))

for name, price in stuff.items():
    print(f'The price of {name} is {price}')

tax = 0.50

for name, price in stuff.items():
    print(f'The total price of {name} is {round(price * (1+tax), 2)}')

template = lambda: f'tes{k}'
template

k = 9
template()

k = 8
template()

def template():
    return f'tes{k}'

k = 9
template()

k = 8
template()

