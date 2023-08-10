stuff = {
    'apple': 1.97,
    'banana': 2.99,
    'cherry': 3.99,
}

# Common pattern of .format use: use numerical indexes

for name, price in stuff.items():
    print('The price of {0} is {1}.'.format(name, price))

# Common pattern of .format use: use parameter names

for name, price in stuff.items():
    print(
        'The price of {name} is {price}.'.
        format(name=name, price=price))

for name, price in stuff.items():
    print(f'The price of {name} is {price}.')

tax_rate = 0.50

for name, price in stuff.items():
    print(f'The total price of {name} is {round(price * (1+tax_rate), 2)}.')

tax_rate = 0.50

for name, price in stuff.items():
    total_price = round(price * (1+tax_rate), 2)
    print(f'The total price of {name} is {total_price}.')

