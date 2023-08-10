print(round(1.5))
print(round(2.1))

print(round(2.5))

print("0: " + str(bin(0)))
print("1: " + str(bin(1)))
print("2: " + str(bin(2)))
print("3: " + str(bin(3)))
print("4: " + str(bin(4)))
print("5: " + str(bin(5)))
print("6: " + str(bin(6)))

import decimal

number = decimal.Decimal(1.1)
print(number)

number = decimal.Decimal('1.1')
print(number)

print(number + 1)
print(number * 5)
print(round(number))

number = decimal.Decimal('2.5')
print(round(number))

print(decimal.getcontext())

decimal.getcontext().rounding = 'ROUND_HALF_UP'
print(decimal.getcontext())

number = decimal.Decimal('2.5')
print(round(number))

number = decimal.Decimal('1.1')
rounded = number.quantize(decimal.Decimal('0.01'))
print(rounded)

number = decimal.Decimal('1.1')
rounded = number.quantize(decimal.Decimal('1'), rounding='ROUND_HALF_UP')
print(rounded)

number = decimal.Decimal('2.5')
rounded = number.quantize(decimal.Decimal('1'), rounding='ROUND_HALF_UP')
print(rounded)

def round_decimal(x, digits = 0):
    #casting to string then converting to decimal
    x = decimal.Decimal(str(x))
    
    #rounding for integers
    if digits == 0:
        return int(x.quantize(decimal.Decimal("1"), rounding='ROUND_HALF_UP'))

    #string in scientific notation for significant digits: 1e^x 
    if digits > 1:
        string =  '1e' + str(-1*digits)
    else:
        string =  '1e' + str(-1*digits)

    #rounding for floating points
    return float(x.quantize(decimal.Decimal(string), rounding='ROUND_HALF_UP'))

print("Built-in Rounding:")
print(round(555.555))
print(round(555.555, 1))
print(round(555.555, -2))

print("Custom Rounding:")
print(round_decimal(555.555))
print(round_decimal(555.555, 1))
print(round_decimal(555.555, -2))

print("Built-in Rounding:")
print(round(2.5))

print("Custom Rounding:")
print(round_decimal(2.5))

