maxN = 987654321
maxN = 654321  # XXX Passer à la vraie valeur
l = len(str(maxN))
sum32 = 0
digits19 = set(range(1, l+1))
for multiplicand in range(1, 1+maxN):  # upto 987 654 321
    multiplier = 1
    product = multiplicand * multiplier
    while multiplier <= maxN and product <= maxN:  # Be smart here!
        digits = str(multiplicand)+str(multiplier)+str(product)
        if len(digits) == l and set(digits) == digits19:
            print("multiplicand = {}, multiplier = {}, product = {}".format(multiplicand, multiplier, product))
            print("digits =", digits)
            sum32 += product
        multiplier += 1
        product = multiplicand * multiplier

print("The sum of all products whose multiplicand/multiplier/product identity can be written as a 1 through", l, "pandigital is")
print(sum32)



