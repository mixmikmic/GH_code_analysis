from __future__ import print_function

filename = '20150410-dojo-prices.txt'

print(open(filename).read())

for line in open(filename):
    print(line.split())

max_price = 0.0
for line in open(filename):
    price = float(line.split()[1])
    max_price = max(max_price, price)
print(max_price)

max((float(line.split()[1]) for line in open(filename)))

max(map(lambda line: float(line.split()[1]), open(filename)))

max(map(float, map(lambda line: line.split()[1], open(filename))))

max(map(float, map(lambda x: x[1] ,map(lambda line: line.split(), open(filename)))))

