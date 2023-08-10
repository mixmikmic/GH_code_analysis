import math

def cumulative(x, m, stdv):
    return 0.5 * (1 + math.erf((x-m)/ (stdv * math.sqrt(2)) ))

mean = 70
stdv = 10

# gt80 = cumulative(100, mean, stdv) - cumulative(80, mean, stdv)
# gte60 = cumulative(100, mean, stdv) - cumulative(60, mean, stdv)
# lt60 = cumulative(60, mean, stdv)
gt80 = 1 - cumulative(80, mean, stdv)
gte60 = 1 - cumulative(60, mean, stdv)
lt60 = cumulative(60, mean, stdv)

print(round(gt80, 4))
print(round(gte60, 4))
print(round(lt60, 4))



