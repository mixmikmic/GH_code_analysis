import math

def get_sum_factors(n):
    sum_factors = 0
    
    sqrt_n = math.sqrt(n)
    
    for i in range(1,math.ceil(sqrt_n)):
        if n % i == 0:
            sum_factors += (i+(n/i))
    
    #deals with numbers that are other numbers squared
    if n % sqrt_n == 0:
        sum_factors += sqrt_n

    return sum_factors-n

get_sum_factors(220)

get_sum_factors(284)

sum_facs = [get_sum_factors(x) for x in range(1,10_000)]

a_nums = []

for i in range(len(sum_facs)):
    j = i
    while j < len(sum_facs):
        
        a = i+1
        b = j+1
        
        d_a = sum_facs[i]
        d_b = sum_facs[j]
        
        if d_a == b and d_b == a and a != b:
            a_nums.append(a)
            a_nums.append(b)
        
        j += 1

print(a_nums)

sum(a_nums)

