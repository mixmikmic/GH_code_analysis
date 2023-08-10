get_ipython().magic('load_ext watermark')
get_ipython().magic("watermark -a 'Sebatian Raschka' -u -d -v")

def naive_gcd(a, b):
    gcd = 0
    if a < b:
        n = a
    else:
        n = a
    for d in range(1, n + 1):
        if not a % d and not b % d:
            gcd = d
    return gcd

print('In: 1/1,', 'Out:', naive_gcd(1, 1))
print('In: 1/2,', 'Out:', naive_gcd(1, 2))
print('In: 3/9,', 'Out:', naive_gcd(3, 9))
print('In: 12/24,', 'Out:', naive_gcd(12, 24))
print('In: 12/26,', 'Out:', naive_gcd(12, 26))
print('In: 26/12,', 'Out:', naive_gcd(26, 12))
print('In: 13/17,', 'Out:', naive_gcd(13, 17))

def eucl_gcd_recurse(a, b):
    if not b:
        return a
    else:
        return eucl_gcd_recurse(b, a % b)
    
print('In: 1/1,', 'Out:', naive_gcd(1, 1))
print('In: 1/2,', 'Out:', naive_gcd(1, 2))
print('In: 3/9,', 'Out:', naive_gcd(3, 9))
print('In: 12/24,', 'Out:', naive_gcd(12, 24))
print('In: 12/26,', 'Out:', naive_gcd(12, 26))
print('In: 26/12,', 'Out:', naive_gcd(26, 12))
print('In: 13/17,', 'Out:', naive_gcd(13, 17))

def eucl_gcd_dynamic(a, b):
    while b:
       tmp = b 
       b = a % b 
       a = tmp 
    return a

print('In: 1/1,', 'Out:', naive_gcd(1, 1))
print('In: 1/2,', 'Out:', naive_gcd(1, 2))
print('In: 3/9,', 'Out:', naive_gcd(3, 9))
print('In: 12/24,', 'Out:', naive_gcd(12, 24))
print('In: 12/26,', 'Out:', naive_gcd(12, 26))
print('In: 26/12,', 'Out:', naive_gcd(26, 12))
print('In: 13/17,', 'Out:', naive_gcd(13, 17))

a = 12313432
b = 34234232342

get_ipython().magic('timeit -r 3 -n 5 naive_gcd(a, b)')

get_ipython().magic('timeit -r 3 -n 5 eucl_gcd_recurse(a, b)')

get_ipython().magic('timeit -r 3 -n 5 eucl_gcd_dynamic(a, b)')

