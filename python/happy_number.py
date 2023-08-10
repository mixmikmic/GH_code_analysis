def get_sum(number):
    arr = [int(x) for x in str(number)]
    sum = 0
    for n in arr:
        sum += n*n
    return sum

def is_happy_number(number,iterations=1000):
    i = 1
    for i in range(1,iterations):
        s = get_sum(number)
        if (s == 1):
            return True
            break
        else:
            number = s
    return False
    print(i)

get_ipython().magic('time is_happy_number(12345)')



