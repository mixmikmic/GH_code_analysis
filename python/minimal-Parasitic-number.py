def new_num(n):
    n_str = str(n)
    res = n_str[-1:] + n_str[:-1]
    return int(res)

def is_num(n, k):
    new_n = new_num(n)
    if new_n == k * n:
        return True
    else:
        return False

for i in range(1000000):
    if is_num(i,4):
        print(i)

