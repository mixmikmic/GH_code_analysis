import time
def hello_recursion():
    time.sleep(0.5) # Pauses instruction for 0.5 second
    print("hello recursion!")
    return hello_recursion()

def hello_recursion(n):
    if n <= 0:
        print("Hello Base Case! : %d" % (n))
        return
    
    print("Hello Recursion! : %d" % (n))
    return hello_recursion(n-1)

hello_recursion(10)

def print_box(string, arrow=True):
    print('+' + len(string)*'-' + '+')
    print('|' + '%s' % (string) + '|')
    print('+' + len(string)*'-' + '+')

    if arrow:
        print('     |')
        print('     |')
        print('     \/')

def visualize_hello_recursion(n):
    if n <= 0:
        print_box("Hello Base Case! : %d" % (n), False)
        return

    print_box("Hello Recursion! : %d" % (n))
    return visualize_hello_recursion(n-1)

visualize_hello_recursion(10)

def fib(n):
    if n < 0:
        return 0

    if n == 1:
        return 1

    return fib(n-1) + fib(n-2)

def sum(i):
    return list[i] + sum(i-1)

def sum(i, list):
    if i == 0:
        return 0

    return list[i] + sum(i-1, list)

