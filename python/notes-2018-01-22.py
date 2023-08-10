for number in [9,0,2,1,0]:
    print(number)

for count in range(1,11):
    print(count, "time!")
print("Ahahaha!!!")

# Write a for loop to create a list of squares
squares = [] # Create an empty list
N = 6
for number in range(1,N+1):
    squares.append(number**2)
print(squares)

N = 6
squares = [n**2 for n in range(1,N+1)]
print(squares)

count = 10
while count > 0:
    print(count)
    count = count - 1
print("Blast off!")

def fibonacci(N):
    "Compute first N Fibonacci numbers."
    if N == 1:
        return [1]
    fib_list = [1,1]
    for n in range(2,N):
        next_fib = fib_list[n-1] + fib_list[n-2]
        fib_list.append(next_fib)
    return fib_list

fibonacci(10)

fib_1000 = fibonacci(1000)
fib_1000[-1]/fib_1000[-2]

(1+5**0.5)/2

def is_prime(N):
    "Determine if N is a prime number."
    if N < 2:
        return False
    for d in range(2,N):
        if N % d == 0:
            return False
    return True

for n in range(1,100):
    if is_prime(n):
        print(n,"is prime!")

