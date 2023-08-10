import math

def isprime(n):
    #0 and 1 are not prime
    if n in [0,1]:
        return False
    #Check if number is even and greater than 2
    if (n>2) and (n%2==0):
        return False
    #Check if any odd number between 3 and the roof of the number is a factor
    for i in range(3, int(math.sqrt(n))+1, 2):
        if n%i==0:
            return False
    return True

#Check all numbers between 0 and 3000 for prime numbers
prime=[]
for i in range(3001):
    if isprime(i):
        prime.append(i)
print(prime)



