def collatz(n):
    count = 0
    while n > 1:
        if n % 2 == 0:
            n /= 2
        else:
            n = 3*n + 1
        count += 1
    return count+1

longest = 0
for i in range(1_000_000):
    length = collatz(i)
    if length > longest:
        n = i
        longest = length
print(n, longest)

