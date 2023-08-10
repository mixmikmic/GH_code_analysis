import math
def getFactors(n):
    out, curr = [], []
    if n <= 2:
        return out


    def dfs(start, n):
        for i in range(start, int(math.sqrt(n))+1):
            # if i is a factor
            if n%i == 0 and i <= n//i:
                # it means [i,n/i] is a combination
                curr.append(i)
                out.append(curr+[n//i])
                dfs(i, n//i) # solve all factors starting with i
                curr.remove(i)

    dfs(2, n)
    return out


print getFactors(2)
print getFactors(8)
print getFactors(14)
print getFactors(20)
print getFactors(21)
print getFactors(36)



