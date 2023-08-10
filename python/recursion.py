def C(n):
    return 4 ** n

print(C(0))
print(C(1))
print(C(3))

def C(n):
    a = 1
    for i in range(n):
        a = a * 4
    return a
    
print(C(0))
print(C(1))
print(C(3))

def C(n):
    if n == 0:
        return 1
    return C(n - 1) * 4
    
print(C(0))
print(C(1))
print(C(3))

def C(n):
    if n == 0:
        return 1
    else:
        return C(n - 1) * 4
    
print(C(0))
print(C(1))
print(C(3))

def C(n):
    if n > 0:
        return C(n - 1) * 4
    return 1
    
print(C(0))
print(C(1))
print(C(3))

def C(n):
    return 1 if n == 0 else C(n - 1) * 4
    
print(C(0))
print(C(1))
print(C(3))

def C(n):
    return C(n - 1) * 4

print(C(0))
print(C(1))
print(C(3))

def C(n):
    if n > 0:
        return C(n) * 4
    return 1


print(C(0))
print(C(1))
print(C(3))

def build_binary_tree(n):
    if n == 0:
        return 1
    return [build_binary_tree(n - 1), build_binary_tree(n - 1)]

print(build_binary_tree(0))
print(build_binary_tree(1))
print(build_binary_tree(2))
print(build_binary_tree(3))

def build_binary_tree(n):
    if n == 0:
        return 1
    return [build_binary_tree(n - 1), build_binary_tree(max(n - 2, 0))]

print(build_binary_tree(0))
print(build_binary_tree(1))
print(build_binary_tree(2))
print(build_binary_tree(3))

