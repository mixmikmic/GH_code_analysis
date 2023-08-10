def outer():
    
    def inner():
        
        print('I\'m inner')
    
    inner()

outer()

def outer():
    
    def inner():
        
        print('Inner:\t\t', x)
    
    print('Outer (before):\t', x)
    inner()
    print('Outer (after):\t', x)

    
x = 'global'
print('Global (before):', x)
outer()
print('Global (after): ', x)

def outer():
    
    def inner():
        
        nonlocal x
        x = 'inner'
        print('Inner:\t\t', x)
    
    x = 'outer'
    print('Outer (before):\t', x)
    inner()
    print('Outer (after):\t', x)

    
x = 'global'
print('Global (before):', x)
outer()
print('Global (after): ', x)

