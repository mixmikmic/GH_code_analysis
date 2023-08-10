class Stack:
    
    # variables 
    # __size__ = 0 # another way to implement size()
    items = None
    
    # functions
    def __init__(self):
        self.items = []
    
    def isEmpty(self):
        return self.items == []
    
    def size(self):
        return len(self.items)
        
    def push(self, item):
        self.items.append(item) # The method append() appends a passed obj into the existing list.

    def pop(self):
        return self.items.pop() # The method pop() removes and returns last object or obj from the list.
    
    def peek(self):
        return self.items[len(self.items)-1]

s = Stack()
print(s.isEmpty())
print(s.size())
print()
s.push(1)
s.push(2)
s.push(3)
print(s.isEmpty())
print(s.size())
print()
print(s.pop())
print(s.size())
print(s.peek())

# Are they balanced?

par_li = [

'(()()()())',

'(((())))',

'(()((())()))',

'((((((())',

'()))',

'(()()(()',

]

def checkBalancedParentheses(test_string):
    s = Stack()
    for c in test_string:
        if c == "(":
            s.push("(")
        elif c == ")":
            if s.isEmpty():
                return False
            s.pop()
        else:
            pass
            # ?
    
    if s.isEmpty():
        return True
    else:
        return False

for test in par_li:
    print(checkBalancedParentheses(test))

sym_li = [

'{ { ( [ ] [ ] ) } ( ) }',

'[ [ { { ( ( ) ) } } ] ]',

'[ ] [ ] [ ] ( ) { }',

'( [ ) ]',

'( ( ( ) ] ) )',

'[ { ( ) ]',
   
]

def checkBalancedSym(test_string):
    s = Stack()
    for c in test_string:
        if c in "([{":
            s.push(c)
        elif c == ' ': # remove this if input string is condensed.
            pass
        else:
            if s.isEmpty():
                return False
            if s.pop() != {")":"(", "]":"[", "}":"{"}[c]:
                return False
    
    if s.isEmpty():
        return True
    else:
        return False

for sym in sym_li:
    print(checkBalancedSym(sym))

