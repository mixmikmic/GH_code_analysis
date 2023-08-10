# Stack implementation using python built in list
class Stack:
    def __init__(self):
        self.data = []
        
    def size(self):
        return len(self.data)
    
    # Push data to the top of the stack
    def push(self, data):
        self.data.append(data)
        
    # Pop the top element of the stack   
    def pop(self):
        return None if self.size() == 0 else self.data.pop()
    
    # Get the value of the top element without popping it
    def peek(self):
        return None if self.size() == 0 else self.data[-1]

s = Stack()
print(s.pop())
for char in tuple("abcdefg"):
    s.push(char)
for c in s.data[::-1]:
    print(c)
print("Size: {}\nTop element: {}".format(s.size(), s.peek()))
del s

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class Stack:
    def __init__(self):
        self.top = None
        self.size = 0
        
    def push(self, data):
        node = Node(data)
        if self.top is None:
            self.top = node
        else:
            node.next = self.top
            self.top = node
            
    def pop(self):
        if self.top is None:
            return None
        node = self.top
        self.top = self.top.next
        return node.data
    
    def peek(self):
        return self.top.data if self.top is not None else None
            
    def __iter__(self):
        current = self.top
        while current is not None:
            yield current.data
            current = current.next

s = Stack()
print(s.peek())
print(s.pop())
for char in tuple("abcdefg"):
    s.push(char)
print("Popped: {}".format(s.pop()))
for value in iter(s):
    print(value)
del s

