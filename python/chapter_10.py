class Stack(object):
    
    def __init__(self):
        self.data = [None]
        self.head = 0
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __setitem__(self, index, value):
        self.data[index] = value
        
    def pop(self):
        if self.head == 0:
            raise BaseException("Stack underflow")
        self.head -= 1
        return self[self.head + 1]
    
    def push(self, value):
        if self.head == len(self.data) - 1:
            self.data += [None] * (1 + self.head//2)
        self.head += 1
        self[self.head] = value

stack = Stack()
for i in range(10):
    stack.push(i)
stack.pop()

class Queue(object):
    
    def __init__(self):
        self.data = [None]
        self.head = 0
        self.tail = 0
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __setitem__(self, index, value):
        self.data[index] = value
    
    def dequeue(self):
        if self.head == self.tail:
            raise BaseException("Queue empty")
        self.head += 1
        return self[self.head]
    
    def enqueue(self, value):
        if self.tail == len(self.data) - 1:
            self.data += [None] * (1 + self.tail//2)
        self.tail += 1
        self[self.tail] = value

queue = Queue()
for i in range(10):
    queue.enqueue(i)
queue.dequeue()

from collections import namedtuple

class LinkedList(object):
    
    def __init__(self):
        self.data = [[0, 'sentinel', 0]]
        self.empties = []
        
    def __iter__(self):
        index = 0
        while index != self.data[0][0]:
            index = self.data[index][2]
            value = self.data[index][1]
            yield index, value

    def delete(self, key):
        prev_item = self.data[key][0]
        next_item = self.data[key][2]
        self.data[prev_item][2] = next_item
        self.data[next_item][0] = prev_item
        self.empties.append(key)
        
    def insert(self, item):
        if self.empties:
            key = self.empties.pop()
        else:
            key = len(self.data)
            self.data.append([None, None, None])
        self.data[key][0] = self.data[0][0]
        self.data[key][1] = item
        self.data[key][2] = 0
        self.data[self.data[0][0]][2] = key
        self.data[0][0] = key
        
    def search(self, value):
        for index, result in self:
            if result == value:
                return index
        return None

ll = LinkedList()
for name in ['alice', 'bob', 'cat', 'dillon']:
    ll.insert(name)
ll.search('bob')

ll.delete(2)
print(ll.data)

ll.insert('eve')
print(ll.data)



