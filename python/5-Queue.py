from collections import deque

# Queue implementation using deque (Double ended queue)
class Queue:
    def __init__(self):
        self.data = deque()
        
    def size(self):
        return len(self.data)
        
    def enqueue(self, data):
        self.data.append(data)
        
    def dequeue(self):
        return None if self.size() == 0 else self.data.popleft()
    
    def peek(self):
        return None if self.size() == 0 else self.data[0]

q = Queue()
for language in ("python", "java", "c++", "kotlin", "go", "javascript"):
    q.enqueue(language)
print(q.size())
print(q.peek())
print(q.dequeue())
print(q.size())
print(q.peek())
del q

