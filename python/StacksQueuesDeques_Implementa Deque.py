class Deque(object):
    ## Going to make end of the list = front, and 0th = rear.
    
    def __init__(self):
        self.items = []
        
    def isEmpty(self):
        # OR len(self.items) == 0
        return self.items == []
    
    def size(self):
        return len(self.items)
    
    def addFront(self,item):
        self.items.append(item)
    
    def addRear(self,item):
        self.items.insert(0,item)
    
    def removeFront(self):
        self.items.pop()
    
    def removeRear(self):
        self.items.pop(0)



