from objects import DoublyLinkedList

class ChainedHashTable(object):
    
    def __init__(self, size):
        self.size = size
        self.node = DoublyLinkedList
        self.data = [self.node() for i in range(size)]
        
    def delete(self, item):
        key = self.shorthash(item)
        self.data[key].delete(item)
        
    def insert(self, item):
        key = self.shorthash(item)
        self.data[key].insert(item)
        
    def search(self, item):
        key = self.shorthash(item)
        return self.data[key].search(item)
        
    def shorthash(self, item):
        return hash(item) % self.size
     

table = ChainedHashTable(10)
for name in ['alice', 'bob', 'cat', 'dillon']:
    table.insert(name)
print(table.data)

table.search('bob'), table.delete('alice'), table.search('bob')

class AddressedHashTable(object):
    
    def __init__(self, size):
        self.size = size
        self.data = [None for i in range(self.size)]
        
    def delete(self, item):
        key = self.search(item)
        self.data[key] = 'DELETED'
        
    def insert(self, item):
        for key in self.probhash(item):
            if self.data[key] in [None, 'DELETED']:
                self.data[key] = item
                break
        
    def probhash(self, item):
        raise NotImplementedError
        
    def search(self, item):
        for key in self.probhash(item):
            if self.data[key] == None:
                return None
            elif self.data[key] == item:
                return key

class LinearHashTable(AddressedHashTable):
    
    def __init__(self, size):
        super(LinearHashTable, self).__init__(size)
        
    def probhash(self, item):
        for probe in range(self.size):
            yield (hash(item) + probe) % self.size

table = LinearHashTable(10)
for name in ['alice', 'bob', 'cat', 'dillon']:
    table.insert(name)
print(table.data)

table.search('bob'), table.delete('alice'), table.search('bob')

class DoubleHashTable(AddressedHashTable):
    
    def __init__(self, size):
        super(DoubleHashTable, self).__init__(size)
        
    def probhash(self, item):
        for probe in range(self.size):
            yield (hash(item) + probe * hash(item)) % self.size

table = DoubleHashTable(10)
for name in ['alice', 'bob', 'cat', 'dillon']:
    table.insert(name)
print(table.data)

table.search('bob'), table.delete('alice'), table.search('bob')



