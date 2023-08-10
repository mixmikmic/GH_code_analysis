# A very straight forward hash function for string keys
def hash(str_key):
    return sum([ord(char) for char in str_key])

def test_hash(hash_func, str_key):
    # f-string is a formatted string
    return f'Hash value of "{str_key}" is {hash_func(str_key)}.' 
print(test_hash(hash, 'hello')) 
print(test_hash(hash, 'world'))
print(test_hash(hash, 'hello world'))
print(test_hash(hash, 'gello xorld'))

# A better version of the hash function
def better_hash(str_key):
    return sum([ord(char) * (index + 1) for (index, char) in enumerate(str_key)])

print(test_hash(better_hash, 'hello')) 
print(test_hash(better_hash, 'world'))
print(test_hash(better_hash, 'hello world'))
print(test_hash(better_hash, 'gello xorld'))

class HashTable:
    def __init__(self, table_size = 256):
        self.table_size = table_size
        self.data = [None] * table_size
        
    # The hash function; 
    # this function can be anything you want, 
    # as long as it returns a integer that is within the range(0, talbe_size)
    def _hash(self, str_key):
        v = sum([ord(char) * (index + 1) for (index, char) in enumerate(str_key)])
        return v % self.table_size
        
    def add(self, key, value):
        hash_value = self._hash(key)
        if self.data[hash_value] is None:
            self.data[hash_value] = [(key, value)]
        else:
            self.data[hash_value].append((key, value))
            
    def get(self, key):
        hash_value = self._hash(key)
        if self.data[hash_value] is None:
            return None
        for stored_key, value in self.data[hash_value]:
            if key == stored_key:
                return value
        return None

phone_book = HashTable(512)
phone_book.add("tom", "123-467-0000")
phone_book.add("jack", "777-888-9394")
phone_book.add("mary", "666-0303-2222")
print(f"tom\'s phone number is {phone_book.get('tom')}.")
print(f"mary\'s phone number is {phone_book.get('mary')}.")
print(f"jerry\'s phone number is {phone_book.get('jerry')}.")

class HashTable:
    def __init__(self, table_size = 256):
        self.table_size = table_size
        self.data = [None] * table_size
        
    # The hash function; 
    # this function can be anything you want, 
    # as long as it returns a integer that is within the range(0, talbe_size)
    def _hash(self, str_key):
        v = sum([ord(char) * (index + 1) for (index, char) in enumerate(str_key)])
        return v % self.table_size
        
    def add(self, key, value):
        hash_value = self._hash(key)
        if self.data[hash_value] is None:
            self.data[hash_value] = [(key, value)]
        else:
            self.data[hash_value].append((key, value))
            
    def get(self, key):
        hash_value = self._hash(key)
        if self.data[hash_value] is None:
            return None
        for stored_key, value in self.data[hash_value]:
            if key == stored_key:
                return value
        return None
    
    def __setitem__(self, key, value):
        self.add(key, value)
    
    def __getitem__(self, key):
        return self.get(key)

my_final_report = HashTable()
my_final_report["math"] = 90
my_final_report["advanced programming"] = 85
my_final_report["operating system"] = 75
my_final_report["principles of programming language"] = 50  # Opps
print(f'I got {my_final_report["math"]} for my math final.')
print(f'I got {my_final_report["principles of programming language"]} for my POPL final.')
print(f'My grade for history is {my_final_report["history"]} because I never took it.')

