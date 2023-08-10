class Counting(object):
    
    def __init__(self, max_value):
        self.max_value = max_value
        self._counts = [0] * (self.max_value + 1)
    
    def sort(self, iterable):
        result = [0] * (len(iterable) + 1)
        for value in iterable:
            self._counts[value] += 1
        for index in range(1, len(self._counts)):
            self._counts[index] += self._counts[index - 1]
        for index in list(range(len(iterable)))[::-1]: # we iterate backwards for stable order
            value = iterable[index]
            result[self._counts[value]] = value
            self._counts[value] -= 1
        return result[1:] # because python is 0 indexed, and we've been pretending otherwise
                  

array = [5,4,3,2,1,0]
ceiling = max(array)
Counting(ceiling).sort(array)

from sorts import insertion_sort

class Bucket(object):
        
    def sort(self, iterable):
        bucket_list = [[] for i in range(10)]
        for value in iterable:
            index = int(value * 10)
            bucket_list[index].append(value)
        result = []
        for bucket in bucket_list:
            result.extend(bucket)
        return insertion_sort(result)

array = [0.876, 0.589, 0.567, 0.456, 0.345, 0.234, 0.123, 0]
Bucket().sort(array)



