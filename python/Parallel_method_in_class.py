from joblib import Parallel, delayed

# let's re-write the exact function before into a class
class square_class:
    def square_int(self, i):
        return i * i
     
    def run(self, num):
        results = []
        results = Parallel(n_jobs= -1, backend="threading")            (delayed(self.square_int)(i) for i in num)
        print(results)

square_int = square_class()
square_int.run(num = range(10))

def unwrap_self(arg, **kwarg):
    return square_class.square_int(*arg, **kwarg)

class square_class:
    def square_int(self, i):
        return i * i
     
    def run(self, num):
        results = []
        results = Parallel(n_jobs= -1, backend="threading")            (delayed(unwrap_self)(i) for i in zip([self]*len(num), num))
        print(results)

square_int = square_class()
square_int.run(num = range(10))

num = range(10)
# note, I use a string "self" here to print it out
print(zip(["self"]*len(num), num))

