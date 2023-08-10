class List():
    def show(self):
        return str(self)

    
class Nil(List):
    def __init__(self):
        self.head = None
    
    def __str__(self):
        return "Nil"
    
    def __repr__(self):
        return "Nil"
    
    
class Cons(List):
    def __init__(self, head, tail):
        self.head = head
        self.tail = tail
    
    def __str__(self):
        return str(self.head) + "::" + str(self.tail) 
    
    def __repr__(self):
        return "Cons(" + str(self.head) + ", " + repr(self.tail) +")"
    
    
def list(*a):
    if len(a) == 0 :
        return Nil()
    else:
        return Cons(a[0], list(*a[1:]))



