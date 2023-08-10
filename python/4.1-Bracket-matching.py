# Stack implementation using python built in list
class Stack:
    def __init__(self):
        self.data = []
        
    def size(self):
        return len(self.data)
    
    def push(self, data):
        self.data.append(data)
           
    def pop(self):
        return None if self.size() == 0 else self.data.pop()
    
    def peek(self):
        return None if self.size() == 0 else self.data[-1]
    
    def isEmpty(self):
        return (len(self.data) == 0)

def has_matching_brackets(statement, brackets=(("[", "]"), ("{", "}"), ("(", ")"))):
    bStack = Stack()
    for char in statement:
        for startBracket, endBracket in brackets:
            if char == startBracket:
                bStack.push(startBracket)
            elif char == endBracket:
                topChar = bStack.pop()
                if topChar is None or topChar != startBracket:
                    return False
    return bStack.isEmpty()

sl = ( 
   "{(foo)(bar)}[hello](((this)is)a)test", 
   "{(foo)(bar)}[hello](((this)is)atest", 
   "{(foo)(bar)}[hello](((this)is)a)test))" 
) 
for s in sl: 
   m = has_matching_brackets(s) 
   print("{}: {}".format(s, m)) 

