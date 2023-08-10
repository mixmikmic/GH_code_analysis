get_ipython().run_cell_magic('javascript', '', '\nclass Queue {\n  constructor(){\n    this._storage = {};\n    this._start = -1; //replicating 0 index used for arrays\n    this._end = -1; //replicating 0 index used for arrays\n  }\n\n  enqueue(val){\n    this._storage[++this._end] = val; \n  }\n\n  dequeue(){\n    if(this.size()){ \n      let nextUp = this._storage[++this._start];\n      delete this._storage[this._start];\n\n      if(!this.size()){ \n        this._start = -1;\n        this._end = -1; \n      }\n\n      return nextUp;\n    }\n  }\n        \n  size(){\n   return this._end - this._start;\n  }\n} //end Queue\n\nvar microsoftQueue = new Queue();\n\nmicrosoftQueue.enqueue("{user: ILoveWindows@gmail.com}");\nmicrosoftQueue.enqueue("{user: cortanaIsMyBestFriend@hotmail.com}");\nmicrosoftQueue.enqueue("{user: InternetExplorer8Fan@outlook.com}");\nmicrosoftQueue.enqueue("{user: IThrowApplesOutMyWindow@yahoo.com}");\n\nvar sendTo = function(s){\n    element.append(s + " gets a Surface Studio<br />");\n}\n\n//Function to send everyone their Surface Studio!\nlet sendSurface = recepient => {\n   sendTo(recepient);\n}\n\n//When your server is ready to handle this queue, execute this:\nwhile(microsoftQueue.size() > 0){\n  sendSurface(microsoftQueue.dequeue());\n}')

class Queue:
    
    def __init__(self):
        self._storage = {}
        self._start = -1   # replicating 0 index used for arrays
        self._end = -1     # replicating 0 index used for arrays
        
    def size(self):
        return self._end - self._start

    def enqueue(self, val):
        self._end += 1
        self._storage[self._end] = val

    def dequeue(self):
        if self.size():
            self._start += 1
            nextUp = self._storage[self._start]
            del self._storage[self._start]
    
            if not self.size(): 
                self._start = -1
                self._end = -1
            return nextUp
        
microsoftQueue = Queue()

microsoftQueue.enqueue("{user: ILoveWindows@gmail.com}")
microsoftQueue.enqueue("{user: cortanaIsMyBestFriend@hotmail.com}")
microsoftQueue.enqueue("{user: InternetExplorer8Fan@outlook.com}")
microsoftQueue.enqueue("{user: IThrowApplesOutMyWindow@yahoo.com}") 

def sendTo(recipient):
    print(recipient, "gets a Surface Studio")

# Function to send everyone their Surface Studio!
def sendSurface(recepient):
   sendTo(recepient)

# When your server is ready to handle this queue, execute this:

while microsoftQueue.size() > 0:
  sendSurface(microsoftQueue.dequeue())

get_ipython().run_cell_magic('javascript', '', 'var sendTo = function(s){\n    element.append(s + "<br />");\n}\n\n//Function to send everyone their Surface Studio!\nlet sendSurface = recepient => {\n   sendTo(recepient);\n}\n\nfunction recipe(ingredient0, ingre1, ing2, ...more){\n    sendSurface(ingredient0 + " is one ingredient.");\n    sendSurface(more[1] + " is another.");\n}\nrecipe("shrimp", "avocado", "tomato", "potato", "squash", "peanuts");')

def recipe(ingr0, *more, ingr1, meat="turkey", **others):
    print(more)
    print(others)
    
recipe("avocado", "tomato", "potato", ingr1="squash", dessert="peanuts", meat = "shrimp")

