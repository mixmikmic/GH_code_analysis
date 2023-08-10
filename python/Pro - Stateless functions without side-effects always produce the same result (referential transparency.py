current_speaker = None


def register(name):
    
    global current_speaker
    current_speaker = name
    
    
def speak(text):
    
    print('[%s] %s' % (current_speaker, text))
    
    
register('John')
speak('Hello world!')
register('Carlos')
speak('Foobar!')

class Speaker():
    
    def __init__(self, name):
        
        self._name = name
        
    def speak(self, text):
        
        print('[%s] %s' % (self._name, text))
        

john = Speaker('John')
john.speak('Hello world!')
carlos = Speaker('Carlos')
carlos.speak('Foobar!')

def speak(speaker, text):
    
    print('[%s] %s' % (speaker, text))
    

john = 'John'
speak(john, 'Hello world!')
carlos = 'Carlos'
speak(carlos, 'Foobar!')

