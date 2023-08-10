class Net(object):
    
    def __init__(self, name, n_layers):
        self.name = name
        self.n_layers = n_layers
        
    def getName(self):
        return self.name
    
    def getLayers(self):
        return self.n_layers
    
    def __str__(self):
        return "Define a %s of %d layers" % (self.name, self.n_layers)

fnn = Net("FNN", 3)
print "This is the standard way to get he number of layers: %d" % fnn.getLayers()
print "By passing the instance, we get the number of layers: %d" % Net.getLayers(fnn)

print fnn

class CNN(Net):
    
    def __init__(self, name, n_layers, n_filters):
        Net.__init__(self, name, n_layers)
        self.fileters = n_filters
        
    def getFilters(self):
        return self.fileters

cnn = CNN("Simple_CNN", 2, 5) # define a CNN with 2 layers, and 5 filters
print "This is %s with %d layers and %d filters" % (cnn.getName(), cnn.getLayers(), cnn.getFilters())

print cnn

class RNN(Net):
    def __init__(self, *argv):
        super(RNN, self).__init__(*argv)
        self.timesteps = 0
       
    def set_timesteps(self, T):
        self.timesteps = T
        
    def get_timesteps(self):
        return self.timesteps
        

rnn = RNN("Simple RNN", 2)

print "This is %s with %d layers and %d time steps" % (rnn.getName(), rnn.getLayers(), rnn.get_timesteps())

rnn.set_timesteps(10)
print "This is %s with %d layers and %d time steps" % (rnn.getName(), rnn.getLayers(), rnn.get_timesteps())

print rnn

def test_var_args(f_arg, *argv):
    print "first normal arg:", f_arg
    for arg in argv:
        print "another arg through *argv :", arg



def greet_me(**kwargs):
    if kwargs is not None:
        for key, value in kwargs.iteritems():
            print "%s == %s" %(key,value)

            
test_var_args('yasoob','python','eggs','test')

greet_me(name1="yasoob", name2="python")



