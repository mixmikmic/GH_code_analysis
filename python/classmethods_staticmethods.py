class House: 
    '''define the House class'''
    #define class variable value_increase 
    value_increase = 0.10 
    
    def __init__(self, color, size, location, value):
        # initialize the constructor 
        self.color = color 
        self.size = size 
        self.location = location 
        self.value = value 
    
    def tagline(self):
        #tagline from house data  
        tag = 'A beautiful {} sized, {} home in {}'.format(self.size, 
                                                    self.color, 
                                                    self. location)
        return tag 

#how input data may appear 
house_data = 'pink-medium-San Francisco-1000000'

#parse data to format for instantiation 
color, size, location, value = house_data.split('-')
print (color, size, location, value)

#instantiate class 
house1 = House(color, size, location, value)

#check that it worked 
print (house1.tagline())

class House: 
    '''define the House class'''
    #define class variable value_increase 
    value_increase = 0.10 
    
    def __init__(self, color, size, location, value):
        # initialize the constructor 
        self.color = color 
        self.size = size 
        self.location = location 
        self.value = value 
        
    def tagline(self):
        #tagline from house data  
        tag = 'A beautiful {} sized, {} home in {}'.format(self.size, 
                                                    self.color, 
                                                    self. location)
        return tag 

    @classmethod #decorator 
    def from_string(cls, data):
        #parse the data
        color, size, location, value = data.split('-')
        #put parsed data into class object 
        return cls(color, size, location, value)
    
#how input data may appear 
house_data = 'pink-medium-San Francisco-1000000'

#parse and instantiate class using classmethod
house1 = House.from_string(house_data)

#check that it worked 
print (house1.tagline())

import datetime

#use a current time object from time.time()
date1 = datetime.datetime.today()

#use ordinal time formatting 
date2 = datetime.datetime.fromordinal(700001)

#use a timestamp object
date3 = datetime.datetime.fromtimestamp(700001)

print (date1)
print (date2)
print (date3)

class House: 
    '''define the House class'''
    #define class variable value_increase 
    value_increase = 0.10 
    
    def __init__(self, color, size, location, value):
        # initialize the constructor 
        self.color = color 
        self.size = size 
        self.location = location 
        self.value = value 
        
    @staticmethod 
    def is_in_budget(price): 
        if price<=500000: 
            return True 
        else: 
            return False 
        
print (House.is_in_budget(400000))
print (House.is_in_budget(1000000))



