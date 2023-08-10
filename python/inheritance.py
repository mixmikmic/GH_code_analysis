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
        
    def good_year_increase(self):
        #increase value of home by value_increase 
        self.value = self.value* (1 + self.value_increase) 
        
    def tagline(self):
        #tagline from house data  
        tag = 'A beautiful {} sized, {} home in {}'.format(self.size, 
                                                    self.color, 
                                                    self. location)
        return tag 
    

class Apartment(House): 
    pass 

#create a parent instance
house1 = House ('blue', 'small', 'Paris', '300000')
print (house1.tagline())

#create a child instance
apartment1 = Apartment('blue', 'small', 'Paris', '300000')
print (apartment1.tagline())

print (help(Apartment))

class House: 
    '''define the House class'''
    #define class variable value_increase 
    value_increase = 0.10 
    
    def __init__(self, color, size, location, value):
        # initialize the constructor 
        self.color = color 
        self.size = size 
        self.location = location 
        self.value = int(value)
        
    def good_year_increase(self):
        #increase value of home by value_increase 
        self.value = self.value* (1 + self.value_increase) 
        
    def tagline(self):
        #tagline from house data  
        tag = 'A beautiful {} sized, {} home in {}'.format(self.size, 
                                                    self.color, 
                                                    self. location)
        return tag 
    

class Apartment(House): 
    #reduce year-to-year value increase 
    value_increase = 0.02 

    
#create a parent instance
house1 = House ('blue', 'small', 'Paris', '300000')
house1.good_year_increase()
print ('house increase: ', house1.value)

#create a child instance
apartment1 = Apartment('blue', 'small', 'Paris', '300000')
apartment1.good_year_increase()
print ('apartment increase: ', apartment1.value)

#original parent House class does not change 
class House: 
    '''define the House class'''
    #define class variable value_increase 
    value_increase = 0.10 
    
    def __init__(self, color, size, location, value):
        # initialize the constructor 
        self.color = color 
        self.size = size 
        self.location = location 
        self.value = int(value)
        
    def good_year_increase(self):
        #increase value of home by value_increase 
        self.value = self.value* (1 + self.value_increase) 
        
    def tagline(self):
        #tagline from house data  
        tag = 'A beautiful {} sized, {} home in {}'.format(self.size, 
                                                    self.color, 
                                                    self. location)
        return tag 
    

class Apartment(House): 
    #reduce year-to-year value increase 
    value_increase = 0.02 
    
    #copy House init 1st line, add apt_num, gym 
    def __init__(self, color, size, location, value, apt_num, gym):
        
        #copy __init__ attributes from House we want to keep 
        super().__init__(color, size, location, value)
        
        #init new child attributes 
        self.apt_num = apt_num
        self.gym = gym 

#create a child instance
#its starting to get long, so I'll stack 
apartment1 = Apartment('blue',
                       'small',
                       'Paris',
                       20000, 
                       '3B', 
                       True)

#print out some attributes 
print (apartment1.gym)
print (apartment1.value)
print (apartment1.apt_num)

class Condo(House):
    '''subclass Condo inheriting from House'''
    
    value_increase = 0.8
    
    def __init__(self, color, size, location, value, pool, ocean_access): 
        super().__init__(color, size, location, value)
        
        self.pool = pool
        self.ocean_access = ocean_access
        
    def inc_value(self): 
        if self.pool: 
            self.value *= (1+self.value_increase)

        if self.ocean_access: 
            self.value *= (1+self.value_increase)

        
condo1 = Condo('red', 'medium', 'Hawaii', '150000', False, True)
print (condo1.color)
print (condo1.value)
print (condo1.ocean_access)
print (condo1.pool)


condo1.inc_value()
print (condo1.value)

condo1.__dict__



