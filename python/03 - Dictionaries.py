# An empty dictionary
dict()
{}

# Live example

jons_shoes = {'for bike riding':['Left bike shoe','right bike shoe']
              ,'for running':'runners',
              'for pretending to be from colorado':'sandals',
              'for pretending to be Rafa':'tennis shoe'}

# An example of a dictionary with keys and values

released = {    
    "iphone" : 2005,
    "iPhone Pod": 3010,
    "iphone3G" : str(2004),    
    "iphone3GS": str(2009),
    "iPhone Smart": 2001,
    "iphone4" : 2010,
    "iphone4S": '2010',
    'iPhone 4SS' : 2015
}

print(released)

# Accessing your dictionary
jons_shoes['for running']

released['iphone']

# Dictionaries are mutable (they can be changed)
jons_shoes['for running']

jons_shoes['for running'] = 'Fancy new running shoes'
jons_shoes['for running']

# Adding new elements
released['temporary'] = "Temporary record"

released



# Deleting elements
# same as lists, "pop" gives you the deleted element
released.pop("temporary")

# While del removes the key and value from the dictionary. 
del released["iphone4S"]











temp_dict["temp"] = "temporary record"

temp_dict.values()



temp_dict.keys()











