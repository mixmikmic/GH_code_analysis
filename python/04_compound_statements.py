def percent_difference(x, y):
    difference = abs(x-y)
    return 100 * difference / x

result = percent_difference(40, 50)

result

def absolute_value(x):
    """Compute abs(x)."""
    
    if x > 0:
        return x
    elif x == 0: 
        return 0 
    else:
        return -x

absolute_value(-5)

absolute_value(6)

absolute_value(0)

absolute_value(-0.1)

# Another way to write this: 
def absolute_value(x):
    """Compute abs(x)."""
    if not x < 0:
        return x
    else:
        return -x

# we can also nest ifs 
a = 1
b = 2

if a > 5:
    if b > 6: 
        print "a is more than 5 and b is more than 6"
    else: 
        print "a is more than 5 BUT b is not more than 6"    
        
else: 
    print "a is not more than 5"

print "done"    

def is_weekday(day, month, year): 
    import datetime
    return datetime.datetime(year=year, month=month, day=day).weekday() < 5

if is_weekday(30, 9, 2017):
    print "It's a weekday."
else: 
    print "Weekend!!"

