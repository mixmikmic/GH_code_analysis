dim = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

day = 0
sundays = 1

for year in range(1901,2001):
    
    for month in dim:
        
        day += month
        
        if year % 4 == 0 and month == 28:
            day += 1
        
        if day % 7 == 0:
            sundays += 1
            
print(sundays)

