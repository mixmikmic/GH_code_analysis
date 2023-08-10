def TimeForHouse(s,R,p):
    '''Inputs:  s = Initial Sum
                R = Required Sum
                p = Annual Interest Rate
        Outputs: Number of years to reach required sum, R.
                '''
    year = 0                      
    money = s                     
#     print(year, '\t', money)      # Uncomment to print to screen year 0's value
    
    # Loop through years until money is greater than or equal to the required sum
    while money < R:
        money = money + money*p   # New amount is previous amount + interest added
        money = round(money,2)    # Round to two decimal places and carry this value forward
        year +=1                  # Add next year
#         print(year, '\t', money)  # Uncomment to print to screen each year's value
    return year                   # Return year

TimeForHouse(100000,1400000,0.08)



