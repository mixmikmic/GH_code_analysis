def profit (stock_prices):
    """  Go through list once, tracking minimum found so far, and max profit so far.  Return max profit.
    """
    # Take care of edge case where list has no profit calculatability.  ie, less than 2 prices.
    if len(stock_prices) < 2:
        raise Exception("Must provide at least 2 prices!")
    
    # Minimum tracker starting from first element
    min_price = stock_prices[0]
    
    # Profit tracker
    max_profit = 0
    
    for price in stock_prices[1:]:
        
        # update min price if current price is lower
        min_price = min(min_price, price)
        
        # update max profit
        max_profit = max (price-min_price, max_profit)
    
    return max_profit 

stock_prices = [1,5,3,7,23,12,2]
profit(stock_prices)

def profit2(stock_prices):
    """ Same as before, BUT start max_profit from #1-#0.  Still works with a normal up/down day."""
    # Take care of edge case where list has no profit calculatability.  ie, less than 2 prices.
    if len(stock_prices) < 2:
        raise Exception("Must provide at least 2 prices!")
    
    # Minimum tracker starting from first element
    min_price = stock_prices[0] 
    
    # Profit tracker
    max_profit = stock_prices[1]-stock_prices[0]  ## DIFFERENT THAN PREVIOUS $!!!!!!!!!
    
    for price in stock_prices[2:]:
        
        ## RE-ORDER FINDING MAX PROFIT AND MIN, $!!!!!!!!!
        ## TO ALLOW FOR NEGATIVE PROFITS        $!!!!!!!!!
        
        # update max profit
        max_profit = max (price-min_price, max_profit)
        
        # update min price if current price is lower
        min_price = min(min_price, price)       
    
    return max_profit 

stock_prices = [1,5,3,7,23,12,2]
print profit2(stock_prices)
print 
stock_prices = [23,19,7,3,1]
print profit2(stock_prices)

def profit_brute(stock_prices):
    """ For each element, check for max to its right and keep track of global max profit."""
    
    # Profit tracker
    max_profit = stock_prices[1]-stock_prices[0]
    
    # loop through list till one before end, as comparing at least one ahead inside loop
    for i in xrange(2, len(stock_prices)-1):
        
        # Find max on this stock's right
        max_right = max(stock_prices[i+1:])
        max_profit = max(max_profit, max_right-stock_prices[i])
    
    return max_profit

stock_prices = [1,5,3,7,23,12,2]
print profit_brute(stock_prices)
print 
stock_prices = [23,19,7,3,1]
print profit_brute(stock_prices)

