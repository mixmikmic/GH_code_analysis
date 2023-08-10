x = int(input("Enter an integer: "))
y = int(input("Enter a second integer"))
if x%y == 0:
    print(x,"is divisible by",y) #This block will execute if the remainder of x/y is zero
else:
    print(x,"is not divisible by",y)

purchase_price = float(input("Enter the purchase price of the stock: "))
price_now = float(input("Enter the current price of the stock: "))
if price_now < purchase_price * 0.9:
    print("STOP LOSS: Sell the stock! ")
elif price_now > purchase_price * 1.2:
    print("PROFIT TAKING: Sell the stock!")
else:
    print("HOLD: Don't do anything!")

purchase_price = float(input("Enter the purchase price of the stock: "))
price_now = float(input("Enter the current price of the stock: "))
if price_now < purchase_price * 0.9:
    print("STOP LOSS: Sell the stock! ")
    print("You've lost",purchase_price-price_now,"Dollars per share")
elif price_now > purchase_price * 1.2:
    print("PROFIT TAKING: Sell the stock!")
    print("You've gained",price_now-purchase_price,"Dollars per share")

else:
    print("HOLD: Don't do anything!")
    print("Your unrealized profit is",price_now-purchase_price,"Dollars per share")
print("Hope you enjoyed this program!")

len("Hi there!")*2**3/4

5%2



