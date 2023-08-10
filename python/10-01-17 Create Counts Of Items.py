from collections import Counter

# Create a counter of the fruits eaten today
fruit_eaten = Counter(['Apple', 'Apple', 'Apple', 'Banana', 'Pear', 'Pineapple'])

# View counter
fruit_eaten

# Update the count for 'Pineapple' (because you just ate an pineapple)
fruit_eaten.update(['Pineapple'])

# View the counter
fruit_eaten

# View the items with the top 3 counts
fruit_eaten.most_common(3)

