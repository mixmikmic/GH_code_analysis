get_ipython().magic('matplotlib inline')
import seaborn
from matplotlib import pyplot as plt
from collections import Counter

num_friends = [5, 3, 10, 11, 11, 9, 4, 10, 1, 3, 7]

# For this small dataset, we start plotting this data, just to understand and have an intuition of the distribution.
friend_counts = Counter(num_friends)

xs = range(15)
ys = [friend_counts[x] for x in xs]

plt.bar(xs, ys)
plt.axis([0, 12, 0, 5])
plt.title("Histogram of Friend Counts")
plt.xlabel("# of friends")
plt.ylabel("# of people")
plt.show()

num_points = len(num_friends)

# These are special cases of knowing the values at certain position, let's sort it then!
largest = max(num_friends)
smallest = min(num_friends)

sorted_values = sorted(num_friends)

assert smallest == sorted_values[0]
assert largest == sorted_values[-1]

print "dataset size: %s" % num_points
print "largest: %s" % largest
print "smallest: %s" % smallest




