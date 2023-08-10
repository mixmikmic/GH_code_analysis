get_ipython().magic('matplotlib inline')
# ignore this "magic" command -- it's only necessary to setup this notebook... 

import matplotlib.pyplot as plt

xvals = [0, 1, 2, 3]
yvals = [20, 10, 50, -15]
plt.bar(xvals, yvals)

fig, ax = plt.subplots()
ax.plot([1,2,3], [40, 20, 33])

fig, ax = plt.subplots()
xvals = [42, 8, 33, 25, 39]
yvals = [30, 22, 42, 9, 16]
ax.scatter(xvals, yvals)

fig, ax = plt.subplots()
xvals = [0, 1, 2, 3, 4]
y1 = [20, 8, 12, 24, 18]
y2 = [9, 1, 8, 15, 26]
ax.plot(xvals, y1)
ax.plot(xvals, y2)

fig, ax = plt.subplots()
xvals = [0, 1, 2, 3, 4]
y1 = [20, 8, 12, 24, 18]
y2 = [9, 1, 8, 15, 26]
ax.scatter(xvals, y1)
ax.plot(xvals, y2)

yvals = [10, 20, 30]
fig, ax = plt.subplots()
ax.pie(yvals)

a = [10, 20, 30]
b = [0.2, 2, 1]

fig, ax = plt.subplots()
ax.pie(a, b)

xvals = [0, 1, 2, 3, 4]
y1 = [50, 40, 30, 20, 10]
fig, ax = plt.subplots()
ax.bar(xvals, y1)

xvals = [0, 1, 2, 3, 4]
y1 = [50, 40, 30, 20, 10]
y2 = [10, 18, 23, 7, 26]
fig, ax = plt.subplots()
ax.bar(xvals, y1)
ax.bar(xvals, y2, color='orange')

xvals = [0, 1, 2, 3, 4]
y1 = [50, 40, 30, 20, 10]
y2 = [10, 18, 23, 7, 26]
fig, ax = plt.subplots()
ax.bar(xvals, y1)
ax.bar(xvals, y2, color='orange', bottom=y1)

x1 = [0, 1, 2, 3, 4]
y1 = [50, 40, 30, 20, 10]
x2 = [ 10, 11, 12, 13, 14]
y2 = [10, 18, 23, 7, 26]
fig, ax = plt.subplots()
ax.bar(x1, y1)
ax.bar(x2, y2, color='orange', bottom=y1)

# Step 1

xvals = [0, 1]
yvals = [42, 25]

fig, ax = plt.subplots()
ax.bar(xvals, yvals)

# Step 1 & 2

xvals = [0, 1]
yvals = [42, 25]

fig, ax = plt.subplots()
# note that I specify the `align` argument in the `bar()` call:
ax.bar(xvals, yvals, align='center')
ax.set_xticks(xvals)

# Steps 1,2,3
# Step 1 & 2

xlabels = ['apples', 'oranges']
xvals = [0, 1]
yvals = [42, 25]

fig, ax = plt.subplots()
# note that I specify the `align` argument in the `bar()` call:
ax.bar(xvals, yvals, align='center')
ax.set_xticks(xvals)
ax.set_xticklabels(xlabels)



