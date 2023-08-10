import Tkinter
import tkFileDialog
import csv

# Bring in the dataset using Tkinter dialog box
root = Tkinter.Tk()
root.withdraw()
filename = tkFileDialog.askopenfilename(parent=root)
filename_open = open(filename)
dataset = list(csv.reader(filename_open))

dataset[0] 

# remove the headers for simplicity
del dataset[0]

dataset[0] 

# Find the sum of X
numlist = [float(i[1]) for i in dataset]
x_sum = sum(numlist)
print(x_sum)

# Find the mean of X
x_mean = x_sum / len(dataset)
print(x_mean)

# Find the sum of Y
numlist = [float(i[2]) for i in dataset]
y_sum = sum(numlist)
print(y_sum)

# Find the mean of Y
y_mean = y_sum / len(dataset)
print(y_mean)

# Find the sum of XY
numlist = [float(i[1])*float(i[2]) for i in dataset]
xy_sum = sum(numlist)
print(xy_sum)

# Find the slope and round to nearest hundreth
x_b = [float(i[1])-x_mean for i in dataset]
x_b_sum = sum(x_b)
y_b = [float(i[2])-y_mean for i in dataset]
y_b_sum = sum(y_b)

# Slope numerator and denominator
slope_n = [((float(i[1])-x_mean) * (float(i[2])-y_mean)) for i in dataset]
slope_n = sum(slope_n)
slope_d = [((float(i[1])-x_mean)**2) for i in dataset]
slope_d = sum(slope_d)

slope = (slope_n / slope_d)

print "Slope: ", slope_n, "/", slope_d, "=", round(slope, 2)

# Calculate the intercept and round to nearest hundreth
b_int = round(y_mean - slope * x_mean, 2)
print b_int

# Print the model to the console:
print "bo = ", round(slope,2), "br +", b_int

