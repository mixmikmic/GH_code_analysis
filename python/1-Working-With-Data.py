with open("datasets/Iris.csv") as csv_file:
    for row in csv_file:
        print(row)

import csv

with open("datasets/Iris.csv", newline="") as csv_file:
    reader = csv.reader(csv_file, delimiter=",")
    for row in reader:
        print(row)

with open("datasets/Iris.csv", newline="") as csv_file:
    reader = csv.reader(csv_file, delimiter=",")
    next(reader)  # Skip the first row
    for row in reader:
        print(row[0], "\t", row[5])  # Format the output for easier reading
        

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19890722)
data = np.random.randn(2, 100)

fig = plt.figure()
plt.scatter(data[0], data[1])

plt.show()

import csv

iris_species = []
iris_sepal_length = []
iris_sepal_width = []
iris_petal_length = []
iris_petal_width = []

with open("datasets/Iris.csv", newline="") as csv_file:
    reader = csv.reader(csv_file, delimiter=",")
    next(reader)  # Skip the first row
    for row in reader:
        iris_category = 0
        if row[5] == "Iris-setosa":
            iris_category = 1
        elif row[5] == "Iris-versicolor":
            iris_category = 2
        elif row[5] == "Iris-virginica":
            iris_category = 3
        iris_species.append(iris_category)
        iris_sepal_length.append(row[1])
        iris_sepal_width.append(row[2])
        iris_petal_length.append(row[3])
        iris_petal_width.append(row[4])

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].scatter(iris_species, iris_sepal_length)
axs[1, 0].scatter(iris_species, iris_sepal_width)
axs[0, 1].scatter(iris_species, iris_petal_length)
axs[1, 1].scatter(iris_species, iris_petal_width)

plt.show()

iris_sepal_length_dict = {1: [], 2: [], 3: []}
iris_sepal_width_dict =  {1: [], 2: [], 3: []}
iris_petal_length_dict = {1: [], 2: [], 3: []}
iris_petal_width_dict =  {1: [], 2: [], 3: []}

i = 0
for individual in iris_species:
    iris_sepal_length_dict[individual].append(iris_sepal_length[i])
    iris_sepal_width_dict[individual].append(iris_sepal_width[i])
    iris_petal_length_dict[individual].append(iris_petal_length[i])
    iris_petal_width_dict[individual].append(iris_petal_width[i])
    i+= 1
    
print(iris_sepal_length_dict)
print(iris_sepal_width_dict)
print(iris_petal_length_dict)
print(iris_petal_width_dict)

fig = plt.figure()
plt.title('Petal Width vs. Sepal Width')

plt.scatter(iris_petal_width_dict[1], iris_sepal_width_dict[1], label="Iris-setosa")
plt.scatter(iris_petal_width_dict[2], iris_sepal_width_dict[2], label="Iris-versicolor")
plt.scatter(iris_petal_width_dict[3], iris_sepal_width_dict[3], label="Iris-virginica")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

fig, axs = plt.subplots(4, 4, figsize=(15, 15))
plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=2.0)

axs[0, 0].set_title('Sepal Length')

axs[0, 1].set_title('Sepal Length vs. Sepal Width')
axs[0, 1].scatter(iris_sepal_length_dict[1], iris_sepal_width_dict[1])
axs[0, 1].scatter(iris_sepal_length_dict[2], iris_sepal_width_dict[2])
axs[0, 1].scatter(iris_sepal_length_dict[3], iris_sepal_width_dict[3])

axs[0, 2].set_title('Sepal Length vs. Petal Length')
axs[0, 2].scatter(iris_sepal_length_dict[1], iris_petal_length_dict[1])
axs[0, 2].scatter(iris_sepal_length_dict[2], iris_petal_length_dict[2])
axs[0, 2].scatter(iris_sepal_length_dict[3], iris_petal_length_dict[3])

axs[0, 3].set_title('Sepal Length vs. Petal Width')
axs[0, 3].scatter(iris_sepal_length_dict[1], iris_petal_width_dict[1])
axs[0, 3].scatter(iris_sepal_length_dict[2], iris_petal_width_dict[2])
axs[0, 3].scatter(iris_sepal_length_dict[3], iris_petal_width_dict[3])

axs[1, 0].set_title('Sepal Width vs. Sepal Length')
axs[1, 0].scatter(iris_sepal_width_dict[1], iris_sepal_length_dict[1])
axs[1, 0].scatter(iris_sepal_width_dict[2], iris_sepal_length_dict[2])
axs[1, 0].scatter(iris_sepal_width_dict[3], iris_sepal_length_dict[3])

axs[1, 1].set_title('Sepal Width')

axs[1, 2].set_title('Sepal Width vs. Petal Length')
axs[1, 2].scatter(iris_sepal_width_dict[1], iris_petal_length_dict[1])
axs[1, 2].scatter(iris_sepal_width_dict[2], iris_petal_length_dict[2])
axs[1, 2].scatter(iris_sepal_width_dict[3], iris_petal_length_dict[3])

axs[1, 3].set_title('Sepal Width vs. Petal Width', fontsize=14)
axs[1, 3].scatter(iris_sepal_width_dict[1], iris_petal_width_dict[1])
axs[1, 3].scatter(iris_sepal_width_dict[2], iris_petal_width_dict[2])
axs[1, 3].scatter(iris_sepal_width_dict[3], iris_petal_width_dict[3])

axs[2, 0].set_title('Petal Length vs. Sepal Length')
axs[2, 0].scatter(iris_petal_length_dict[1], iris_sepal_length_dict[1])
axs[2, 0].scatter(iris_petal_length_dict[2], iris_sepal_length_dict[2])
axs[2, 0].scatter(iris_petal_length_dict[3], iris_sepal_length_dict[3])

axs[2, 1].set_title('Petal Length vs. Sepal Width')
axs[2, 1].scatter(iris_petal_length_dict[1], iris_sepal_width_dict[1])
axs[2, 1].scatter(iris_petal_length_dict[2], iris_sepal_width_dict[2])
axs[2, 1].scatter(iris_petal_length_dict[3], iris_sepal_width_dict[3])

axs[2, 2].set_title('Petal Length')

axs[2, 3].set_title('Petal Length vs. Petal Width')
axs[2, 3].scatter(iris_petal_length_dict[1], iris_petal_width_dict[1])
axs[2, 3].scatter(iris_petal_length_dict[2], iris_petal_width_dict[2])
axs[2, 3].scatter(iris_petal_length_dict[3], iris_petal_width_dict[3])

axs[3, 0].set_title('Petal Width vs. Sepal Length')
axs[3, 0].scatter(iris_petal_width_dict[1], iris_sepal_length_dict[1])
axs[3, 0].scatter(iris_petal_width_dict[2], iris_sepal_length_dict[2])
axs[3, 0].scatter(iris_petal_width_dict[3], iris_sepal_length_dict[3])

axs[3, 1].set_title('Petal Width vs. Sepal Width', fontsize=14)
axs[3, 1].scatter(iris_petal_width_dict[1], iris_sepal_width_dict[1])
axs[3, 1].scatter(iris_petal_width_dict[2], iris_sepal_width_dict[2])
axs[3, 1].scatter(iris_petal_width_dict[3], iris_sepal_width_dict[3])

axs[3, 2].set_title('Petal Width vs. Petal Length')
axs[3, 2].scatter(iris_petal_width_dict[1], iris_petal_length_dict[1])
axs[3, 2].scatter(iris_petal_width_dict[2], iris_petal_length_dict[2])
axs[3, 2].scatter(iris_petal_width_dict[3], iris_petal_length_dict[3])

axs[3, 3].set_title('Petal Width')

plt.show()

features = []
features.append(iris_sepal_length_dict)
features.append(iris_sepal_width_dict)
features.append(iris_petal_length_dict)
features.append(iris_petal_width_dict)

fig, axs = plt.subplots(4, 4, figsize=(15, 15))
plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=2.0)

for x in range(0, 4):
    for y in range(0, 4):
        if x != y:
            axs[x, y].scatter(features[x][1], features[y][1])
            axs[x, y].scatter(features[x][2], features[y][2])
            axs[x, y].scatter(features[x][3], features[y][3])
        
plt.show()



