get_ipython().system(' touch test.txt')
a = open("test.txt", "r")
print(a)
f = open("crime_rates.csv", "r")

f = open("crime_rates.csv", "r")
data = f.read()

print(data)

# We can split a string into a list.
sample = "john,plastic,joe"
split_list = sample.split(",")
print(split_list)

# Here's another example.
string_two = "How much wood\ncan a woodchuck chuck\nif a woodchuck\ncan chuck wood?"
split_string_two = string_two.split('\n')
print(split_string_two)

# Code from previous cells
f = open('crime_rates.csv', 'r')
data = f.read()
rows = data.split('\n')
print(rows[0:5])

ten_rows = rows[0:10]
for row in ten_rows:
    print(row)

three_rows = ["Albuquerque,749", "Anaheim,371", "Anchorage,828"]
final_list = []
for row in three_rows:
    split_list = row.split(',')
    final_list.append(split_list)
print(final_list)
for elem in final_list:
    print(elem)
print(final_list[0])
print(final_list[1])
print(final_list[2])

f = open('crime_rates.csv', 'r')
data = f.read()
rows = data.split('\n')
final_data = [row.split(",")
              for row in rows]
print(final_data[0:5])

five_elements = final_data[:5]
print(five_elements)
cities_list = [city for city,_ in five_elements]

crime_rates = []

for row in five_elements:
    # row is a list variable, not a string.
    crime_rate = row[1]
    # crime_rate is a string, the city name.
    crime_rates.append(crime_rate)
    
cities_list = [row[0] for row in final_data]

f = open('crime_rates.csv', 'r')
data = f.read()
rows = data.split('\n')
print(rows[0:5])

int_crime_rates = []

for row in rows:
    data = row.split(",")
    if len(data) < 2:
        continue
    int_crime_rates.append(int(row.split(",")[1]))

print(int_crime_rates)





