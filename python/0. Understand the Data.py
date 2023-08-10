# first, read in the data

import os
import csv

os.chdir('../data/')

records = []

with open('call_records.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        records.append(row)

print(records[0]) # print the header
records = records[1:] # remove the header
print(records[0]) # print an example record

import matplotlib.pyplot as plt # import our graphing module
plt.rcParams["figure.figsize"] = (15,5) # set the default figure size

# get the list of times that calls occurred, modulo 24 so we get the hour of day
times = [int(record[0]) % 24 for record in records]

# the histogram of the data
plt.hist(times, bins=24)

plt.xlabel('Hour of Day')
plt.ylabel('Number of Calls')
plt.title('Calls per Hour')

plt.show()

# same thing, but not modulo day so we get per-day volume
import math
times = [int(record[0]) for record in records]
nbins = math.ceil(max(times)/24)
# the histogram of the data
plt.hist(times, bins=nbins) # one bin per day

plt.xlabel('Day')
plt.ylabel('Number of Calls')
plt.title('Calls per Day')
plt.xticks(range(12, max(times), 24), range(nbins+1))


plt.show()

number = '974-703-1399'
cells = [int(r[1]) for r in records if r[2] == number]

uniq_cells = sorted(list(set(cells)))
freq = [cells.count(x) for x in uniq_cells]

plt.bar(uniq_cells, freq)

plt.xlabel('Cell')
plt.ylabel('Number of Calls')
plt.title('Location Pattern (frequency) for {}'.format(number))

plt.xticks(range(0, max(uniq_cells), 50))

plt.show()

time_location = [[int(r[0]), int(r[1])] for r in records if r[2] == number]

time, location = ([a for a,b in time_location], [b for a,b in time_location])

plt.scatter(time, location, alpha=0.5)
plt.title('Location Pattern (time) for {}'.format(number))
plt.ylabel('call location')
plt.xlabel('time')
plt.show()

time_location = [[int(r[0]), int(r[1])] for r in records if r[2] == number]

time, location = ([int(a) % 24 for a,b in time_location], [b for a,b in time_location])

plt.scatter(time, location, alpha=0.2)
plt.title('Location Pattern (time of day) for {}'.format(number))
plt.ylabel('call location')
plt.xlabel('time')
plt.xticks(range(24))
plt.show()

import matplotlib.cm as cm

time_location_to = [[int(r[0]), int(r[1]), (r[3])] for r in records if r[2] == number]

# the recipients of the calls
to = [c for a,b,c in time_location_to]

colors = cm.tab20(range(0, len(set(to))))

# this will only generate a color mapping for 20 numbers - the rest will be left out
colormap = {}
for recipient,color in zip(to, colors):
    colormap[recipient] = color[0:3]

for recipient,color in colormap.items():
    data = list(filter(lambda x: x[2] == recipient, time_location_to))
    time, location = ([a for a,b,c in data], [b for a,b,c in data])
    plt.scatter(time, location, c=color)
    
plt.title('Location Pattern (time) for {}'.format(number))
plt.ylabel('call location')
plt.xlabel('time')
plt.show()

recipients = [r[3] for r in records if r[2] == number]

uniq_recipients = sorted(list(set(recipients)))
freq = [recipients.count(x) for x in uniq_recipients]

plt.bar(range(len(uniq_recipients)), freq)

plt.xlabel('Recipient')
plt.ylabel('Number of Calls')
plt.title('Call Recipient Pattern (frequency) for {}'.format(number))

plt.xticks(range(0, len(uniq_recipients)), uniq_recipients, rotation=-70)

plt.show()

