#
# talk to Arduino!
#

import serial

#port = serial.Serial('COM3')                        # open the serial port (WIN)
port = serial.Serial('/dev/cu.usbserial-DN040GLI')   # open the serial port (MAC/Linux)

print("Waiting for conversation...")

s = port.readline().strip()             # read a line of input from the Arduino

print ("got ",str(s))                   # echo to screen

while s != b'Hello':                    # check for "Hello"
    s = port.readline().strip()         # not yet, try again
    print ("got ",str(s))

port.write(b'g')                        # send a command

s = port.readline().strip()             # get a reply

indexes = []
values = []

while s[:3] != b'End':                  # is it the end?
    print(str(s))                       # nope, echo, repeat
    data = s.decode()
    if ',' in data:
        index,value = data.split(',')
        indexes.append(index)
        values.append(value)
    s = port.readline().strip()
    
print ("Finally got:", str(s))          # that's it!
print ("Finished!")

port.close()                            # shut it down

print(indexes)
print(values)

#
# write values to file
#

f=open('data.csv','w')
for v in values:
    f.write(v)
    
f.close()



