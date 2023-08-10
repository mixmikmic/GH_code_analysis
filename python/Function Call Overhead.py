import datetime

x = 0
def inner(i):
    global x
    x = x + i
    

multiplier = 10000000
start_time = datetime.datetime.now()
print('=== Start time '+ str(start_time)+ ' ========== '); print()
for i in range(multiplier): 
    inner(i)
print("Done")

# Calculate and print run time
end_time = datetime.datetime.now()
run_time_multiple = end_time-start_time
print(); print('=== End ========== '+ str(end_time) + ' ========== ' + " run time: "+ str(run_time_multiple) + ' ========== ')

x = 0
def aggregate(list):
    global x
    for i in list:
        x = x + i

start_time = datetime.datetime.now()
print('=== Start time '+ str(start_time)+ ' ========== '); print()
aggregate(range(multiplier))
# Calculate and print run time
end_time = datetime.datetime.now()
run_time_single = end_time-start_time
print(); print('=== End ========== '+ str(end_time) + ' ========== ' + " run time: "+ str(run_time_single) + ' ========== ')

print("run time Multiple: "+str(run_time_multiple))
print("run time Single: "+str(run_time_single))
print("run time Diff: "+str(run_time_multiple-run_time_single))
print("Multiplier: "+str(multiplier))
print("Overhead per function call: "+str((run_time_multiple-run_time_single)/multiplier))
print("ratio: "+str(run_time_multiple/run_time_single))



