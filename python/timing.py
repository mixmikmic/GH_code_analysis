import time
import timeit

currentTime = time.gmtime()
print(currentTime)

time.strftime("%a, %d %b %Y %H:%M:%S +0000", currentTime)

type(currentTime)

currentTime.tm_hour

def parseTime(timeObj):
    '''parseTime:
    takes time.struct_time instances
    :return  time displayed as string -  year month day hour min sec'''
    return (str(timeObj.tm_year) + str(timeObj.tm_mon) + str(timeObj.tm_mday) +
            str(timeObj.tm_hour) + str(timeObj.tm_min) + str(timeObj.tm_sec))

parseTime(currentTime)

