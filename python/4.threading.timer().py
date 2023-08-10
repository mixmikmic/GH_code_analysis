import time,threading,sys                                                   # 1.import the library
def timer():                                                                    
    import datetime                                                         # 2.set functions (datetime.delta)
    t1 = datetime.datetime.strptime('1,1,2017 00:00','%d,%m,%Y %H:%M')      ## set the time
    t2 = datetime.datetime.now()                                            ## get the current time
    t=t1-t2                                                                 ## get the datetime.delta
    print "\rCountdown to New Year 2017 :",t,                               ## print
    sys.stdout.flush()                                                      ## clear the screen
    threading.Timer(1,timer).start()                                        # 3. print the time at a regular time interval (1 second)
timer()

exit()



