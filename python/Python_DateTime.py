import datetime

now = datetime.datetime.now()
print(now)
print("Now :",str(now))
print("ctime : ",str(now.ctime()))  #ctime with more detail info
print("UTC NOW : ",datetime.datetime.utcnow())
t = datetime.time(now.hour,now.minute,now.second,now.microsecond)
print("Time : ",t , " >> ",str(t))

today = datetime.datetime.today()
print("Today : ",today)
print("Today Str: ",str(today)) #str casting
print("Today DateTime : " ,datetime.date.today())
print("Today DateTime Str : " ,str(datetime.date.today()))
print('Ordinal :', today.toordinal())

print("Date :",str(today.date()))
print("Time : ",str(today.time()))

print("Year : ",today.year)
print("Month : ",today.month)
print("Day : ",today.day)
print("Hour : ",today.hour)
print("Minute : ",today.minute)
print("Second : ",today.second)
print("WeekDay :" ,today.weekday())
print("Iso Calendar :",today.isocalendar())
print("Iso Format :",today.isoformat())

print(today.timetuple())

tt = today.timetuple()
print('tuple  : tm_year  =', tt.tm_year)
print('         tm_mon   =', tt.tm_mon)
print('         tm_mday  =', tt.tm_mday)
print('         tm_hour  =', tt.tm_hour)
print('         tm_min   =', tt.tm_min)
print('         tm_sec   =', tt.tm_sec)
print('         tm_wday  =', tt.tm_wday)
print('         tm_yday  =', tt.tm_yday)
print('         tm_isdst =', tt.tm_isdst)

'''
The method toordinal returns the proleptic Gregorian ordinal. 
The proleptic Gregorian calendar is produced by extending the Gregorian calendar backward
to dates preceding its official introduction in 1582. January 1 of year 1 is day 1
'''
print("Ordinal : ",today.toordinal())
print("From Ordinal : ",today.fromordinal(today.toordinal()))
print("Time :", today.time())
today.timetz() ," and ", today.tzname()

print('Today    :', today)
yesterday = today - datetime.timedelta(days=1)
print('Yesterday:', yesterday)
tomorrow = today + datetime.timedelta(days=1)
print('Tomorrow :', tomorrow)

print('Tomorrow - Yesterday :', tomorrow - yesterday)
print('Yesterday - Tomorrow :', yesterday - tomorrow)
print('Tomorrow > Yesterday :', tomorrow > yesterday)
print('Yesterday > Tomorrow:', yesterday > tomorrow)

print("100 Days Ago : ", str(today - datetime.timedelta(days=100)))
print("4 Days Ago : ", str(today - datetime.timedelta(days=4)))
print("7 Days Ago : ", str(today - datetime.timedelta(days=7)))
print("1 Week Ago : ", str(today - datetime.timedelta(weeks=1)))
print("4 Hour Ago : ", str(today - datetime.timedelta(hours=4)))
print("30 Minutes Ago : ", str(today - datetime.timedelta(minutes=30)))
print("45 Seconds Ago : ", str(today - datetime.timedelta(seconds=45)))
print('-'*50)
print("1 Week After : ", str(today + datetime.timedelta(weeks=1)))
print("100 Days After : ", str(today + datetime.timedelta(days=100)))
print("4 Hour After : " ,str(today + datetime.timedelta(hours=4)))
print("30 Minutes After :" , str(today + datetime.timedelta(minutes=30)))
print("45 Seconds After :", str(today + datetime.timedelta(seconds=45)))



