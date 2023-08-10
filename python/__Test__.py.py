s = raw_input("input your age")
if s == "":
    raise Exception("Input must not be empty")
try:
    i = int(s)
except ValueError:
    print "Could not convert data to an integer."
except:
    print "Unknown exception!"
else:
    print "you are %d years old" %i
finally:
    print "Goodbye!"



