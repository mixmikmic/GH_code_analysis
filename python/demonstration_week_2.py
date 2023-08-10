import re

test_string = "Incident American Airlines Flight 11 involving a Boeing 767-223ER in 2001"

mobject = re.search(r"Incident (.*) involving", test_string)

mobject.group(0)

mobject.group(1)

mobject1 = re.findall(r"Incident (.*) involving", test_string)

mobject1[0]

string1 = '\section{Test the Python regular expressions}'
string1

string2 = "\\\\section"
string2

print string2

x = re.match(string2, string1)
x.group(0)

print x.group(0)

string3 = r"\\section"
string3

re.match(string3, string1)

def year(pattern, m):
    if re.match(pattern, m):
         print m + " is a year"
    else :
        print m + " is NOT a year"

year(r"^\d{4}$", "2016")

year(r"^\d{4}$", "1998")

year(r"^\d{2}$", "16")

year(r"^\d{2}$", "98")

year(r"^(?:\d{2})?\d{2}$", "16")

obj = re.match(r"^(?:\d{2})?\d{2}$", "2016")
print obj.groups()
print obj.group(0)

obj = re.match(r"^(\d{2})?\d{2}$", "2016")
print obj.groups()
print obj.group(0)
print obj.group(1)

def month(pattern, m):
    if re.match(pattern, m):
         print m + " is a month"
    else :
        print m + " is NOT a month"

month(r"^\d{2}$", "9")

month(r"^\d{2}$", "12")

month(r"^\d?\d$", "9")

month(r"^\d?\d$", "13")

month(r"^\d?\d$", "00")

month(r"^(0?[1-9]|1[0-2])$", "10")

month(r"^(0?[1-9]|1[0-2])$", "00") ## try 00

def day(pattern, m):
    if re.match(pattern, m):
         print m + " is a day"
    else :
        print m + " is NOT a day"

day(r"^(\d?\d)$", "1")

day(r"^(\d?\d)$", "21")

day(r"^(\d?\d)$", "32")

day(r"(0[1-9]|[12][0-9]|3[01])", "31")

day(r"(0[1-9]|[12][0-9]|3[01])", "00")

day(r"(0[1-9]|[12][0-9]|3[01])", "33")

def date(pattern, m):
    if re.match(pattern, m):
         print m + " is a date"
    else :
        print m + " is NOT a date"

date(r"(0[1-9]|[12][0-9]|3[01])[/-](0?[1-9]|1[0-2])[/-]((?:\d{2})?\d{2})", "19-10-2019")

date(r"(0[1-9]|[12][0-9]|3[01])[/-](0?[1-9]|1[0-2])[/-]((?:\d{2})?\d{2})", "19/10/2019")

date(r"(0[1-9]|[12][0-9]|3[01])[/-](0?[1-9]|1[0-2])[/-]((?:\d{2})?\d{2})", "19/13/2019")

def isCreditCard(pattern, string):
    if re.match(pattern, string):
        print string + " is a credit card nunmber!"
    else:
        print string + " is NOT a credit card number!"

isCreditCard(r"^4\d{12}$", "4123456789012")

isCreditCard(r"^4\d{12}$", "4123456789012345")

isCreditCard(r"^4\d{15}$", "4123456789012")

isCreditCard(r"^4\d{15}$", "4123456789012345")

isCreditCard(r"^4\d{12}(?:\d{3})?$", "4123456789012")

isCreditCard(r"^4\d{12}(?:\d{3})?$", "4123456789012345")

isCreditCard(r"^5[1-5]\d{14}$", "5123456789012345")

isCreditCard(r"^5[1-5]\d{14}$", "5723456789012345")

isCreditCard(r"3[47]\d{13}", "341234567890123")

isCreditCard(r"3[47]\d{13}", "371234567890123")

isCreditCard(r"3[47]\d{13}", "381234567890123")

cardPattern = r'''(?x)
        4\d{12}(?:\d{3})? | # Visa
        5[1-5]\d{14} |      # Master
        3[47]\d{13}         # American Express 
        '''

isCreditCard(cardPattern, "31234567890123")



