import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime

#getting my timeline html file that I recieved from Facebook
with open("timeline.htm") as f:
    req = f.read()

soup = BeautifulSoup(req, "html.parser")

dates = soup.find_all("div", {"class": "meta"})

#scratch workbook to understand how to get the attributes that I need out of the data (you can ignore this cell)
dates[2].text.split(",")[2].strip()
month = dates[2].text.split(",")[1].strip().split(" ")[0]
day = dates[2].text.split(",")[1].strip().split(" ")[1]
month, day
time = dates[2].text.split(",")[2].strip().split(" ")[2]
time
if "pm" in time:
    time = time.rstrip("pm")
    arr = time.split(":")
    temp = int(arr[0])
    temp += 12
    arr[0] = str(temp)
    time = ":".join(arr)
time, temp, arr
dates[2]
dates[2].find_next("div")

timeline = pd.DataFrame(columns=["Year", "Month", "Day", "Time", "Text", "Hour", "Minutes"])
timeline

Years = []
Months = []
Days = []
Texts = []
Times = []
Hours = []
Minutes = []
for comment in soup.find_all("div", {"class": "comment"}):
    date = comment.find_previous("div", {"class": "meta"})
    month = date.text.split(",")[1].strip().split(" ")[0]
    day = int(date.text.split(",")[1].strip().split(" ")[1])
    year = int(date.text.split(",")[2].strip().split(" ")[0])
    time = date.text.split(",")[2].strip().split(" ")[2]
    if "pm" in time:
        time = time.rstrip("pm")
        arr = time.split(":")
        hour = int(arr[0])
        minute = int(arr[1])
        temp = int(arr[0])
        temp += 12
        arr[0] = str(temp)
        time = ":".join(arr)
    elif "am" in time:
        time = time.rstrip("am")
        arr = time.split(":")
        hour = arr[0]
        minute = arr[1]
    if(day != "October 13"):
        Years.append(year)
        Months.append(month)
        Days.append(day)
        Texts.append(comment.text)
        Times.append(time)
        Hours.append(hour)
        Minutes.append(minute)

timeline["Year"] = Years
timeline["Month"] = Months
timeline["Day"] = Days
timeline["Text"] = Texts
timeline["Time"] = Times
timeline["Hour"] = Hours
timeline["Minutes"] = Minutes
timeline.head()

with open("messages.htm") as f:
    req = f.read()

soup = BeautifulSoup(req, "html.parser")

dates = soup.find_all("span", {"class": "meta"})

messages = pd.DataFrame(columns=["Year", "Month", "Day", "Time", "Text"])
messages

Years = []
Months = []
Days = []
Texts = []
Times = []
Hours = []
Minutes = []
for comment in soup.find_all("p"):
    date = comment.find_previous("span", {"class": "meta"})
    month = date.text.split(",")[1].strip().split(" ")[0]
    day = int(date.text.split(",")[1].strip().split(" ")[1])
    year = int(date.text.split(",")[2].strip().split(" ")[0])
    time = date.text.split(",")[2].strip().split(" ")[2]
    if "pm" in time:
        time = time.rstrip("pm")
        arr = time.split(":")
        hour = int(arr[0])
        minute = int(arr[1])
        temp = int(arr[0])
        temp += 12
        arr[0] = str(temp)
        time = ":".join(arr)
    elif "am" in time:
        time = time.rstrip("am")
        arr = time.split(":")
        hour = int(arr[0])
        minute = int(arr[1])
    if(day != "October 13"):
        Years.append(year)
        Months.append(month)
        Days.append(day)
        Texts.append(comment.text)
        Times.append(time)
        Hours.append(hour)
        Minutes.append(minute)

messages["Year"] = Years
messages["Month"] = Months
messages["Day"] = Days
messages["Text"] = Texts
messages["Time"] = Times
messages["Hour"] = Hours
messages["Minutes"] = Minutes
messages.head()

import pickle
pickle.dump(messages, open("messages.pkl", "wb"))
pickle.dump(timeline, open("timeline.pkl", "wb"))



