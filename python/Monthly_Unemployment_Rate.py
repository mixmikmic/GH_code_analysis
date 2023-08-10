from bs4 import BeautifulSoup
from urllib import request
import numpy as np

monthly_unemployment_data = request.urlopen('http://www.bls.gov/web/empsit/cpseea03.htm')
lines_list = monthly_unemployment_data.readlines()

index = 0
for line in lines_list:
    index = index + 1
    if "cps_eande_m03.r.2 cps_eande_m03.r.2.1 cps_eande_m03.r.2.1.3" in line.decode("utf-8"):
        break

start_index = index
print("start index:", start_index)
        
for line in lines_list[index:]:
    index = index + 1
    if "</tr>" in line.decode("utf-8"):
        break

end_index = index
print("End index:", end_index)

data_string = ""
for data in lines_list[start_index:end_index-1]:
    data_string = data_string + data.decode("utf-8")

print("data string:")    
print(data_string)

soup = BeautifulSoup(data_string)
data = soup.find_all(class_="datavalue")

unemploy_rate = [float(rate.text) for rate in data]
print(unemploy_rate)

bar(np.arange(len(unemploy_rate)),unemploy_rate)
show()

