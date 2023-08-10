cat = True
dog = False

print(type(cat))

from cities import cities

print(cities)
first_alb = cities[0] == 'Albuquerque'
second_alb = cities[1] == 'Albuquerque'
first_last = cities[0] == cities[-1]

print(first_alb, second_alb, first_last)

crime_rates = [749, 371, 828, 503, 1379, 425, 408, 542, 1405, 835, 1288, 647, 974, 1383, 455, 658, 675, 615, 2122, 423, 362, 587, 543, 563, 168, 992, 1185, 617, 734, 1263, 784, 352, 397, 575, 481, 598, 1750, 399, 1172, 1294, 992, 522, 1216, 815, 639, 1154, 1993, 919, 594, 1160, 636, 752, 130, 517, 423, 443, 738, 503, 413, 704, 363, 401, 597, 1776, 722, 1548, 616, 1171, 724, 990, 169, 1177, 742]
print(crime_rates)

first = crime_rates[0]
first_500 = first > 500
first_749 = first >= 749
first_last = first >= crime_rates[-1]

print(first_500, first_749, first_last)

second = crime_rates[1]
second_500 = second < 500
second_371 = second <= 371
second_last = second <= crime_rates[-1]

print(second_500, second_371, second_last)

result = 0

if cities[2] == u"Anchorage":
    result = 1

assert result == 1

reqults = 0

if crime_rates[0] > 500:
    if crime_rates[0] > 300:
        results = 3

five_hundred_list = []

for cr in crime_rates:
    if cr > 500:
        five_hundred_list.append(cr)

assert all([_>500 for _ in five_hundred_list])

print(crime_rates)
highest = crime_rates[0]

for cr in crime_rates:
    if cr > highest:
        highest = cr



