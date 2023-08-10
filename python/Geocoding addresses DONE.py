from geopy.geocoders import GoogleV3
import csv
import time

geolocator = GoogleV3()

address_file = open('payday_lenders.csv', 'rb')
input_file = csv.DictReader(address_file)

geocoded_file = open('payday_geocoded.csv', 'wb')

output_fields = ['NAME', 'DBA', 'STADDR', 'STADDR2', 'CITY', 'STATE', 'ZIP', 'MATCH_ADDR', 'LAT_Y', 'LONG_X']
output = csv.DictWriter(geocoded_file, output_fields)
output.writeheader()

for row in input_file:
    
    if input_file.line_num <= 6:

        addr = (row['STADDR']+' '+row['STADDR2']).strip()+', '+row['CITY']+', '+row['STATE']+' '+row['ZIP']

        location = geolocator.geocode(addr)
        
        row['LAT_Y'] = location.latitude
        row['LONG_X'] = location.longitude
        row['MATCH_ADDR'] = location.address

        output.writerow(row)

        print('Attempted geocode of {0}, row {1}.'.format(addr, input_file.line_num))

        time.sleep(2)
    else:
        print('All done!')
        break

address_file.close()
geocoded_file.close()

