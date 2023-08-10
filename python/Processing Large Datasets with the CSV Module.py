import csv
import urllib2

def pr_min_max(ip_addr):
    mintemp = {'Value': 1000.0}
    maxtemp = {'Value': 0.0}

    cr = csv.DictReader(urllib2.urlopen("http://%s:7645/data.csv" % ip_addr))

    for row in cr:
        temp = float(row['Value'])
        var = row['Variable']
        if var == 'tasmax' and temp > float(maxtemp['Value']):
            maxtemp = row
        if var == 'tasmin' and temp < float(mintemp['Value']):
            mintemp = row

    print "The minimum temperature is %.2f degrees C on %s at (%.3fW, %.3fN)" %         (float(mintemp['Value'])-273.15, mintemp['Date'][:7],         -float(mintemp['Longitude']), float(mintemp['Latitude']))
    print "The maximum temperature is %.2f degrees C on %s at (%.3fW, %.3fN)" %         (float(maxtemp['Value'])-273.15, maxtemp['Date'][:7],         -float(maxtemp['Longitude']), float(maxtemp['Latitude']))

# Note: replace with the IP address of your data server
pr_min_max("192.168.99.100")

