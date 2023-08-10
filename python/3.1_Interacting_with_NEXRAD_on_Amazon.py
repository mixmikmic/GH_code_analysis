#lets start with some imports

#Py-ART, simply the best sowftware around.. Give those guys a grant
import pyart

#Boto3 is the AWS SDK
import boto3

#botocore contains core configuration utilities for boto2 and boto3
from botocore.handlers import disable_signing

#Tempory files in Python.. A very useful module
import tempfile

#datetime modules.. very handy!
from datetime import datetime

#timezone info
import pytz

#plotting
from matplotlib import pyplot as plt

#plotting on a maop
import cartopy
get_ipython().run_line_magic('matplotlib', 'inline')

# So we start with bucket neame
bucket = "noaa-nexrad-level2"
# Create a s3 "client"
s3 = boto3.resource('s3')
# Set it to unsigned 
s3.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)

#So now we connect to the bucket with the radar data
aws_radar = s3.Bucket(bucket)

for obj in aws_radar.objects.filter(Prefix='2011/05/20/KVNX/'):
    print('{0}:{1}'.format(aws_radar.name, obj.key))

my_list_of_keys = [this_object.key for this_object in aws_radar.objects.filter(Prefix='2011/05/20/KVNX/')]
print(my_list_of_keys[1])

#lets test it one one of the keys
my_datetime = datetime.strptime(my_list_of_keys[1][20:35], '%Y%m%d_%H%M%S')
print(my_datetime)

#now make an empty list to populated with the datetimes matching
my_list_of_datetimes = []

#loop over all the contents of the bucket with the prefix
for obj in aws_radar.objects.filter(Prefix='2011/05/20/KVNX/'):
    try:
        my_list_of_datetimes.append(datetime.strptime(obj.key[20:35], '%Y%m%d_%H%M%S'))
    except ValueError:
        pass #usually a tar file left in the bucket

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

#lets see if we can find the key for 11Z on the 5th of may 2011
desired_time = datetime(2011,5,20,11,0)

#find the nearest datetime
my_nearest = nearest(my_list_of_datetimes, desired_time)

print('nearest: ', my_nearest, ' desired: ', desired_time)

#find the index of the nearest
print('index: ', my_list_of_datetimes.index(my_nearest))

#find the key of the nearest datetime
print('key: ', my_list_of_keys[my_list_of_datetimes.index(my_nearest)])

def find_my_key(radar_name, desired_datetime):
    """
    Find the key in Amazon s3 corresponding to a particular radar site and 
    datetime
    
    Parameters
    ----------
    radar_name : str
        Four letter radar name
    desired_datetime : datetime
        The date time desired
    Returns
    -------
    my_key : string
        string matching the key for the radar file on AWS s3
    """
    
    bucket = "noaa-nexrad-level2"
    # Create a s3 "client"
    s3 = boto3.resource('s3')
    # Set it to unsigned 
    s3.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
    target_string = datetime.strftime(desired_datetime, '%Y/%m/%d/'+radar_name)
    my_list_of_keys = [this_object.key for this_object in aws_radar.objects.filter(Prefix=target_string)]
    my_list_of_datetimes = []
    for obj in aws_radar.objects.filter(Prefix=target_string):
        try:
            my_list_of_datetimes.append(datetime.strptime(obj.key[20:35], '%Y%m%d_%H%M%S'))
        except ValueError:
            pass #usually a tar file left in the bucket
    my_nearest = nearest(my_list_of_datetimes, desired_datetime)
    my_key = my_list_of_keys[my_list_of_datetimes.index(my_nearest)]
    return my_key

print(find_my_key('KLOT', datetime(2011,1,1,20,15)))
print(find_my_key('KILX', datetime.utcnow()))
print(find_my_key('TJUA', datetime(2017,9,20,9,0)))

#Lets look at Hurricane Maria
maria_datetime = datetime(2017,9,20,9,0)

#grab the key for Maria
my_key = find_my_key('TJUA', maria_datetime)
print(my_key)

#create a temporary named file
localfile = tempfile.NamedTemporaryFile()

#fetch the data from AWS S3
aws_radar.download_file(my_key, localfile.name)

#read that file into Py-ART!
radar = pyart.io.read(localfile.name)

#Sweep we want to plot
sweep = 0

#Get the date at the start of collection
index_at_start = radar.sweep_start_ray_index['data'][sweep]
time_at_start_of_radar = pyart.io.cfradial.netCDF4.num2date(radar.time['data'][index_at_start], 
                                  radar.time['units'])

#make a nice time stamp
pacific = pytz.timezone('US/Eastern')
local_time = pacific.fromutc(time_at_start_of_radar)
fancy_date_string = local_time.strftime('%A %B %d at %I:%M %p %Z')
print(fancy_date_string)

#Set up our figure
fig = plt.figure(figsize = [10,8])

#create a Cartopy Py-ART display object
display = pyart.graph.RadarMapDisplayCartopy(radar)

#get center of the display for the projection
lat_0 = display.loc[0]
lon_0 = display.loc[1]

# Main difference from Basemap! 
#Cartopy forces you to select a projection first!
projection = cartopy.crs.Mercator(
                central_longitude=lon_0,
                min_latitude=15, max_latitude=20)

title = 'TJUA \n' + fancy_date_string

#plot a PPI! add coastline at 10m resolution
display.plot_ppi_map(
    'reflectivity', sweep, colorbar_flag=True,
    title=title,
    projection=projection,
    min_lon=-67, max_lon=-65, min_lat=17, max_lat=19,
    vmin=-12, vmax=64, resolution='10m', 
    cmap=pyart.graph.cm.LangRainbow12)

# Mark the radar
display.plot_point(lon_0, lat_0, label_text='TJUA')

# Plot some lat and lon lines
gl = display.ax.gridlines(draw_labels=True,
                          linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False

#Now it is your turn
#Grab and plot the data for Hurricane Sandy from KDOX at 2012-10-29-20:00Z 
#set a bounding box min_lon=-78, max_lon=-73.5, min_lat=37, max_lat=40

#Your answer

#run to get our answer!
get_ipython().run_line_magic('load', 'sandy_answer.py')



