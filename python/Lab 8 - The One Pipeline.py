from classes.Data import Data
import pandas as pd 

class Pipeline(Data):
    
    def __init__(self,name, data_list,merge_type,store):
        self.data_list = data_list
        self.name = name
        self.merge_type = merge_type
        self.store = store
        self.df = pd.DataFrame()
        
    def extract(self):
        list(map(lambda x:x.extract(),self.data_list))
        merge = self.merge_type(self.name,self.data_list,self.store)
        
        merge.extract()
        merge.put()
        self.df = merge.df
    
    def put(self):
        if (self.df.empty) :
            print("Pipeline.put(): dataframe is empty, run get() or extract()")
            return
        self.storage.put(self.df, self.name)
    
    def get(self):
        self.df = self.storage.get(self.name)
        
        
        

from classes.DistanceMergeData import DistanceMergeData
from classes.ScraperData import ScraperData
from classes.RidbData import RidbData
from classes.RidbDataLive import RidbDataLive

# import the scrapers
from classes.ReservationsScraper import ReservationsScraper
from classes.UsfsWebScraper import UsfsWebScraper

# import the storage
from classes.MysqlStorage import MysqlStorage
from classes.CsvStorage import CsvStorage
from classes.Pipeline import Pipeline

# import visualization
from classes import BokehPlot
from bokeh.io import output_notebook
from bokeh.plotting import show

import config
import pandas as pd

mysql_store = MysqlStorage()
csv_store = CsvStorage()
destination_info = dict(Latitude=45.4977712, Longitude=-121.8211673, radius=15)
start_date = '06/01/2016'
stay_length = 2
usfs_urls = pd.read_csv('data/usfs_sites.csv')
reservation_urls = pd.read_csv('data/reservation_urls.csv')
reservation_urls = reservation_urls.assign(start_date=start_date, stay_length=stay_length)

ridb_data = RidbData('ridb_merge_lab', "camping", destination_info, mysql_store)
ridb_data_live = RidbDataLive('ridb_live_merge_lab', "camping", destination_info, mysql_store)
usfs_data = ScraperData('usfs_merge_lab',usfs_urls,mysql_store,UsfsWebScraper())
res_data = ScraperData('res_merge_lab',reservation_urls,mysql_store,ReservationsScraper())

one_data_list = [res_data, usfs_data,ridb_data]
# live version
#one_data_list = [res_data, usfs_data,ridb_data_live]

pipe = Pipeline('the_one_pipelne', one_data_list, DistanceMergeData,mysql_store)

pipe.extract()

pipe.df.columns

output_notebook()

cols_to_display=['FacilityName','SitesAvailable', 'Water','Restroom']

sites_with_availability = pipe.df.dropna(subset=['SitesAvailable'])



pipe.df[['FacilityName', 'Reservations', 'Restroom','SitesAvailable']]

p = BokehPlot.create_plot(sites_with_availability, cols_to_display)

show(p)



