# import the data types
from classes.ScraperData import ScraperData
from classes.RidbData import RidbData
from classes.RidbDataLive import RidbDataLive

# import the scrapers
from classes.ReservationsScraper import ReservationsScraper
from classes.UsfsWebScraper import UsfsWebScraper
from classes.UsfsWebScraperLocal import UsfsWebScraperLocal

# import the storage
from classes.MysqlStorage import MysqlStorage
from classes.CsvStorage import CsvStorage

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

ridb_data.extract()

ridb_data.df

one_data_list = [res_data, usfs_data,ridb_data]
# live version
# one_data_list = [res_data, usfs_data,ridb_hiking_live, ridb_data_live]

list(map(lambda x:x.extract(),one_data_list))

list(map(lambda x:x.df.shape,one_data_list))

list(map(lambda x:x.name,one_data_list))

list(map(lambda x:x.df.columns,one_data_list))

from classes.DistanceMergeData import DistanceMergeData

merge_data = DistanceMergeData("merge_res_usfs",one_data_list,mysql_store)

merge_data.extract()

merge_data.df.columns

merge_data.put()

merge_data.name



