from classes.ScraperData import ScraperData
from classes.Scraper import Scraper
from classes.MysqlStorage import MysqlStorage
from classes.CsvStorage import CsvStorage
from classes.ReservationsScraper import ReservationsScraper
import pandas as pd
import numpy as np

res_scraper = ReservationsScraper()
mysql_store = MysqlStorage()

df_res = pd.read_csv('data/reservation_urls.csv')
df_res.head()

start_date = '06/01/2016'
stay_length = 2

df_res = df_res.assign(start_date = start_date, stay_length=stay_length)

df_res.head()

res_data = ScraperData('reservations',df_res,mysql_store,res_scraper)

res_data.extract()

res_data.df

res_data.put()





