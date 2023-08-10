from classes.ScraperData import ScraperData
from classes.Scraper import Scraper
from classes.MysqlStorage import MysqlStorage
from classes.CsvStorage import CsvStorage
from classes.UsfsWebScraper import UsfsWebScraper
from classes.UsfsWebScraperLocal import UsfsWebScraperLocal
import pandas as pd
import numpy as np

webscraper = UsfsWebScraper()
mysql_store = MysqlStorage()

df_urls = pd.read_csv('data/usfs_sites.csv')
df_urls.head()

usfs_data = ScraperData('usfs',df_urls,mysql_store,webscraper)

webscraper = UsfsWebScraperLocal()
usfs_data = ScraperData('usfs',df_urls,mysql_store,webscraper)

usfs_data.extract()

usfs_data.df







