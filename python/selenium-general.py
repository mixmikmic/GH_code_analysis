from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup

driver = webdriver.Chrome()  # or Firefox

driver.get("http://www.google.com")

target = driver.find_element_by_name("")
target = driver.find_elements_by_tag_name("")

with open("page-source.html", "r") as f:
    f.write(str(driver.page_source))

soup = BeautifulSoup(str(driver.page_source))

chrome_profile = webdriver.ChromeOptions()
profile = {
    "download.default_directory": "",
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "plugins.plugins_disabled": ["Chrome PDF Viewer"]}
chrome_profile.add_experimental_option("prefs", profile)

# give the profile as an argument to the webdriver
driver = webdriver.Chrome(chrome_options=chrome_profile)

