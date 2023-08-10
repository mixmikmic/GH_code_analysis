from selenium import webdriver
from selenium.webdriver.common.keys import Keys

driver = webdriver.Chrome()
driver.get("https://www.google.com")

element = driver.find_element_by_name("q")
element.clear()
element.send_keys("Hello World!" + Keys.RETURN)



