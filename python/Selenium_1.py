from selenium import webdriver
# change Chrome() below with Firefox(), if the latter is the driver you decided to use
browser = webdriver.Chrome()
url = 'http://inventwithpython.com'
browser.get(url)
our_element = browser.find_element_by_link_text('Read It Online')
type(our_element)
our_element.click() # follows the "Read It Online" link

browser.close()

from selenium import webdriver
browser = webdriver.Chrome()
browser.get('https://mail.yahoo.com')
email_element = browser.find_element_by_id('login-username')
email_element.send_keys('hrantdavtyan@yahoo.com')
next_button_element = browser.find_element_by_id('login-signin')
next_button_element.click()
password_element = browser.find_element_by_id('login-passwd')
password_element.send_keys('my_password')
password_element.submit()

browser.close()

