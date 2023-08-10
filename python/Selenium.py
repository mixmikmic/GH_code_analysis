from selenium import webdriver
browser = webdriver.Firefox()
browser.get('http://www.google.com')

type(browser)

def get_flipkart(x):
    print('Opening Flipkart.com............')
    from selenium import webdriver
    browser=webdriver.Firefox()
    browser.get('http://www.flipkart.com')
    #searchElem=browser.find_element_by_css_selector('.LM6RPg+')
    searchElem=browser.find_element_by_class_name('LM6RPg')
    searchElem.send_keys(x)
    (browser.find_element_by_css_selector('.vh79eN')).click()
    #searchsubmit=browser.find_element_by_css_selector('.LM6RPg+')
    #searchElem.submit()
    print('\n')
    print('Flipkart.com has been opened ')
    print('\n')
    print('\n')

x=raw_input()
#call()
get_flipkart(x)

from selenium import webdriver
browser=webdriver.Firefox()
browser.get('http://www.flipkart.com')



