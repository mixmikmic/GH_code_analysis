from selenium import webdriver
browser = webdriver.Chrome()
url = "https://offense.roadpolice.am/violation"
browser.get(url)

pin_code = "173679JNYEJM"

input_form = browser.find_element_by_xpath('//*[@id="pin"]')
input_form.send_keys(pin_code)
submit_button = browser.find_element_by_tag_name("button")
submit_button.click()

js = '''html = document.getElementsByTagName('html')[0];
            return html.outerHTML;'''
html = browser.execute_script(js).encode('utf-8')

with open("page_source.html","w") as f:
    f.write(html)

with open("page_source.html","w") as f:
    f.write(browser.page_source.encode('utf-8'))

date = browser.find_element_by_xpath('//*[@id="main_data"]/tbody/tr[6]/td[3]')
print(date.text)

amount = browser.find_element_by_css_selector("li b").text
print(amount)

import re

drams = re.findall("[0-9]+",amount)
print(drams[0])

browser.close()

