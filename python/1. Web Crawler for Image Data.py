from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.common.exceptions import TimeoutException

def go_to_main_page(tissue_type):
    
    # Launch Chrome
    driver = webdriver.Remote(
        command_executor='http://127.0.0.1:4444/wd/hub',
        desired_capabilities=DesiredCapabilities.CHROME)
    
    # Navigate to Page
    driver.get("https://brd.nci.nih.gov/brd/image-search/search_specimen/searchForm")
    
    # Type in Search Field 
    search_box = driver.find_element_by_name("query")
    search_box.send_keys("tissue:"+tissue_type)
    search_box.send_keys(Keys.RETURN)
    
    print("Reached main page for tissue_type: " + tissue_type)
    
    # Tell Chrome that which is the main window
    main_window_handle = None
    while not main_window_handle:
        main_window_handle = driver.current_window_handle
    
    return driver, main_window_handle


def navigate_to_page(driver, page_num):
    delay = 5
    for i in range(0,page_num):
        print("Navigating to page: "+ str(i+1))
        next_button = driver.find_element_by_css_selector('a.nextLink')
        next_button.click()
        try:
            wait = WebDriverWait(driver, delay)
            next_button = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'a.nextLink')))
        except TimeoutException:
            print("Timeout, cannot find nextbutton")
    print("Navigated to page: " + str(page_num))

def run_batch(driver, main_window_handle, batch_num): 
    delay = 5 # seconds
    if (batch_num != 5):
        for j in range(6*batch_num-5,6*batch_num+1):
            glass = driver.find_element_by_xpath('//*[@id="container"]/div[3]/table/tbody/tr['+str(j)+']/td[10]/a')
            driver.execute_script("arguments[0].click()", glass)

            image_window_handle = None
            while not image_window_handle:
                for handle in driver.window_handles:
                    if handle != main_window_handle:
                        image_window_handle = handle
                        break

            driver.switch_to_window(image_window_handle)

            try:
                wait = WebDriverWait(driver, delay)
                download_button = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="img"]')))
                driver.execute_script("arguments[0].click()", download_button)
            except TimeoutException:
                print("Timeout, cannot find download button!")

            driver.close()
            driver.switch_to_window(main_window_handle)
    else:
        for j in range(6*batch_num-5,6*batch_num-4):
            glass = driver.find_element_by_xpath('//*[@id="container"]/div[3]/table/tbody/tr['+str(j)+']/td[10]/a')
            driver.execute_script("arguments[0].click()", glass)

            image_window_handle = None
            while not image_window_handle:
                for handle in driver.window_handles:
                    if handle != main_window_handle:
                        image_window_handle = handle
                        break

            driver.switch_to_window(image_window_handle)

            try:
                wait = WebDriverWait(driver, delay)
                download_button = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="img"]')))
                driver.execute_script("arguments[0].click()", download_button)
            except TimeoutException:
                print("Timeout, cannot find download button!")

            driver.close()
            driver.switch_to_window(main_window_handle)
    print("Batch " + str(batch_num) + " completed!")

driver, main_window_handle = go_to_main_page("liver")

page_num = 9
navigate_to_page(driver, page_num)

batch_num = 1 # Select from 1 to 5 ONLY
run_batch(driver, main_window_handle, batch_num)

# When download is completed
driver.close()



