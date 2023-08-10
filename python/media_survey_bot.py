"""A selenium-based webscraper to answer the Trump media survey"""

import random
from random_person import personal_info
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException
import time

import os
chromedriver = "/Users/ramohse/Documents/chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver
driver = webdriver.Chrome(chromedriver)

def answer_survey():

    url_base = 'https://action.donaldjtrump.com/mainstream-media-accountability-survey/'
    driver.get(url_base)
    WebDriverWait(driver, 10)
    
    ### Question 01
    driver.find_element_by_id('id_question_3382_1').click()
    WebDriverWait(driver, (random.random()*0.5))

    ### Question 02
    ### Randomized answer
    q_02_ans = str(random.randint(0,2))
    q_02_id = 'id_question_3383_' + q_02_ans
    driver.find_element_by_id(q_02_id).click()
    WebDriverWait(driver, (random.random()*0.5))

    ### Question 03
    driver.find_element_by_id('id_question_3384_0').click()
    WebDriverWait(driver, (random.random()*0.5))

    ### Question 04
    driver.find_element_by_id('id_question_3385_1').click()
    WebDriverWait(driver, (random.random()*0.5))

    ### Question 05
    ### Will click a random number of answers
    q_05_ans = random.randint(0,7)


    for i in range(0, q_05_ans):
        str_i = str(i)
        q_05_id = 'id_question_3386_' + str_i
        driver.find_element_by_id(q_05_id).click()

    WebDriverWait(driver, (random.random()*0.5))


    ### Question 06
    ### Will click a random number of answers
    q_06_ans = random.randint(0,3)

    for i in range(0, q_06_ans + 1):
        str_i = str(i)
        q_06_id = 'id_question_3387_' + str_i
        driver.find_element_by_id(q_06_id).click()
    WebDriverWait(driver, (random.random()*0.5))


    ### Question 07
    news_sources_list = ['the Economist', 'Brietbart', 'Infowars', 'Reuters', 'The Guardian', 'CSPAN', 'Huffington Post', 
                         'Wall Street Journal', 'Forbes', 'Time', 'ABC', 'NBC', 'CBS', 'ESPN', 'AP', 'New York Times', 
                         'Columbus Dispatch', 'Miami Herald', 'LA Times', 'Chicago Tribune', 'The Herald', 'The Sentinel',
                        'Facebook', 'Twitter', 'al Jazeera', 'The Onion', 'NPR', 'USA Today', 'Daily News', 'Boston Globe',
                        'New York Post', 'Yahoo News', 'Google', 'The Examiner', 'Topix', 'Bing', 'Drudge Report', 'BBC']

    ### Random sampling to pick n < 5 news sources to populate our answer
    num_sources = random.randint(1,4)
    list_indices = list(range(0, len(news_sources_list)))
    sources = []

    for i in range(0, num_sources):
        idx = random.choice(list_indices)
        sources.append(news_sources_list[idx])
        list_indices.remove(idx)

    sources_str = ', '.join(sources)
    news_sources = 'Yes. ' + sources_str

    driver.find_element_by_id('id_question_3388').send_keys(news_sources)
    WebDriverWait(driver, (random.random()*0.5))


    ### Question 08
    ### Chooses one random source from the list to populate answer
    top_choice = random.choice(sources)
    driver.find_element_by_id('id_question_3390').send_keys(top_choice)
    WebDriverWait(driver, (random.random()*0.5))


    ### Question 09
    driver.find_element_by_id('id_question_3392_0_0').click()
    WebDriverWait(driver, (random.random()*0.5))


    ### Question 10
    driver.find_element_by_id('id_question_3393_0_1').click()
    WebDriverWait(driver, (random.random()*0.5))


    ### Question 11
    driver.find_element_by_id('id_question_3394_0_1').click()
    WebDriverWait(driver, (random.random()*0.5))


    ### Question 12
    driver.find_element_by_id('id_question_3395_0_0').click()
    WebDriverWait(driver, (random.random()*0.5))


    ### Question 13
    driver.find_element_by_id('id_question_3396_0_1').click()
    WebDriverWait(driver, (random.random()*0.5))


    ### Question 14
    driver.find_element_by_id('id_question_3397_0_2').click()
    WebDriverWait(driver, (random.random()*0.5))


    ### Question 15
    ### Will click random answer. If "other" is chosen, "muslims" is put into the text box
    q_15_ans = str(random.randint(0,3))

    if q_15_ans == '3':
        driver.find_element_by_id('id_question_3399_0_Other').click()
        driver.find_element_by_id('id_question_3399_1').send_keys('muslims')   
    else:
        q_15_id = 'id_question_3399_0_' + q_15_ans
        driver.find_element_by_id(q_15_id).click()
    WebDriverWait(driver, (random.random()*0.5))


    ### Question 16
    driver.find_element_by_id('id_question_3400_0_1').click()
    WebDriverWait(driver, (random.random()*0.5))


    ### Question 17
    driver.find_element_by_id('id_question_3402_0_1').click()
    WebDriverWait(driver, (random.random()*0.5))


    ### Question 18
    driver.find_element_by_id('id_question_3403_0_1').click()
    WebDriverWait(driver, (random.random()*0.5))


    ### Question 19
    driver.find_element_by_id('id_question_3404_0_1').click()
    WebDriverWait(driver, (random.random()*0.5))


    ### Question 20
    driver.find_element_by_id('id_question_3406_0_1').click()
    WebDriverWait(driver, (random.random()*0.5))


    ### Question 21
    driver.find_element_by_id('id_question_3407_0_1').click()
    WebDriverWait(driver, (random.random()*0.5))


    ### Question 22
    driver.find_element_by_id('id_question_3408_0_0').click()
    WebDriverWait(driver, (random.random()*0.5))


    ### Question 23
    ### Randomized answer
    q_23_ans = str(random.randint(0,2))
    q_23_id = 'id_question_3409_0_' + q_23_ans
    driver.find_element_by_id(q_23_id).click()
    WebDriverWait(driver, (random.random()*0.5))


    ### Question 24
    ### Randomized answer
    q_24_ans = str(random.randint(0,2))
    q_24_id = 'id_question_3410_0_' + q_24_ans
    driver.find_element_by_id(q_24_id).click()
    WebDriverWait(driver, (random.random()*0.5))


    ### Question 25
    ### Randomized answer
    q_25_ans = str(random.randint(0,2))
    q_25_id = 'id_question_3411_' + q_25_ans
    driver.find_element_by_id(q_25_id).click()
    WebDriverWait(driver, (random.random()*0.5))

    ### Enters in randomized person info, submits results
    user_info = personal_info()
    full_name = user_info[0]
    email = user_info[1]
    postal_code = user_info[2]

    driver.find_element_by_id('id_full_name').send_keys(full_name)
    driver.find_element_by_name('email').send_keys(email)
    driver.find_element_by_name('postal_code').send_keys(postal_code)
    WebDriverWait(driver, 10)
    driver.find_element_by_name('respond').click()
    time.sleep(5)

i = 0

while i < 6:
    answer_survey()
    i += 1

