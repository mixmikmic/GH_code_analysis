### import 
from collections import namedtuple
from datetime import datetime
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

### 
options = webdriver.ChromeOptions()
#options.add_argument('headless') 
options.add_argument('window-size=1200x600')
driver = webdriver.Chrome(chrome_options=options)
driver.get('http://184.73.28.182/')

### get list of all games listed as suggested bets
container_element = driver.find_element_by_class_name('container')
table_element = container_element.find_element_by_css_selector('table.table-striped')
suggested_games = table_element.find_elements_by_class_name('accordion-toggle')

### expand all the suggested games
for game in suggested_games:
    game.click()

### extract odds from bookies
odds_matrices = table_element.find_elements_by_class_name('hiddenRow')

### create named tuple from game and odd_matrix containing the following fields:
#   * query_time 
#   * time_to_game
#   * game
#   * league
#   * odd_matrix

# get input for the function to be implemented
game = suggested_games[0]
odd_matrix = odds_matrices[0]

# initialize named tuple
scraped_game = namedtuple('ScrapedGame',  ['timestamp', 'date', 'game', 'league', 'odd_matrix'])

# fill values in named tuple from game
game_contents = [element.text for element in game.find_elements_by_css_selector('td')]
scraped_game.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
scraped_game.date = str(game_contents[4])
scraped_game.game = str(game_contents[1])
scraped_game.league = str(game_contents[2])

# get single rows from odd matrix
table = odd_matrix.find_element(By.CLASS_NAME, 'accordian-body').find_element(By.CLASS_NAME, 'table-striped')
table_body = table.find_element(By.CSS_SELECTOR,'tbody')
table_body_elements = table_body.find_elements(By.CSS_SELECTOR, 'tr')

# add column in scraped_odd_matrix, containing the bookie as column name, and odds as values
scraped_odd_matrix = pd.DataFrame(index=['1', 'X', '2'])
for body in table_body_elements:
    body_elements = body.find_elements(By.CSS_SELECTOR, 'td')
    
    bookie = str(body_elements[0].text)
    odds_1 = float(body_elements[1].text.split('\n')[0])
    odds_X = float(body_elements[2].text.split('\n')[0])
    odds_2 = float(body_elements[3].text.split('\n')[0])
    
    scraped_odd_matrix[bookie] = pd.Series([odds_1, odds_X, odds_2], index=scraped_odd_matrix.index)

scraped_game.odd_matrix = scraped_odd_matrix

scraped_game.odd_matrix

### extract information from game
from collections import namedtuple
import datetime

game = suggested_games[0]
game_contents = [element.text for element in game.find_elements_by_css_selector('td')]

game_info = {}
game_info['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
game_info['sport'] = str(game_contents[0])
game_info['match_title'] = str(game_contents[1])
game_info['league'] = str(game_contents[2])
game_info['result_to_bet'] = str(game_contents[3])
game_info['date'] = str(game_contents[4])
game_info['time_to_match'] = str(game_contents[5])
game_info['best_bookie'] = str(game_contents[6])
game_info['best_odds'] = float(game_contents[7])
game_info['mean'] = float(game_contents[8].split('/')[0].strip())
game_info['median'] = float(game_contents[8].split('/')[1].strip())

game_info

### extract information from odds_matrix
odds_matrix = odds_matrices[0]
table = odds_matrix.find_element(By.CLASS_NAME, 'table-striped.table-bordered')
table_body = table.find_element(By.CSS_SELECTOR,'tbody')
table_body_elements = table_body.find_elements(By.CSS_SELECTOR, 'tr')
# iterate over table_body_elements
OddsInfoArray = []
for body in table_body_elements:
    body_elements = body.find_elements(By.CSS_SELECTOR, 'td')

    OddsInfo = {}
    OddsInfo['bookie'] = str(body_elements[0].text)
    OddsInfo['odds_1'] = float(body_elements[1].text.split('\n')[0])
    OddsInfo['timestamp_1'] = str(body_elements[1].text.split('\n')[1])
    OddsInfo['odds_X'] = float(body_elements[2].text.split('\n')[0])
    OddsInfo['timestamp_X'] = str(body_elements[2].text.split('\n')[1])
    OddsInfo['odds_2'] = float(body_elements[3].text.split('\n')[0])
    OddsInfo['timestamp_2'] = str(body_elements[3].text.split('\n')[1])
    OddsInfoArray.append(OddsInfo)

### check if games are present:
str(container_element.text) == 'No advantageous bet opportunities currently available.'

