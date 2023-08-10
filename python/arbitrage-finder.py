### imports
import pandas as pd
import sys

sys.path.append('..')
from src.dashboardScraper import DashboardScraper
from src.arbitrageOptimizer import ArbitrageOptimizer

### get current games
scraper = DashboardScraper()
scraper.connect()
games = scraper.get_json_data()
scraper.disconnect()

### load game into the arbitrageOptimizer
arbitrageOptimize = ArbitrageOptimizer()
for game in games:
    arbitrageOptimize.load_game(game)
    print arbitrageOptimize.get_optimal_ratio()
    print arbitrageOptimize.get_arbitrage_opportunity()

games

### imports
import joblib
import pandas as pd
import sys

sys.path.append('..')
from src.arbitrageOptimizer import ArbitrageOptimizer

games = joblib.load('../data/sample.dat')
arbitrageOptimizer = ArbitrageOptimizer()

game = games[0]
arbitrageOptimizer.load_game(game)

available_bookies = ['bet365', 'Interwetten', 'William Hill', 'Unibet', 'bwin', 'Tipico']
bookies_mask = [bookie in available_bookies for bookie in arbitrageOptimizer._odds_matrix.columns]

filtered_odd_matrix = arbitrageOptimizer._odds_matrix.iloc[:,bookies_mask]

filtered_odd_matrix

for game in games:
    arbitrageOptimizer.load_game(game)
    print arbitrageOptimizer._odds_matrix.columns



