# Dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

refreshments = {'Chicago': 8, 'Boston': 7, 'Atlanta': 6, 'Washington DC': 5, 'Raleigh': 4, 
                'New York': 3, 'Los Angeles': 2, 'Austin': 1}
accomodation = {'Chicago': 7, 'New York': 8, 'Boston': 6, 'Austin': 5, 'Atlanta': 2, 
                'Raleigh': 4, 'Los Angeles': 3, 'Washington DC': 1}
crime =        {'Atlanta':8, 'Washington DC':7, 'Austin':6, 'Chicago':5, 'Raleigh':4,
                'Boston':3, 'Los Angeles':2, 'New York':1}
emergency =    {'Chicago': 7, 'New York': 8, 'Boston': 6, 'Washington DC': 5, 'Austin': 1,
                'Atlanta': 3, 'Los Angeles': 4, 'Raleigh': 2}
errands =      {'Chicago': 6, 'New York': 7, 'Boston': 8, 'Washington DC': 5, 'Atlanta': 3, 'Austin': 2,
                'Los Angeles': 4, 'Raleigh': 1}
fitness =      {'Chicago': 7, 'New York': 8, 'Boston': 6, 'Washington DC': 5, 'Austin': 3,
                'Los Angeles': 4, 'Atlanta': 1, 'Raleigh': 2}

amenities_ranking = {'City Name':          ['Chicago city, Illinois','Boston city, Massachusetts',
                                            'Atlanta city, Georgia','Washington city, District of Columbia DC',
                                            'Raleigh city, North Carolina','New York city, New York',
                                            'Los Angeles city, California','Austin city, Texas'],
                    'Refreshments':        [8,7,6,5,4,3,2,1],
                    'Accomodations':       [7,6,2,1,4,8,3,5],
                    'Crime Rate':          [5,3,8,7,4,1,2,6],
                    'Emergency':           [7,6,3,5,2,8,4,1],
                    'Errands':             [6,8,3,5,1,7,4,2],
                    'Fitness':             [7,6,1,5,2,8,4,3]}

amenities_ranking_df = pd.DataFrame(amenities_ranking)
amenities_ranking_df = amenities_ranking_df[['City Name','Refreshments','Accomodations',
                                            'Emergency','Errands','Fitness',
                                            'Crime Rate']]
amenities_ranking_df['Total Score'] = amenities_ranking_df.sum(axis=1)
amenities_ranking_df

# Sort by total score
amenities_ranking_df = amenities_ranking_df.sort_values('Total Score')[::-1]
amenities_ranking_df = amenities_ranking_df.reset_index(drop=True)
rank = [1,2,3,4,5,6,7,8]

amenities_ranking_df['Final Rank'] =  rank
amenities_ranking_df

amenities_ranking_df.to_csv("../Results/Final City Ranking for Amenities.csv")

