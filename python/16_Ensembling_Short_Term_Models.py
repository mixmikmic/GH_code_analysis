from stock_utils import *

df = pd.DataFrame()
df = df.from_csv('stock_data/spy.csv')

daily_movements = get_price_movement_percentages(df)
movement_categories = categorize_movements(daily_movements, n_cats=4)

period_len = int(len(daily_movements) / 5)
train_movement_categories = movement_categories[0:4*period_len]
valid_movement_categories = movement_categories[4*period_len+1:5*period_len]

train_two_day_movement_trends = get_trends(train_movement_categories, 2)
train_three_day_movement_trends = get_trends(train_movement_categories, 3)

valid_two_day_movement_trends = get_trends(valid_movement_categories, 2)
valid_three_day_movement_trends = get_trends(valid_movement_categories, 3)

## Volume
relative_volumes = get_relative_volume(df, relative_period=20)
train_volumes = relative_volumes[0:4*period_len]
valid_volumes = relative_volumes[4*period_len+1:5*period_len]

train_volume_categories = categorize_volumes(train_volumes)
valid_volume_categories = categorize_volumes(valid_volumes)

sample_size = 27
one = OneDayModel(train_movement_categories)
two = TwoDayModel(train_movement_categories)
three = ThreeDayModel(train_movement_categories)
two_volume = TwoDayVolumeModel(train_movement_categories, train_volume_categories)
n_runs = 100000

one_wins = 0
two_wins = 0
three_wins = 0
two_volume_wins = 0
ensemble_wins = 0

for i in range(n_runs):
    one_score = 0
    two_score = 0
    three_score = 0
    two_volume_score = 0
    ensemble_score = 0
    
    ## Generate a sample
    sample_categories, sample_volumes = select_data_sample(valid_movement_categories, sample_size, data2=valid_volumes)
    sample_volume_categories = categorize_volumes(sample_volumes)
    
    one_predictions = one.predict(sample_categories[1:])
    two_predictions = two.predict(sample_categories[1:])
    three_predictions = three.predict(sample_categories[0:]) ## Needs an extra category in the beginning
    two_volume_predictions = two_volume.predict(sample_volume_categories[1:])

    for i in range(len(sample_categories) - 2):
        if (sample_categories[i+2] == one_predictions[i]):
            one_score += 1
        if (sample_categories[i+2] == two_predictions[i]):
            two_score += 1
        if (sample_categories[i+2] == three_predictions[i]):
            three_score += 1
        if (sample_categories[i+2] == two_volume_predictions[i]):
            two_volume_score += 1
        
        mode = stats.mode([three_predictions[i], two_predictions[i], two_volume_predictions[i], one_predictions[i]])[0][0]
        ensemble_prediction = mode
        if (sample_categories[i+2] == ensemble_prediction):
            ensemble_score += 1

    together = np.array([one_score, two_score, three_score, two_volume_score, ensemble_score])
    winner = np.argwhere(together == np.amax(together)).flatten()
    #print(len(winner))
    if len(winner) > 1:
        winner = np.random.choice(winner)
                   
    if winner == 0:
        one_wins += 1
        #print('One won, score is '+ str(one_score))
    elif winner == 1:
        two_wins += 1
        #print('Two won, score is '+ str(two_score))
    elif winner == 2:
        three_wins += 1
        #print(three_score)
    elif winner == 3:
        two_volume_wins += 1
        #print(two_volume_score)
        
    elif winner == 4:
        ensemble_wins += 1
        #print(ensemble_score)
    
print('One day model won ' + str(one_wins) + ' times, or ' 
      + str('{0:.2f}'.format(100*one_wins/n_runs)) + ' percent of the time')
print('Two day model won ' + str(two_wins) + ' times, or ' 
      + str('{0:.2f}'.format(100*two_wins/n_runs)) + ' percent of the time')
print('Three day model won ' + str(three_wins) + ' times, or ' 
      + str('{0:.2f}'.format(100*three_wins/n_runs)) + ' percent of the time')
print('Volume day model won ' + str(two_volume_wins) + ' times, or ' 
      + str('{0:.2f}'.format(100*two_volume_wins/n_runs)) + ' percent of the time')
print('Ensemble model won ' + str(ensemble_wins) + ' times, or ' 
      + str('{0:.2f}'.format(100*ensemble_wins/n_runs)) + ' percent of the time')

