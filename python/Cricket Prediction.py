import pandas as pd

dataset=pd.read_csv('ipl.csv',index_col=0)

dataset = dataset.drop(columns=['gender', 'match_type','date','umpire_1','umpire_2','player of the match','win_by_runs','win_by_wickets'])

# columns with missing values
dataset.columns[dataset.isnull().any()]

# replace missing column with mode value
dataset['city'].fillna(dataset['city'].mode()[0], inplace=True)

dataset.columns[dataset.isnull().any()]

dataset.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW','RPS'],inplace=True)

dataset.head(5)

def createDict(series) :
    
    dictionary={}
    
    i=0
    
    for ser in series :
        if(ser in dictionary) :
            continue
        dictionary[ser]=i
        i=i+1
        
    return dictionary

teamDict=createDict(dataset['team 1'])

cityDict=createDict(dataset['city'])

venueDict=createDict(dataset['venue'])

tossDecisionDict=createDict(dataset['toss_decision'])

winnerDict=dict(teamDict)

winnerDict['tie']=14

winnerDict['no result']=15

venueDict

encode = {
'team 1': teamDict,
'team 2': teamDict,
'toss_winner': teamDict,
'winner': winnerDict,
'city':cityDict,
'venue':venueDict,
'toss_decision': tossDecisionDict    
 }
dataset.replace(encode, inplace=True)



def prediction(Model,X_train,y_train,X_test,y_test) :
    
    clf=Model()
    
    clf.fit(X_train,y_train)
    
    print(clf.score(X_test,y_test))
    
    return clf

def predictWinner():    
    
    from sklearn.neural_network import MLPClassifier

    from sklearn.svm import LinearSVC

    from sklearn.linear_model import LogisticRegression

    from sklearn.ensemble import RandomForestClassifier

    clf_A = prediction(MLPClassifier,X_train,y_train,X_test,y_test)

    clf_B = prediction(LinearSVC,X_train,y_train,X_test,y_test)

    clf_C = prediction(LogisticRegression,X_train,y_train,X_test,y_test)

    clf_D = prediction(RandomForestClassifier,X_train,y_train,X_test,y_test)
    

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegression

def buildModel(dataset,team1,team2) :

    
    dataset=dataset[
        ((dataset['team 1']==team1)&(dataset['team 2']==team2) | 
         (dataset['team 1']==team2)&(dataset['team 2']==team1))
    ]


    winner = dataset['winner']

    features = dataset.drop('winner',axis=1)

    features = pd.get_dummies(features)

    clf=LogisticRegression()

    clf.fit(features,winner)

    return clf

def getPrediction(city,team1,team2,team1_batting_avg,team1_bowling_avg,team2_batting_avg,team2_bowling_avg,toss_decision,toss_winner,venue) :

    predictionSet = pd.DataFrame({
        'city':cityDict[city],
        'team 1':teamDict[team1],
        'team 2':teamDict[team2],
        'team_1_batting_average':team1_batting_avg,
        'team_1_bowling_average':team1_bowling_avg,
        'team_2_batting_average':team2_batting_avg,
        'team_2_bowling_average':team2_bowling_avg,
        'toss_decision':[toss_decision],
        'toss_winner':teamDict[toss_winner],
        'venue':venueDict[venue]
    })

    predictionSet = pd.get_dummies(predictionSet)
    
    clf=buildModel(dataset,teamDict[team1],teamDict[team2])
    
    prediction=clf.predict(predictionSet)
    
    for key,value in teamDict.items() :
        
        if(value==prediction) :
            
            print(key)


getPrediction('Bangalore','KKR','RCB',5.0,5.000000,3.0,12.0,'field','RCB','M Chinnaswamy Stadium')

getPrediction('Chandigarh','KXIP','CSK',5.0,54.000000,5.0,47.0,'bat','CSK','Punjab Cricket Association Stadium, Mohali')

getPrediction('Delhi','DD','RR',5.0,21.000000,5.0,5.0,'bat','RR','Feroz Shah Kotla')



