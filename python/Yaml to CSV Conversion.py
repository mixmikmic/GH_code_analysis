import yaml

dataset=[]

batting_data={}

for num in range(335982,1082651) :
    try :
        with open("ipl/"+str(num)+".yaml", 'r') as stream:
            for data in yaml.load_all(stream):
                dataset.append(data.get('info'))
                batting_data[num]=(data.get('innings'))
    except Exception :
        continue
        

batsmanDict={}

bowlerDict={}

def setAverage(innings,team_batting_average,team_bowling_average) :
    
    wickets=0

    totalRuns=0

    runs=0

    for over in innings :

        bowler_name={}
            
        batsman_name={}

        for key,value in over.items() :

            bowler=over[key]['bowler']

            batsman=over[key]['batsman']
            
            if(bowler not in bowler_name) :
                
                bowler_name[bowler]=bowler
                
            if(batsman not in batsman_name) :
                
                batsman_name[batsman]=batsman

            if(batsman not in batsmanDict) :

                batsmanDict[batsman]={'runs':0,'wickets':0}

            if(bowler not in bowlerDict) :

                bowlerDict[bowler]={'runs':0,'wickets':0}

            batsmanDict[batsman]['runs']=over[key]['runs']['batsman'] + batsmanDict[batsman]['runs']

            bowlerDict[bowler]['runs']=over[key]['runs']['total']+bowlerDict[bowler]['runs']

            if('wicket' in over[key]) :

                wicket=over[key]['wicket']
                
                if(wicket['player_out'] in batsmanDict) :

                    batsmanDict[wicket['player_out']]['wickets']=batsmanDict[wicket['player_out']]['wickets']+1

                if(wicket['kind'] != 'run out') :

                    #print(wicket['kind'])

                    bowlerDict[bowler]['wickets']=bowlerDict[bowler]['wickets']+1

            totalRuns=totalRuns+over[key]['runs']['batsman']
            
            #print(str(batsmanDict[batsman])+":"+str(totalRuns))

    bowling_avg=0

    total_avg=0

    for name,value in bowler_name.items() :

        if(bowlerDict[name]['wickets']==0) :

            bowlerDict[name]['average']=5

        else :

            bowlerDict[name]['average']=bowlerDict[name]['runs']/bowlerDict[name]['wickets']

        total_avg=total_avg+bowlerDict[name]['average']

    bowling_avg=total_avg/len(bowler_name)
    
    team_bowling_average.append(bowling_avg)

    batting_avg=0

    total_avg=0

    for name,value in batsman_name.items() :

        if(batsmanDict[name]['wickets']==0) :

            batsmanDict[name]['average']=5

        else :

            batsmanDict[name]['average']=batsmanDict[name]['runs']/batsmanDict[name]['wickets']

        total_avg=total_avg+batsmanDict[name]['average']

    batting_avg=total_avg/len(batsman_name)
    
    team_batting_average.append(batting_avg)



team_1=[]
team_2=[]

team_1_batting_average=[]

team_2_batting_average=[]

team_1_bowling_average=[]

team_2_bowling_average=[]

i=0

for key,data in batting_data.items() :

    batting=batting_data[key]
    
    try :
        
        team1=batting[0]['1st innings']['team']

        team_1_innings=batting[0]['1st innings']['deliveries']

        team2=batting[1]['2nd innings']['team']

        team_2_innings=batting[1]['2nd innings']['deliveries']
        
    except Exception :
        
        print('error'+ str(key))
        
        continue

    team_1.append(team1)

    team_2.append(team2)


    setAverage(team_1_innings,team_1_batting_average,team_1_bowling_average)

    setAverage(team_2_innings,team_2_batting_average,team_2_bowling_average)
    

city=[]
dates=[]
gender=[]
match_type=[]
win_by_runs=[]
win_by_wickets=[]
winner=[]
toss_winner=[]
toss_decision=[]
umpire_1=[]
umpire_2=[]
venue=[]
player_of_match=[]

for info in dataset :
    
    if 'city' in info :
        
        city.append(info['city'])
        
    else :
        
        city.append('')
        
    dates.append(info['dates'][0])
    
    gender.append(info['gender'])
    
    match_type.append(info['match_type'])
    
    outcome = info['outcome']
    
    teams=info['teams']
    
    venue.append(info['venue'])
    
    if 'player_of_match' in info :
    
        player_of_match.append(info['player_of_match'][0])
        
    else :
        
        player_of_match.append('')
    
    toss_winner.append(info['toss']['winner'])
    
    toss_decision.append(info['toss']['decision'])
    
    if 'umpires' in info :
    
        umpires = info['umpires']

        umpire_1.append(umpires[0])

        umpire_2.append(umpires[1])
        
    else :
        
        umpire_1.append('')

        umpire_2.append('')
    
    if 'winner' in outcome :
        
        winner.append(outcome['winner'])
        
    else :
        
        winner.append(outcome['result'])
        
        win_by_wickets.append(0)
        
        win_by_runs.append(0)
        
    if 'by' in outcome :
        
        by = outcome['by']
        
        if 'runs' in by :
            
            win_by_runs.append(by['runs'])
            
            win_by_wickets.append(0)
        else :
            
            win_by_wickets.append(by['wickets'])
            
            win_by_runs.append(0)
            
print(len(team_1_batting_average))





raw_data = {'team 1':team_1,
            'team 2':team_2,
            'venue':venue,
            'date':dates,
            'city':city,
            'gender':gender,
            'toss_winner':toss_winner,
            'toss_decision':toss_decision,
            'team_1_batting_average':team_1_batting_average,
            'team_2_batting_average':team_2_batting_average,
            'team_1_bowling_average':team_1_bowling_average,
            'team_2_bowling_average':team_2_bowling_average,
            'umpire_1':umpire_1,
            'umpire_2':umpire_2,
            'match_type':match_type,
            'win_by_runs':win_by_runs,
            'win_by_wickets':win_by_wickets,
            'winner':winner,
            'player of the match':player_of_match}

import pandas as pd

raw_data['team 1']

df = pd.DataFrame(raw_data)

df.tail()

df.head()

df.to_csv('ipl.csv')



