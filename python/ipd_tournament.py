from prisoners import *
from random import shuffle

num_defectors = 0
num_cooperators = 0

agents = []
agents.append(Goldfish())
agents.append(PavlovianTFT())
agents.append(Roki())
agents.append(Average_Joe())
agents.append(Bob())
agents.append(ElepertPrisoner())
agents.append(Kaitlyn())
agents.append(SandwichPrisoner())
agents.append(SmoothCriminal2AkaPrettyPrincess())
agents.append(Snail())
agents.append(TigerMountainPrisoner())
agents.append(VickyPrisoner())
agents.append(wtw())
agents.append(NotAMobster())
agents.append(RandomDefectTFT())
agents.append(MostBestUnderling())

for i in range(num_defectors):
    agents.append(Defector())

for i in range(num_cooperators):
    agents.append(Cooperator())


for agent in agents:
    agent.__init__()
    agent.score = 0
    agent.wins = 0
    agent.cooperations = 0
    agent.num_rounds = 0

scoring = {'CC': (3, 3, 1, 1), 'CD': (0, 5, 0, 1), 'DC': (5, 0, 1, 0), 'DD': (1, 1, 0, 0)}

def play_game(agent1, agent2, rounds, printing=False):
    # Reinit the agents (Clear their history)
    agent1.__init__()
    agent2.__init__()
    
    if len(agent1.personal_history) != 0 or len(agent2.personal_history) != 0:
        raise ValueError('Agent histories are not empty')
    
    if printing: print("1 2 S S W W P P")
    
    for _ in range(rounds):
        decision1 = agent1.makeDecision(agent2.getHistory())
        decision2 = agent2.makeDecision(agent1.getHistory())
        
        agent1.addToHistory(decision1)
        agent2.addToHistory(decision2)
        
        score1, score2, win1, win2 = scoring[decision1 + decision2]
        
        
        agent1.score += score1
        agent2.score += score2
        
        agent1.wins += win1
        agent2.wins += win2
        
        agent1.cooperations += (decision1 == 'C')
        agent2.cooperations += (decision2 == 'C')
        
        if printing: print(decision1, decision2, score1, score2, win1, win2, agent1.score, agent2.score)
        
        agent1.num_rounds += 1
        agent2.num_rounds += 1

tournament_rounds = 50
min_rounds = 200
max_rounds = 300

for _ in range(tournament_rounds):
    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents[i + 1:]):
            play_game(agent1, agent2, random.randint(min_rounds, max_rounds))
    print('.', end="")
    
print()
print('done')

print('{:<50}| {:<10}| {:<10}'.format('Name', 'Score', 'Cooperations'))
print('----------------------------------------------------------------------------')
for agent in sorted(agents, key=lambda a:a.score / a.num_rounds, reverse = True):
    print('{:<50}| {:<9f} | {:<9f}'.format(agent.name, agent.score / agent.num_rounds, agent.cooperations / agent.num_rounds))

print('{:<50}| {:<10}| {:<10}'.format('Name', 'Wins', 'Cooperations'))
print('----------------------------------------------------------')
for agent in sorted(agents, key=lambda a:a.wins / a.num_rounds, reverse = True):
    print('{:<50}| {:<10f}| {:<10f}'.format(agent.name, agent.wins / agent.num_rounds, agent.cooperations / agent.num_rounds))

print('{:<50}| {:<10}| {:<10}'.format('Name', 'Score', 'Cooperations'))
print('----------------------------------------------------------')
for agent in sorted(agents, key=lambda a:a.cooperations / a.num_rounds, reverse = True):
    print('{:<50}| {:<10f}| {:<10f}'.format(agent.name, agent.score / agent.num_rounds, agent.cooperations / agent.num_rounds))

print('{:<50}| {:<10}| {:<10}'.format('Name', 'Score', 'Defects'))
print('----------------------------------------------------------')
for agent in sorted(agents, key=lambda a:(a.num_rounds - a.cooperations) / a.num_rounds, reverse = True):
    print('{:<50}| {:<10f}| {:<10f}'.format(agent.name, agent.score / agent.num_rounds, (agent.num_rounds - agent.cooperations) / agent.num_rounds))

test_agents = [MostBestUnderling(), PavlovianTFT()]
for agent in test_agents:
    agent.__init__()
    agent.score = 0
    agent.wins = 0
    agent.cooperations = 0
    agent.num_rounds = 0

play_game(test_agents[0], test_agents[1], 10, printing=True)



