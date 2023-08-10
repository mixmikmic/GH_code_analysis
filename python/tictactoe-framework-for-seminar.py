from copy import deepcopy
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
import itertools
from IPython import display

class Board(object):
    """ A TicTacToe Board. """
    def __init__(self, choose_random = True, win_pos = np.nan, rand_seed = 2):
        self.fields = np.array(9*[0])
        self.num_moves = sum(abs(self.fields))
        if choose_random:
            np.random.seed(rand_seed)
            # contains all choose(9,3) == 84 possible combinations of three numbers between 0 and 8
            all_combinations_of_three = list(itertools.combinations(range(9), 3))
            #randomly choose 8 of those
            self.winning_positions = list(all_combinations_of_three[i] for i in np.random.randint(0, 83, 8))
            print self.winning_positions
        else:
            self.winning_positions = win_pos
            
    def reset(self):
        self.fields = np.array(9*[0])
        self.num_moves = 0
        


class Game(object):
    def __init__(self):
        self.prob = []
        self.game_state = []
        self.action = []
        self.final_game_state = []
        self.result = 0
        
    def reset(self):
        self.prob = []
        self.game_state = []
        self.action = []
        self.final_game_state = []
        self.result = 0
    
    def __str__(self):
        return self.prob
        

class Table(object):
    def __init__(self, player1, player2, rand_seed = 2):
        self.player1 = player1
        self.player2 = player2
        self.player1_won = 0
        self.player2_won = 0
        self.credits = []
        self.output = True
        self.rand_seed = rand_seed
        self.board = Board(rand_seed = self.rand_seed)
        self.game_is_on = True
        self.num_games = 0
    
    def reset(self):
        self.board.reset()
        self.game_is_on = True

    def play_a_game(self, game_id):
        self.reset()
        while self.game_is_on:
            if (self.board.num_moves + game_id)%2 == 0: #player1's turn
                a = self.player1.action(1 * self.board.fields, self.board.winning_positions)
                if self.output:
                    print(self.player1.name + " decided to play " + str(a))
                if (a<9 and 0<=a ):
                    if (self.board.fields[a] == 0):
                        self.board.fields[a] = 1
                    else:
                        print self.board.fields
                        print a
                        print self.player1.name
                        raise NameError('This field is already occupied!!')
                else:
                    raise NameError('The fields are numbered from 0 to 8!')
                    
            else: #player 2's turn
                a = self.player2.action(-1 * self.board.fields, self.board.winning_positions)
                if self.output:
                    print(self.player2.name + " decided to play " + str(a))
                if (a<9 and 0<=a):
                    if (self.board.fields[a] == 0):
                        self.board.fields[a] = -1 
                    else:    
                        print self.board.fields
                        print a
                        print self.player2.name
                        raise NameError('This field is already occupied!!')
                else:
                    raise NameError('The fields are numbered from 0 to 8!')
            self.board.num_moves += 1
            self.evaluate_board()
            
    
    def end_game(self, result):
        if self.num_games == 0:
            self.credits = np.array([self.player1_won,self.player2_won])
        else:
            self.credits = np.vstack((self.credits, [self.player1_won,self.player2_won]))     
        self.num_games += 1
        self.game_is_on = False
        self.player1.end_game(self.board.fields, result, self.num_games)
        self.player2.end_game(-self.board.fields, -result, self.num_games) 
        
        
    def evaluate_board(self):
        # if less than 5 fields are occupied, keep playing.
        if self.board.num_moves < 5:
            return 0
        else:
            for i in range(8):
                cur_win_pos = self.board.winning_positions[i]
                field_evaluated = sum( [self.board.fields[j] for j in cur_win_pos ])
                if (self.game_is_on and field_evaluated == 3):
                    self.player1_won += 1
                    if self.output:
                        print self.player1.name + " won! Winning position: " + str(cur_win_pos)
                    self.end_game(1) #player1 won!
                                        
                elif (self.game_is_on and field_evaluated == -3):
                    self.player2_won += 1
                    if self.output:
                        print self.player2.name + " won! Winning position: " + str(cur_win_pos)
                    self.end_game(-1) #player2 won!
                    
            if (self.game_is_on and (self.board.num_moves == 9)):
                if self.output: 
                    print "draw!"
                self.end_game(0) #draw!

  

class Strategy(object):
    def __init__(self):
        self.table = {}

class Player(object):
    """ A Player. """
    def __init__(self, name):
        self.name = name
    
    def reset(self):
        self.game.reset()
         
    def action(self, state, winningpos):
        pass
        
    def end_game(self, state, result, num_games):
        pass
    
    def update_strategy(self, num_games):
        pass
    

class History(object):
    def __init__(self):
        self.game = []
        
    def add_game(self, g):
        self.game.append(deepcopy(g)) 
        
    def remove_almost_all_games(self, nn):
        self.game = self.game[(-nn):(-1)]

class StrategicPlayer(Player):
    """ A player with different strategies. """
    #has an object strategy, which is a look-up table and defines all actions
    def __init__(self, name, cre=0):
        self.name = name
        self.game = Game()
        self.history = History()
        self.strategy = Strategy()    
        self.rangeeight = np.array([0,1,2,3,4,5,6,7,8]) 
        
    def hash_state(self,state): #this is not really hashing. 
        return np.array_str(state)
    
    def dehash_state(self, gs):
        c = gs.replace('[', '').replace(']','')
        d = np.fromstring(c, sep = " ")
        return d        
    
    def action(self, state, winningpos):
        hstate = self.hash_state(state)
        if hstate in self.strategy.table:
            # print 'I have seen the state before!'
            pr = self.strategy.table[hstate] #vector of probabilities
        else:
            pr = np.zeros(9)
            emptyfields = np.array(np.where(state == 0)).flatten()
            pr[emptyfields] = 1/float(len(emptyfields))
            self.strategy.table[hstate] = pr

        act = np.random.choice(self.rangeeight, p = pr)
        
        self.game.prob.append(pr[act])
        self.game.game_state.append(state)
        self.game.action.append(act)
        #if(np.size(np.array(self.strategy.table.keys())) % 200 == 0): 
        #    print np.size(np.array(self.strategy.table.keys()))
        return act
 

    def end_game(self, state, result, num_games):
        self.game.final_game_state = state
        self.game.result = result
        self.history.add_game(self.game)
        self.update_strategy(num_games)
        self.game = Game()
        # do sth with the history
    

    def update_strategy(self, num_games):
        pass
        

class RandomPlayer(Player):
    """ Random, a player with a predefined (completely random) strategy. """
    def __init__(self):
        self.name = "Rando M."
        
    def action(self, state, winningpos):
        emptyfields = np.array(np.where(state == 0)).flatten()
        if len(emptyfields) == 1:
            ind = 0
        else:
            ind = np.random.randint(0, sum(state == 0))    
        ret = emptyfields[ind]
        return ret 

class AlwaysLeftPlayer(Player):
    """ A player who always puts a mark on the first free field. """
    def __init__(self):
        self.name = "Lefto"
            
    def action(self, state, winningpos):
        emptyfields = np.array(np.where(state == 0)).flatten()
        ret = emptyfields[0]
        return ret 

class HumanPlayer(Player):
    def __init__(self, name="Human"):
        self.name = name
        
    
    def action(self, state, winningpos):
        response = "a"
        print("The current position is: " + str(state) + ".")
        s = "where do you want to make the next cross?"
        while type(response) != int:
            response = input(self.name + ", " + s + " (number between 0 and 8): ")
        return response
    
        

class ExploreExploitPlayerSolution(StrategicPlayer):
    """ A player with different strategies. """
    #has an object strategy, which is a look-up table and defines all actions
    def __init__(self, name, cre=0):
        self.name = name
        self.game = Game()
        self.history = History()
        self.strategy = Strategy()
        self.rangeeight = np.array([0,1,2,3,4,5,6,7,8]) 
    
    
    
    def update_strategy(self, num_games):
        if (num_games % 100 == 0):
    
            cum_gain_action_taken = dict(zip(self.strategy.table.keys(),
                                             [np.zeros(9) for k in range(len(self.strategy.table.keys()))]))
            #this creates a dictionary with the same keys in table but with zero entries. (could be simplified!?)
            
            
            
            # go through all games in history
            for g in self.history.game:
                # go through all game states in game g
                for decisions_game_i in range(len(g.game_state)):
                    # get hashed game state
                    a = self.hash_state(g.game_state[decisions_game_i])              
                    ac = g.action[decisions_game_i]
                    cum_gain_action_taken[a][ac] = cum_gain_action_taken[a][ac] + g.result
            
            #update strategy to always use the best action
            for game_state_hashed in self.strategy.table.keys():
                d = self.dehash_state(game_state_hashed)
                nonzerooos = np.array(np.where(d != 0)).flatten()
                cum_gain_action_taken[game_state_hashed][nonzerooos] = -np.inf
                bestact = np.argmax(cum_gain_action_taken[game_state_hashed])
                self.strategy.table[game_state_hashed] = np.zeros(9)
                self.strategy.table[game_state_hashed][bestact] = 1                

class ValuePlayerSolution(Player):
    """ Random, a player with a predefined (completely random) strategy. """
    def __init__(self):
        self.name = "Val Hand solution"
        
    def provide_value(self, state, future_state, winningpos):
        # done by Jonas Peters, Mar 2016
        field_evaluated = np.zeros(8)
        for i in range(8):
            cur_win_pos = winningpos[i]
            field_evaluated[i] = sum( [future_state[j] for j in cur_win_pos ])
        if np.max(field_evaluated) < 3:
            for i in range(8):
                if field_evaluated[i] == -2:
                    return -10
        return np.max(field_evaluated)

        
        
    def provide_value_morecases(self, state, future_state, winningpos):
        #done by Christina Heinze, Mar 2016
        field_evaluated = np.zeros(8)
        opponent_catch_22_count = 0
        opponent_wins_with_next_move = False
        
        for i in range(8):
            cur_win_pos = winningpos[i]
            overlap_future_state_w_winning = sorted([future_state[j] for j in cur_win_pos])
            overlap_current_state_w_winning = sorted([state[j] for j in cur_win_pos])
            
            if(overlap_future_state_w_winning[0] == -1 and overlap_future_state_w_winning[1] == -1
                and overlap_future_state_w_winning[2] == 1 and overlap_current_state_w_winning[2] == 0):
                # if action is not taken, then opponent can win in next move
                opponent_wins_with_next_move = True
            elif(overlap_future_state_w_winning[0] == -1 and overlap_future_state_w_winning[1] == 0
                and overlap_future_state_w_winning[2] == 1):
                # if this pattern occurs twice and if action is not taken, 
                # then opponent can build a catch-22 with next move
                opponent_catch_22_count = opponent_catch_22_count + 1  
            elif(overlap_future_state_w_winning[0] == -1):
                field_evaluated[i] = -100
            else:
                field_evaluated[i] = sum(overlap_future_state_w_winning) 
        max_entry = np.max(field_evaluated)
        
        if(max_entry == 3.0):
            # if we can win by choosing this state, value is inf
             max_entry = np.inf
        elif(opponent_wins_with_next_move):
            max_entry = 70
        elif(max_entry == 2.0 and sum(field_evaluated == max_entry) > 1):
            # if we cannot win and if we can build a 'catch-22' from this position, value should be larger than 2
             max_entry = 60
        elif(opponent_catch_22_count == 2):
            max_entry = 50
        elif(opponent_catch_22_count == 1):
            max_entry = 40
        elif(max_entry == 1.0 and sum(field_evaluated == max_entry) > 1):
            max_entry = sum(field_evaluated == max_entry)
        
        return max_entry
    
    def neighbouring_states(self, state):
        emptyfields = np.array(np.where(state == 0)).flatten()
        n_states = list()
        for ind in emptyfields:
            tmp = np.array(state)
            tmp[ind] = 1
            n_states.append(tmp)
        return n_states
    
    def action(self, state, winningpos):
        emptyfields = np.array(np.where(state == 0)).flatten()
        a = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        for i in emptyfields:
            future_state = np.array(state)
            future_state[i] = 1
            a[i] = self.provide_value(state, future_state, winningpos)
            #a[i] = self.provide_value_morecases(state, future_state, winningpos)
#            if a[i]*self.provide_value2(state, future_state, winningpos) < 0:
#                print future_state
#                print self.provide_value2(state, future_state, winningpos)                
#                print a[i]
        return np.argmax(a)

class ValuePlayerDPJonas(Player):
    """ A player with a optimal Value Function strategy (value iteration). """
    # done by Jonas Peters, April 2016
    def __init__(self):
        self.name = "DPforValueFunction"
        self.valuetable = {}    
        self.winningpos = -1
        self.gamma = 1
        
    def computetable(self):
        state_zeros = np.zeros(9).astype(np.int64)
        self.valuetable[self.hash_state(state_zeros)] = 0
        for i in range(9):
            state_ones = np.zeros(9).astype(np.int64)
            state_ones[i] = -1
            self.valuetable[self.hash_state(state_ones)] = 0    
        for i in range(6):
            delta = 0
            print np.size(self.valuetable.keys())
            for state_sh in self.valuetable.keys():
                #print state_sh
                state_s = self.dehash_state(state_sh)
                nbstates = self.neighbouring_statesplus1(state_s)
                num_actions = np.size(nbstates)/9
                if(num_actions > 0):
                    vec_vals = np.zeros(num_actions)
                    for j in range(num_actions):
                        tmp = nbstates[j]
                        #print tmp
                        if(self.return_r(tmp) == 0):
                            nbstates2 = self.neighbouring_statesminus1(tmp)
                            if(np.size(nbstates2)/9 > 0):
                                for state_sprime in nbstates2:
                                    if(self.return_r(state_sprime) == 0):
                                        state_sprimeh = self.hash_state(state_sprime)                    
                                        if (state_sprimeh in self.valuetable):
                                            vec_vals[j] = vec_vals[j] + 9.0/np.size(nbstates2)*(0 + self.gamma*self.valuetable[state_sprimeh])
                                        else:
                                            self.valuetable[state_sprimeh] = 0
                                    else: #the opponent has performed an action that ends the game
                                        vec_vals[j] = vec_vals[j] + 9.0/np.size(nbstates2)*self.return_r(state_sprime)
                            else: # we've filled the last field and it's a draw
                                vec_vals[j] = 0
                        else: # we've performed an action that ends the game (in our favor)
                            vec_vals[j] = self.return_r(tmp)
                    self.valuetable[state_sh] = np.max(vec_vals)
        #print self.valuetable.keys()              
    
    def neighbouring_statesplus1(self, state):
        emptyfields = np.array(np.where(state == 0)).flatten()
        n_states = list()
        for ind in emptyfields:
            tmp = state.copy().astype(np.int)
            tmp[ind] = 1
            n_states.append(tmp)
        return n_states

    def neighbouring_statesminus1(self, state):
        emptyfields = np.array(np.where(state == 0)).flatten()
        n_states = list()
        for ind in emptyfields:
            tmp = state.copy().astype(np.int)
            tmp[ind] = -1
            n_states.append(tmp)
        return n_states

    
    def hash_state(self,state): #this is not really hashing. 
        return np.array_str(state)
    
    def dehash_state(self, gs):
        c = gs.replace('[', '').replace(']','')
        d = np.fromstring(c, sep = " ")
        return d        
    
    def provide_value(self, state, future_state, winningpos):
        # this should never be called!?
        print future_state
        return k2
       
    def return_r(self,state_s):
        for i in range(8):
            cur_win_pos = self.winningpos[i]
            k = sum( [state_s[j] for j in cur_win_pos ])
            if k == 3:
                return 1
            if k == -3:
                return -1
        return 0
    
    
    def action(self, state, winningpos):
        
        if (self.winningpos != winningpos):
            self.valuetable = {}
            self.winningpos = winningpos
            self.computetable()
        
        emptyfields = np.array(np.where(state == 0)).flatten()
        aa = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        for i in emptyfields:
            future_state = np.array(state)
            future_state[i] = 1
            aa[i] = 0

            if(self.return_r(future_state) == 0):
                nbstates2 = self.neighbouring_statesminus1(future_state)
                if(np.size(nbstates2)/9 > 0):
                    for state_sprime in nbstates2:
                        if(self.return_r(state_sprime) == 0):
                            state_sprimeh = self.hash_state(state_sprime)                    
                            aa[i] = aa[i] + 9.0/np.size(nbstates2)*(0 + self.gamma*self.valuetable[state_sprimeh])
                        else: #the opponent has performed an action that ends the game (he wins)
                            aa[i] = aa[i] + 9.0/np.size(nbstates2)*self.return_r(state_sprime)
                else: # we've filled the last field and it's a draw
                    aa[i] = 0
            else: # we've performed an action that ends the game (in our favor)
                aa[i] = self.return_r(future_state)
        return np.argmax(aa)

class MonteCarloPlayer(Player):
    """ A player which uses a Monte Carlo Method. """
    #done by Michael Heinzer, Apr 2016
    #has an object strategy, which is a look-up table and defines all actions
    def __init__(self):
        self.name = "MonteCarlo"
        self.game = Game()
        self.history = History()
        self.strategy = Strategy()
        self.episode = []
        self.rangeeight = np.array([0,1,2,3,4,5,6,7,8]) 
        self.epsilon = 0.05
        
    def hash_state(self,state): #this is not really hashing. 
        return np.array_str(state)
    
    def dehash_state(self, gs):
        c = gs.replace('[', '').replace(']','')
        d = np.fromstring(c, sep = " ")
        return d
    
    def getNumberOfVisitedStates(self):
        return len(self.strategy.table)
    
    def getPossibleActions(self, state):
        emptyfields = np.array(np.where(state == 0)).flatten()
        actions = []
        for i in emptyfields:
            actions.append(self.hash_state(state)+str(i))
        return actions
    
    
    def getGreedyAction(self, asValues):
        maxIndex = np.argmax(asValues)
        if asValues.count(asValues[maxIndex]) == 1:
            return maxIndex
        places = []
        maxVal = asValues[maxIndex]
        for i in range(len(asValues)):
            if asValues[i] == maxVal:
                places.append(i)
        return places[int(random.uniform(0,1)*(len(places)-1))]
        
                
    
    def action(self, state, winningpos):
        #Get the attainable state action values
        stateActions = self.getPossibleActions(state)
        asValues = []
        
        #Get the state action values
        for stateaction in stateActions:
            val = 0.0
            if stateaction in self.strategy.table:
                value = self.strategy.table[stateaction]
                val = value[0]
            asValues.append(val)
        
        #Pick an action with epsilon greedy strategy
        p = np.random.uniform(0,1)
        nb = len(asValues)
        #gives us better exploration at the beginning
        greedy = self.getGreedyAction(asValues)
        #Should we choose the greedy action?
        if p < 1-self.epsilon + self.epsilon/nb:
            action = stateActions[greedy]
        else:
            #delete greedy action
            stateActions.pop(greedy)
            asValues.pop(greedy)
            nb -= 1
            #avoid bias to lower probability
            p = np.random.uniform(0,1)
            action = stateActions[int(round(p*(nb-1)))]

        #store action in episode
        self.episode.append(action)
        
        #return the action, last char of 
        return int(action[-1:])
 

    def end_game(self, state, result, num_games):
        #update the value function of the state actions
        for stateAction in self.episode:
            if stateAction in self.strategy.table:
                val = self.strategy.table[stateAction]
                avg = val[0]
                n = val[1]+1
                avg -= avg/n
                avg += result/n
                val[0] = avg
                val[1] = n
            else:
                val = [result+0.0,1]
                self.strategy.table[stateAction] = val
        #print "Game number: ",num_games," result: ", result
        #clear list
        del self.episode[:]

get_ipython().magic('matplotlib')


do_plot = False
n_games = 2000


# create and add players
p1 = HumanPlayer("Human1")
p2 = RandomPlayer()
p3 = AlwaysLeftPlayer()
p4 = ValuePlayerSolution()
p5 = ExploreExploitPlayerSolution("ExploreExploitSol")
p6 = ValuePlayerDPJonas()
p7 = MonteCarloPlayer()


players = [p2, p6, p7]


num_players = len(players)

num_seeds = 1
aaa = np.transpose(np.matrix(np.zeros(num_seeds)))*np.zeros(num_players)
bbb = np.transpose(np.matrix(np.zeros(num_seeds)))*np.zeros(num_players)
    
for rs in range(num_seeds):

    np.random.seed(2)
    results = np.transpose(np.matrix(np.zeros(num_players)))*np.zeros(num_players)
    for i in range(num_players):
        results[i,i] = np.nan


    for j in range(num_players):
        for i in range(j):
            if (i != j): # not necessary
                p1t = players[i]
                p2t = players[j]

                Credits = np.zeros((2,n_games))
                tableETH = Table(p1t, p2t, rand_seed = rs)
                tableETH.output = False

                if(do_plot):
                    plt.figure()
                    plt.axis([0, n_games, -n_games, n_games])
                    lines = [plt.plot([], [], label=p.name)[0] for p in players]
                    plt.legend(loc='upper right', fontsize=10)
                    plt.show()     

                print "\nPlayer", p1t.name, " is playing against ", p2t.name, "..."

                for game_id in range(n_games):

                    if(game_id == n_games - 2):
                        tableETH.output = False 

                    #play game
                    tableETH.play_a_game(game_id)
                    if tableETH.output:
                        print "\rYou have finished %d games!" % game_id,
                        print "Board: " + str(tableETH.board.fields)
                        sys.stdout.flush()


                    if(do_plot):
                        # Update plot every 20 games
                        if ((game_id > 0) and (game_id % 20 == 0) or (game_id == n_games-1)):
                            Credits = np.transpose(tableETH.credits)
                            if (np.min(Credits) < plt.gca().get_ylim()[0]):
                                plt.gca().set_ylim([np.min(Credits)-10,100])
                            for k in range(2):
                                if(k == 0):
                                    ind = i
                                else:
                                    ind = j 
                                lines[ ind ].set_xdata(range(game_id+1))
                                lines[ ind ].set_ydata(Credits[k,0:(game_id+1)])
                            plt.draw()
                            time.sleep(0.1)

                print "Player", p1t.name, " has won ", tableETH.player1_won, " games."
                print "Player", p2t.name, " has won ", tableETH.player2_won, " games." 

                
                results[i,j] = tableETH.player1_won - tableETH.player2_won
                results[j,i] = tableETH.player2_won - tableETH.player1_won

    print [p.name for p in players]
                
    print results
            
    if 0:
        
        for game_id in range(10):
            #play game
            tableHuman = Table(s6, p1, rand_seed = 0)
            tableHuman.play_a_game(game_id)
    
    
    if tableETH.output:
        print np.transpose(tableETH.credits)
    for i in range(num_players):
        results[i,i] = 0
    aa = sum(np.transpose(results)>0)
    bb = sum(np.transpose(results))
    aaa[rs,:] = aa
    bbb[rs,:] = bb
    
    print '---------'
    print 'Total number of games the players have won:'
    print [p.name for p in players]
    print sum(np.transpose(results))
    print '---------'
    print '\n'
    # print s5.valuetable

print bbb
print aaa

import matplotlib.pyplot as plt
plt.axis([0, n_games, -n_games, n_games])
lines = [plt.plot([], [], label=p.name)[0] for p in [p1t, p2t]]
plt.legend(loc='upper left', fontsize=10)
plt.show()
Credits = np.transpose(tableETH.credits)
if (np.min(Credits) < plt.gca().get_ylim()[0]):
    plt.gca().set_ylim([np.min(Credits)-10,100])
for k in range(2):
    lines[k].set_xdata(range(game_id+1))
    lines[k].set_ydata(Credits[k,0:(game_id+1)])
plt.draw()

