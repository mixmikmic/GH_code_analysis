get_ipython().magic('pylab inline')

import matplotlib.cm as cm
import matplotlib.patches as patches

from collections import defaultdict
from six.moves import zip_longest
import numpy as np
import random, sys, pickle, os, time

from IPython.core.debugger import set_trace
#set_trace()

# For some reason Notebook doesn't like this...
#from builtins import input
# Hacky py3 backwards compatibility
try:
    input = raw_input
except NameError:
    pass

class BoardState(object):
    # 0 1 2
    # 3 4 5
    # 6 7 8
    winning_spots = np.array([
        [0, 1, 2], [3, 4, 5], [6, 7, 8], # Horizontal
        [0, 3, 6], [1, 4, 7], [2, 5, 8], # Vertical
        [0, 4, 8], [2, 4, 6]             # Diagonal
        ])
    
    board_format = '\n'.join([
        ' {} | {} | {} ',
        '---+---+---',
        ' {} | {} | {} ',
        '---+---+---',
        ' {} | {} | {} ',
        ])

    def __init__(self, prev=None, action=None):
        if prev is not None:
            self.marks = prev.marks.copy()
            self.marks[action] = prev.active_player
            self.active_player = 'X' if prev.active_player == 'O' else 'O'
        else:
            self.active_player = 'X'
            self.marks = np.array(['_']*9)

    def __repr__(self):
        return ''.join(self.marks) + ',' + self.active_player

    def __str__(self):
        return BoardState.board_format.format(*self.marks)

    def __eq__(self, other):
        return isinstance(other, self.__class__)             and np.array_equal(self.marks, other.marks)             and self.active_player == other.active_player

    def __hash__(self):
        return hash(repr(self))

    @staticmethod
    def from_repr(s):
        out = BoardState()
        out.active_player = s[-1]
        out.marks = np.array(list(s[:-2]))
        return out

    def render(self):
        print(self.__str__())

    # (action:int) -> (next_state:BoardState, reward:float, done:bool)
    def step(self, action:int):
        # Construct next_state by applying action to the current board
        # (placing 0 or X on square# `action` depending on whose turn it is).        
        next_state = BoardState(self, action)

        # Score the resulting board by performing a static evaluation:
        #    -1  if the action is an illegal move (attempting to mark a nonempty cell), else
        #    +1  if the action wins the game, else
        #     0  if it completes the board without winning (tie)
        reward = -1.0 if self.marks[action] != '_'             else +1.0 if next_state.check_win(self.active_player)             else  0.0

        done = next_state.is_full()  or  reward != 0.0
        
        return (next_state, reward, done)

    def check_win(self, player):
        slices = self.marks[BoardState.winning_spots]
        return (slices == player).all(axis=1).any()

    def is_full(self):
        return (self.marks != '_').all()
    
    def draw(self, agent):
        fig = figure(figsize=[3,3])
        ax = fig.add_subplot(111)

        def draw_cell(pos, mark, val):
            y, x = divmod(pos, 3)
            
            # If the game has been won, give cells in winning line(s) a red background
            slices = self.marks[self.winning_spots]
            O = (slices == 'O').all(axis=1)
            X = (slices == 'X').all(axis=1)
            winningSliceIndices = np.append( O.nonzero(), X.nonzero() )
            winningSquares = np.unique( self.winning_spots[ winningSliceIndices ] )           
            
            if pos in winningSquares:
                ax.add_patch(patches.Rectangle((x,y), 1, 1, ec='none', fc='red'))
                            
            if mark == 'X':
                ax.plot([x+.2, x+.8], [y+.8, y+.2], 'k', lw=2.0)
                ax.plot([x+.2, x+.8], [y+.2, y+.8], 'k', lw=2.0)
            elif mark == 'O':
                ax.add_patch(patches.Circle((x+.5,y+.5), .35, ec='k', fc='none', lw=2.0))
            else:
                # Colour empty squares according to predicted value
                color = cm.viridis((val+1)/2.)
                ax.add_patch(patches.Rectangle((x,y), 1, 1, ec='none', fc=color))
                ax.text(x+.5 , y+.5, '%.2f'%val    , ha='center', va='center') 
                ax.text(x+.08, y+.12,  '%d'%(pos+1), ha='center', va='center') 

        for i in range(9):
            draw_cell(i, self.marks[i], agent.Q_read(i,self))

        ax.set_position([0,0,1,1])
        ax.set_axis_off()

        ax.set_xlim(0,3)
        ax.set_ylim(3,0)

        for x in range(1,3):
            ax.plot([x, x], [0,3], 'k', lw=2.0)
            ax.plot([0,3], [x, x], 'k', lw=2.0)
        show()

# The agent is a computer player.
# It builds a Q_table, e.g.
#
#       Q_tables[4][('XO_X___OX','O')] = +1.0
#
# This says that placing a O at location 4 (i.e. center) on the board:
#       X O -
#       X - -
#       - O X
# ... will evaluate a score (or Quality Q) of +1.0 (i.e. a win)
#
# Use Q_read & Q_writeOrUpdate to access.


class TabularAgent(object):
    def __init__(self, num_actions, alpha=0.75, gamma=1.0, epsilon=1.0, default_Q=0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.default_Q = default_Q
        self.num_actions = num_actions  # should be 9 (9 squares so 9 possible actions)
        self.Q_tables = [{} for _ in range(self.num_actions)]
        
    # For a given board, return the action# that predicts the highest Q
    def max_action(self, state):
        # type: (BoardState) -> int
        predictions = [self.Q_read(ndx, state) for ndx in range(self.num_actions)]
        return np.argmax(predictions)

    # Choose a random action (0-8) with probability epsilon, 
    # or the optimal action with probability 1-epsilon
    def choose_action(self, state):
        # type: (BoardState) -> int
        if random.random() > self.epsilon:
            return random.choice(range(self.num_actions))
        return self.max_action(state)

    # Get Q-value for a particular action on a given board(-state)
    def Q_read(self, nAction, state):
        # type: (int, BoardState) -> float
        return self.Q_tables[nAction].get(state, self.default_Q)

    # Update Q-value for a particular state+action pair
    # (creating a new entry if necessary)
    def Q_writeOrUpdate(self, nAction, state, new_Q):
        # type: (int, BoardState, float) -> None
        buf = self.Q_tables[nAction]
        if state in buf:
            buf[state] = (1-self.alpha)*buf[state] + self.alpha*new_Q
        else:
            buf[state] = new_Q

    def train(self, history):
        raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - 
# train: (history:[(action:int, reward:float, state:BoardState)]) -> (none)
#     Use history (from the perspective of one player over the duration of one game) to update Q-table.
#     For each (state, action) pair, replace the entry (reward) in the Q-table with a one-step lookahead prediction.
#     i.e. 
#         - Apply the action to the state resulting in a new state s'
#         - from s' determine Q of the best action
#         - Q(state, action) <- reward(state, action) + gamma * Q(s', best action)

class MonteCarloAgent(TabularAgent):
    def train(self, history):
        if len(history) == 0:
            return
        prev_action, return_, _ = history[-1]
        for (action, reward, state) in reversed(history[:-1]):
            self.Q_writeOrUpdate(prev_action, state, return_)

            prev_action = action
            if reward is not None:
                return_ += reward
                
class TemporalDifferenceAgent(TabularAgent):
    def new_val(self, history, ndx):
        raise NotImplementedError()

    def train(self, history):
        if len(history) == 1:
            return
        for i in range(len(history)-2):
            (_, _, state), (action, _, _) = history[i:i+2]
            self.Q_writeOrUpdate(action, state, self.new_val(history, i))
            
        (_, _, state), (action, reward, _) = history[-2:]
        
        self.Q_writeOrUpdate(action, state, reward)

# - - - - - - - - - - - - - - - - - - - - - 
# One-step Q-table lookahead reward prediction

class QLearningAgent(TemporalDifferenceAgent):
    def new_val(self, history, ndx):
        (_, _, state), (action, reward, next_state)                      = history[ndx:ndx+2]
        next_action = self.max_action(next_state)
        return reward + self.gamma * self.Q_read(next_action, next_state)

class SarsaAgent(TemporalDifferenceAgent):
    def new_val(self, history, ndx):
        (_, _, state), (action, reward, next_state), (next_action, _, _) = history[ndx:ndx+3]
        return reward + self.gamma * self.Q_read(next_action, next_state)

# TRAIN (Computer vs. Computer)
episodes = 10000
fname = 'tictac.txt'

def progressbar(callback, iters, refresh_rate=2.0):
    prev_clock = time.time()
    start_clock = prev_clock

    for i in range(iters):
        callback(i)
        curr_clock = time.time()
        if (curr_clock-prev_clock)*refresh_rate >= 1:
            sys.stdout.write('\r[ %s / %s ]' % (i, iters))
            sys.stdout.flush()
            prev_clock = curr_clock

    clearstr = ' '*len('[ %s / %s ]' % (iters, iters))
    sys.stdout.write('\r%s\r' % clearstr)
    sys.stdout.flush()

    return time.time() - start_clock

# Assumes zero-sum, two-player, sequential-turn game
def train_episode(agent, state=None):
    if state is None:
        # Start at a random previously encountered state
        keys = list( agent.Q_tables[random.randint(0,8)] )  # list of keys for Q-table dict
        if len(keys) > 0:
            state = random.choice(keys)
        else:
            state = BoardState()

    # Play out a game, recording each (action, reward, state) tuple.
    first_player = state.active_player
    history = [(None, None, state)]

    while True:
        action = agent.choose_action(state)
        state, reward, done = state.step(action)
        history.append((action, reward, state))
        if done:
            break

    # Split history into a separate history for each player.
    #    history stores things like [(None, None, s1), (p1a1, p1r1, s2), (p2a1, p2r1, s3), (p1a2, p1r2, s4), ...]
    #    player_history transforms that to [(None, None, s1), (p1a1, p1r1-p2r1, s3), (p1a2, p1r2-p2r2, s5), ...]
    #    You subtract the reward given to the other player because of the assumption of it being a zero-sum game.
    #    (Think: relative reward)
    def player_history(history):
        # e.g.  grouped('ABCDEFG', 3, 'x') --> 'ABC' 'DEF' 'Gxx'
        def grouped(iterable, n, fillvalue=None):
            "Collect data into fixed-length chunks or blocks"
            # https://docs.python.org/2/library/itertools.html#recipes
            args = [iter(iterable)] * n
            return zip_longest(fillvalue=fillvalue, *args)

        out = [(None, None, history[0][2])]
        for (action, reward, state), (_, other_reward, other_state)                                 in grouped(history[1:], 2, (None,)*3):
            if other_reward is None:
                out.append((action, reward, state))
            else:
                out.append((action, reward-other_reward, other_state))
        return out
    
    first_history = player_history(history)
    second_history = player_history(history[1:])
    
    # Update Q-tables
    agent.train(first_history)
    agent.train(second_history)

    
if os.path.isfile(fname):
    print('Loading agent from %s...' % fname)
    agent = pickle.load(open(fname, 'rb'))
else:
    agent = QLearningAgent(num_actions=9, epsilon=0.8, default_Q=2)

#init_state = BoardState() # Always start from actual inital state
init_state = None # Random restarts

print('Training for %d episodes...' % episodes)
progressbar(lambda x: train_episode(agent, init_state), episodes)

print('Saving agent to %s...' % fname)
pickle.dump(agent, open(fname, 'wb'))

def play_vs_human(agent, state=None):
    if state is None:
        state = BoardState()    
    state.draw(agent)
    
    # Flip a coin for who goes first
    compToMove = random.random() > 0.5
    
    while True:
        if compToMove:
            state, reward, done = state.step(agent.choose_action(state))
        else:
            move = int(input('Choose your move [1-9]: ')) - 1
            state, reward, done = state.step(move)
            
        state.draw(agent)
        
        if done:
            if compToMove:
                s = 'Tie!' if reward == 0  else   'I win!' if reward > 0  else   'I lose!'
            else:
                s = 'Tie!' if reward == 0  else 'You win!' if reward > 0  else 'You lose!'
                
            input(s + "\nPress Enter to play again...")
            return
            
        compToMove = not compToMove

# PLAY
fname = 'tictac.txt'

agent = pickle.load(open(fname, 'rb'))
agent.epsilon = 0.99
try:
    while True:
        play_vs_human(agent)
except Exception as e:
    print(e)

