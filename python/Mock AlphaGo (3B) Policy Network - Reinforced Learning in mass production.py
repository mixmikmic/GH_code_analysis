import os, numpy as np
from caffe2.python import core, model_helper, workspace, brew, utils
from caffe2.proto import caffe2_pb2
from sgfutil import BOARD_POSITION

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot

# how many games will be run in one minibatch
GAMES_BATCHES = 16 # [1,infinity) depends on your hardware
SEARCH_WIDE = 1600 # [1, infinity) for each step, run MCTS to obtain better distribution
# how many iterations for this tournament
TOURNAMENT_ITERS = 10000 # [1,infinity)

if workspace.has_gpu_support:
    device_opts = core.DeviceOption(caffe2_pb2.CUDA, workspace.GetDefaultGPUID())
    print('Running in GPU mode on default device {}'.format(workspace.GetDefaultGPUID()))
else :
    device_opts = core.DeviceOption(caffe2_pb2.CPU, 0)
    print('Running in CPU mode')

arg_scope = {"order": "NCHW"}

ROOT_FOLDER = os.path.join(os.path.expanduser('~'), 'python', 'tutorial_data','go','param') # folder stores the loss/accuracy log

### Config for primary player
PRIMARY_WORKSPACE = os.path.join(ROOT_FOLDER, 'primary')
PRIMARY_CONV_LEVEL = 4
PRIMARY_FILTERS = 128
PRIMARY_PRE_TRAINED_ITERS = 1
# before traning, where to load the params
PRIMARY_LOAD_FOLDER = os.path.join(ROOT_FOLDER, "RL-conv={}-k={}-iter={}"
                                   .format(PRIMARY_CONV_LEVEL,PRIMARY_FILTERS,PRIMARY_PRE_TRAINED_ITERS))
BASE_LR = -0.01 # (-0.003,0) The base Learning Rate; 0 to disable it.
NEGATIVE_BASE_LR = 0.0 # [BASE_LR,0] Dues to multi-class softmax, this param is usually smaller than BASE_LR; 0 to disable it.
TRAIN_BATCHES = 16 # how many samples will be trained within one mini-batch, depends on your hardware
# after training, where to store the params
PRIMARY_SAVE_FOLDER = os.path.join(ROOT_FOLDER, "RL-conv={}-k={}-iter={}"
                           .format(PRIMARY_CONV_LEVEL,PRIMARY_FILTERS,PRIMARY_PRE_TRAINED_ITERS+TOURNAMENT_ITERS))
if not os.path.exists(PRIMARY_SAVE_FOLDER):
    os.makedirs(PRIMARY_SAVE_FOLDER)

### Config for sparring partner
SPARR_WORKSPACE = os.path.join(ROOT_FOLDER, 'sparring')
SPARR_LOAD_FOLDER = os.path.join(ROOT_FOLDER, "conv={}-k={}-iter={}".format(4,128,1))

print('Training model from {} to {} iterations'.format(PRIMARY_PRE_TRAINED_ITERS,PRIMARY_PRE_TRAINED_ITERS+TOURNAMENT_ITERS))

from modeling import AddConvModel, AddTrainingOperators

import caffe2.python.predictor.predictor_exporter as pe

data = np.empty(shape=(TRAIN_BATCHES,48,19,19), dtype=np.float32)
label = np.empty(shape=(TRAIN_BATCHES,), dtype=np.int32)

workspace.SwitchWorkspace(PRIMARY_WORKSPACE, True)
# for learning from winner
with core.DeviceScope(device_opts):
    primary_train_model = model_helper.ModelHelper(name="primary_train_model", arg_scope=arg_scope, init_params=True)
    workspace.FeedBlob("data", data)
    predict = AddConvModel(primary_train_model, "data", conv_level=PRIMARY_CONV_LEVEL, filters=PRIMARY_FILTERS)
    workspace.FeedBlob("label", label)
    AddTrainingOperators(primary_train_model, predict, "label", None, base_lr=BASE_LR)
    workspace.RunNetOnce(primary_train_model.param_init_net)
    workspace.CreateNet(primary_train_model.net, overwrite=True)
# for learning from negative examples
with core.DeviceScope(device_opts):
    primary_train_neg_model = model_helper.ModelHelper(name="primary_train_neg_model", arg_scope=arg_scope, init_params=True)
    workspace.FeedBlob("data", data)
    predict = AddConvModel(primary_train_neg_model, "data", conv_level=PRIMARY_CONV_LEVEL, filters=PRIMARY_FILTERS)
    ONES = primary_train_neg_model.ConstantFill([], "ONES", shape=[TRAIN_BATCHES,361], value=1.0)
    negative = primary_train_neg_model.Sub([ONES, predict], 'negative')
    workspace.FeedBlob("label", label)
    AddTrainingOperators(primary_train_neg_model, negative, None, expect, base_lr=NEGATIVE_BASE_LR)
    workspace.RunNetOnce(primary_train_neg_model.param_init_net)
    workspace.CreateNet(primary_train_neg_model.net, overwrite=True)
    
    primary_predict_net = pe.prepare_prediction_net(os.path.join(PRIMARY_LOAD_FOLDER, "policy_model.minidb"), "minidb")

def LearnFromWinningGames(history, winner, mini_batch=TRAIN_BATCHES):
    data = np.empty(shape=(mini_batch,48,19,19), dtype=np.float32)
    label = np.empty(shape=(mini_batch,), dtype=np.int32)
    #iter = 0
    k = 0
    for i in range(len(winner)):
        #print('Learning {} steps in {} of {} games'.format(iter * TRAIN_BATCHES, i, GAMES_BATCHES))
        for step in history[i]:
            if (step[0] == 'B' and winner[i] == 'B+') or (step[0] == 'W' and winner[i] == 'W+'):
                data[k] = step[2]
                label[k] = step[1]
                k += 1
                #iter += 1
                if k == mini_batch:
                    k = 0
                    workspace.SwitchWorkspace(PRIMARY_WORKSPACE)
                    with core.DeviceScope(device_opts):
                        workspace.FeedBlob("data", data)
                        workspace.FeedBlob("label", label)
                        workspace.RunNet(primary_train_model.net)

def LearnFromLosingGames(history, winner, mini_batch=TRAIN_BATCHES):
    data = np.empty(shape=(mini_batch,48,19,19), dtype=np.float32)
    label = np.empty(shape=(mini_batch,), dtype=np.int32)
    #iter = 0
    k = 0
    for i in range(len(winner)):
        #print('Learning {} steps in {} of {} games'.format(iter * TRAIN_BATCHES, i, GAMES_BATCHES))
        for step in history[i]:
            if (step[0] == 'B' and winner[i] == 'W+') or (step[0] == 'W' and winner[i] == 'B+'):
                data[k] = step[2]
                label[k] = step[1]
                k += 1
                #iter += 1
                if k == mini_batch:
                    k = 0
                    workspace.SwitchWorkspace(PRIMARY_WORKSPACE)
                    with core.DeviceScope(device_opts):
                        workspace.FeedBlob("data", data)
                        workspace.FeedBlob("label", label)
                        workspace.RunNet(primary_train_neg_model.net)

from go import GameState, BLACK, WHITE, EMPTY, PASS
from preprocessing import Preprocess
from game import DEFAULT_FEATURES
from datetime import datetime
from sgfutil import GetWinner, WriteBackSGF
import sgf

np.random.seed(datetime.now().microsecond)

# construct the model to be exported
pe_meta = pe.PredictorExportMeta(
    predict_net=primary_predict_net.Proto(),
    parameters=[str(b) for b in primary_train_model.params],
    inputs=["data"],
    outputs=["softmax"],
)

for tournament in range(PRIMARY_PRE_TRAINED_ITERS, PRIMARY_PRE_TRAINED_ITERS+TOURNAMENT_ITERS):
    # Every 500 tournament, copy current player to opponent. i.e. checkpoint
    if tournament > 0 and tournament % 20 == 0:
        pe.save_to_db("minidb", os.path.join(PRIMARY_SAVE_FOLDER, "policy_model.minidb"), pe_meta)
        print('Checkpoint saved to {}'.format(PRIMARY_SAVE_FOLDER))
        pe.save_to_db("minidb", os.path.join(SPARR_LOAD_FOLDER, "policy_model_RL_{}.minidb".format(PRIMARY_PRE_TRAINED_ITERS+tournament)), pe_meta)
        print('Checkpoint saved to {}'.format(SPARR_LOAD_FOLDER))
    
    # Randomly change color of player
    PRIMARY_PLAYER = np.random.choice(['B','W'])
    if PRIMARY_PLAYER == 'B':
        SPARRING_PLAYER = 'W'
    else:
        SPARRING_PLAYER = 'B'
    
    # Randomly pickup sparring partner
    workspace.SwitchWorkspace(SPARR_WORKSPACE, True)
    sparring_param_file = np.random.choice(os.listdir(SPARR_LOAD_FOLDER))
    with core.DeviceScope(device_opts):
        sparring_predict_net = pe.prepare_prediction_net(os.path.join(SPARR_LOAD_FOLDER, sparring_param_file), "minidb")
    print('Tournament {} Primary({}) vs Sparring({}|{}) started @{}'
          .format(tournament, PRIMARY_PLAYER, SPARRING_PLAYER, sparring_param_file, datetime.now()))

    
    # Initialize game board and game state
    game_state = [ GameState() for i in range(GAMES_BATCHES) ]
    game_result = [0] * GAMES_BATCHES # 0 - Not Ended; BLACK - Black Wins; WHITE - White Wins
    p = Preprocess(DEFAULT_FEATURES) # Singleton
    history = [ [] for i in range(GAMES_BATCHES) ] # history[n][step] stores tuple of (player, x, y, board[n])
    board = None # The preprocessed board with shape Nx48x19x19
    
    # for each step in all games
    for step in range(0,722):
        
        # Preprocess the board
        board = np.concatenate([p.state_to_tensor(game_state[i]).astype(np.float32) for i in range(GAMES_BATCHES)])

        if step % 2 == 0:
            current_player = BLACK
            current_color = 'B'
        else:
            current_player = WHITE
            current_color = 'W'

        if step % 2 == (PRIMARY_PLAYER == 'W'): # if step %2 == 0 and Primary is Black, or vice versa.
            # primary player make move
            workspace.SwitchWorkspace(PRIMARY_WORKSPACE)
            with core.DeviceScope(device_opts):
                workspace.FeedBlob('data', board)
                workspace.RunNet(primary_predict_net)
        else:
            # sparring partner make move
            workspace.SwitchWorkspace(SPARR_WORKSPACE)
            with core.DeviceScope(device_opts):
                workspace.FeedBlob('data', board)
                workspace.RunNet(sparring_predict_net)

        predict = workspace.FetchBlob('softmax') # [0.01, 0.02, ...] in shape (N,361)

        for i in range(GAMES_BATCHES):
            if game_result[i]: # game end
                continue
            else: # game not end
                legal_moves = [ x*19+y for (x,y) in game_state[i].get_legal_moves(include_eyes=False)] # [59, 72, ...] in 1D
                if len(legal_moves) > 0: # at least 1 legal move
                    probabilities = predict[i][legal_moves] # [0.02, 0.01, ...]
                    # use numpy.random.choice to randomize the step,
                    # otherwise use np.argmax to get best choice
                    # current_choice = legal_moves[np.argmax(probabilities)]
                    if np.sum(probabilities) > 0:
                        current_choice = np.random.choice(legal_moves, 1, p=probabilities/np.sum(probabilities))[0]
                    else:
                        current_choice = np.random.choice(legal_moves, 1)[0]
                    (x, y) = (current_choice/19, current_choice%19)
                    history[i].append((current_color, current_choice, board[i]))
                    game_state[i].do_move(action = (x, y), color = current_player) # End of Game?
                    #print('game({}) step({}) {} move({},{})'.format(i, step, current_color, x, y))
                else:
                    game_state[i].do_move(action = PASS, color = current_player)
                    #print('game({}) step({}) {} PASS'.format(i, step, current_color))
                    game_result[i] = game_state[i].is_end_of_game

        if np.all(game_result):
            break
    
    # Get the winner
    winner = [ GetWinner(game_state[i]) for i in range(GAMES_BATCHES) ] # B+, W+, T
    print('Tournament {} Finished with Primary({}) {}:{} Sparring({}) @{}'.
          format(tournament, PRIMARY_PLAYER, sum(np.char.count(winner, PRIMARY_PLAYER)),
                 sum(np.char.count(winner, SPARRING_PLAYER)), SPARRING_PLAYER, datetime.now()))
    
    # Save the games(optional)
    for i in range(GAMES_BATCHES):
        filename = os.path.join(
            os.path.expanduser('~'), 'python', 'tutorial_files','selfplay',
            '({}_{}_{})vs({})_{}_{}_{}'.format(PRIMARY_CONV_LEVEL, PRIMARY_FILTERS, PRIMARY_PRE_TRAINED_ITERS+tournament,
                                            sparring_param_file, i, winner[i],
                                            datetime.now().strftime("%Y-%m-%dT%H:%M:%S%Z")))
        WriteBackSGF(winner, history[i], filename)
    
    # After each tournament, learn from the winner
    if BASE_LR != 0:
        LearnFromWinningGames(history, winner, mini_batch=TRAIN_BATCHES)
    
    # And learn from negative examples
    if NEGATIVE_BASE_LR != 0:
        LearnFromLosingGames(history, winner, mini_batch=TRAIN_BATCHES)

if TOURNAMENT_ITERS>0 :
    pe.save_to_db("minidb", os.path.join(PRIMARY_SAVE_FOLDER, "policy_model.minidb"), pe_meta)
    print('Results saved to {}'.format(PRIMARY_SAVE_FOLDER))
    pe.save_to_db("minidb", os.path.join(SPARR_LOAD_FOLDER, "policy_model_RL_{}.minidb".format(PRIMARY_PRE_TRAINED_ITERS+TOURNAMENT_ITERS)), pe_meta)
    print('Results saved to {}'.format(SPARR_LOAD_FOLDER))

