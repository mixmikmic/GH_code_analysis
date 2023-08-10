from preprocessing import Preprocess
from go import GameState, BLACK, WHITE, EMPTY
import os, sgf
import numpy as np

# input SGF files
FILE_FOLDER = os.path.join(os.path.expanduser('~'), 'python', 'tutorial_files','go')
# output archive SGF files
SUCCEED_FOLDER = os.path.join(os.path.expanduser('~'), 'python', 'tutorial_files','succeed')
FAIL_FOLDER = os.path.join(os.path.expanduser('~'), 'python', 'tutorial_files','fail')
# output database
TRAIN_DATA = os.path.join(os.path.expanduser('~'), 'python', 'tutorial_data', 'zero', 'train_data')
TEST_DATA = os.path.join(os.path.expanduser('~'), 'python', 'tutorial_data', 'zero', 'test_data')

# Config this to indicate whether it's training or testing data
DATA_FOLDER = TRAIN_DATA

# BOARD_POSITION contains SGF symbol which represents each row (or column) of the board
# It can be used to convert between 0,1,2,3... and a,b,c,d...
# Symbol [tt] or [] represents PASS in SGF, therefore is omitted
BOARD_POSITION = 'abcdefghijklmnopqrs'

# Only 3 features are needed for AlphaGo Zero
# 0 - Player Stone, 1 - Opponent Stone, 3 - Current Player Color
DEFAULT_FEATURES = ["board", "color"]

# reverse the index of player/opponent
# 0,2,4,6... are player, 1,3,5,7... are opponent
OPPONENT_INDEX = [1,0,3,2,5,4,7,6,9,8,11,10,13,12]

from caffe2.python import core, utils
from caffe2.proto import caffe2_pb2

def write_db(db_type, db_name, base_name, features, labels, rewards):
    db = core.C.create_db(db_type, db_name, core.C.Mode.write)
    transaction = db.new_transaction()
    for i in range(features.shape[0]):
        feature_and_label = caffe2_pb2.TensorProtos()
        feature_and_label.protos.extend([
             utils.NumpyArrayToCaffe2Tensor(features[i]),
             utils.NumpyArrayToCaffe2Tensor(labels[i]),
             utils.NumpyArrayToCaffe2Tensor(rewards[i])
        ])
        transaction.put(
            '{}_{:0=3}'.format(base_name,i),
            feature_and_label.SerializeToString())
    # Close the transaction, and then close the db.
    del transaction
    del db

#%%capture output
p = Preprocess(DEFAULT_FEATURES)
for dirname, subDirList, fileList in os.walk(FILE_FOLDER):
    for filename in fileList:
        with open(os.path.join(dirname, filename)) as f:
            collection = sgf.parse(f.read())
            for game in collection:
                # Size of the Board should only be 19x19, Komi should be 7.5 according to Chinese rule
                if (game.nodes[0].properties['SZ'] == ['19']
#                    and game.nodes[0].properties['RU'] == ['Chinese']
#                    and game.nodes[0].properties['KM'] == ['7.50']
                   ):
                    try:
                        state = GameState() # Initialize GameState
                        features = np.empty(shape=(0,17,19,19), dtype=np.int8)
                        feature_history = np.zeros(shape=(1,17,19,19), dtype=np.int8)
                        labels = np.empty(shape=(0,), dtype=np.int32)
                        rewards = np.empty(shape=(0,), dtype=np.float32)
                        result = 'B' if game.nodes[0].properties['RE'][0:2] == ['B+'] else 'W'
                        for node in game.nodes[1:]: # Except nodes[0] for game properties
                            feature_current = p.state_to_tensor(state).astype(np.int8) # Player/Opponent/Empty/Color
                            feature_history = np.concatenate((feature_current[0:1,0:2], # Xt, Yt
                                                              feature_history[0:1,OPPONENT_INDEX],
                                                              feature_current[0:1,3:4]), # Color
                                                            axis=1)
                            if 'B' in node.properties and len(node.properties['B'][0]) == 2: # Black move
                                x = BOARD_POSITION.index(node.properties['B'][0][0])
                                y = BOARD_POSITION.index(node.properties['B'][0][1])
                                state.do_move(action=(x,y),color = BLACK)
                            elif 'W' in node.properties and len(node.properties['W'][0]) == 2: # White move
                                x = BOARD_POSITION.index(node.properties['W'][0][0])
                                y = BOARD_POSITION.index(node.properties['W'][0][1])
                                state.do_move(action=(x,y),color = WHITE)
                            reward = np.asarray([1.0 if result in node.properties else -1.0], dtype=np.float32)
                            features = np.append(features, feature_history, axis=0)
                            labels = np.append(labels, np.asarray([x * 19 + y], dtype=np.int32), axis=0)
                            rewards = np.append(rewards, reward, axis=0)
                        write_db(
                            db_type = 'leveldb',
                            db_name = DATA_FOLDER, # replace this with TRAIN_DATA or TEST_DATA if you want to separate the dataset
                            base_name = os.path.basename(filename),
                            features = features,
                            labels = labels,
                            rewards = rewards
                        )
                        os.rename(f.name,os.path.join(SUCCEED_FOLDER,filename)) # move the file to SUCCEED_FOLDER, so Preprocess can resume after interrupted
                        print('{} succeeded'.format(filename))
                    except Exception as e:
                        os.rename(f.name,os.path.join(FAIL_FOLDER,filename)) # move the file to FAIL_FOLDER, so Preprocess can resume after interrupted
                        print('{} failed dues to {}'.format(filename, e))
                else:
                    os.rename(f.name,os.path.join(FAIL_FOLDER,filename)) # move the file to FAIL_FOLDER, so Preprocess can resume after interrupted
                    print('{} unqualified dues to Size, Rule or Komi'.format(filename))

