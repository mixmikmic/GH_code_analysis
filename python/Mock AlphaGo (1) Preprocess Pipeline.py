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
DATA_FOLDER = os.path.join(os.path.expanduser('~'), 'python', 'tutorial_data','go')
TRAIN_DATA = os.path.join(DATA_FOLDER,'train_data')
TEST_DATA = os.path.join(DATA_FOLDER,'test_data')

# BOARD_POSITION contains SGF symbol which represents each row (or column) of the board
# It can be used to convert between 0,1,2,3... and a,b,c,d...
# Symbol [tt] or [] represents PASS in SGF, therefore is omitted
BOARD_POSITION = 'abcdefghijklmnopqrs'

DEFAULT_FEATURES = [
    "board", "ones", "turns_since", "liberties", "capture_size",
    "self_atari_size", "liberties_after", "ladder_capture", "ladder_escape",
    "sensibleness", "zeros"]

from caffe2.python import core, utils
from caffe2.proto import caffe2_pb2

def write_db(db_type, db_name, base_name, features, labels):
    db = core.C.create_db(db_type, db_name, core.C.Mode.write)
    transaction = db.new_transaction()
    for i in range(features.shape[0]):
        feature_and_label = caffe2_pb2.TensorProtos()
        feature_and_label.protos.extend([
             utils.NumpyArrayToCaffe2Tensor(features[i]),
             utils.NumpyArrayToCaffe2Tensor(labels[i])
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
                if game.nodes[0].properties['SZ'] == ['19']: # Size of the Board should only be 19x19
                    state = GameState() # Initialize GameState
                    features = np.empty(shape=(0,48,19,19), dtype=np.int8)
                    labels = np.empty(shape=(0,1), dtype=np.uint16)
                    try:
                        for node in game.nodes[1:]: # Except nodes[0] for game properties
                            features = np.append(features, p.state_to_tensor(state).astype(np.int8), axis=0)
                            if 'B' in node.properties and len(node.properties['B'][0]) == 2: # Black move
                                x = BOARD_POSITION.index(node.properties['B'][0][0])
                                y = BOARD_POSITION.index(node.properties['B'][0][1])
                                state.do_move(action=(x,y),color = BLACK)
                            elif 'W' in node.properties and len(node.properties['W'][0]) == 2: # White move
                                x = BOARD_POSITION.index(node.properties['W'][0][0])
                                y = BOARD_POSITION.index(node.properties['W'][0][1])
                                state.do_move(action=(x,y),color = WHITE)
                            labels = np.append(labels, np.asarray([[x * 19 + y]], dtype=np.uint16), axis=0)
                        write_db(
                            db_type = 'leveldb',
                            db_name = TRAIN_DATA, # replace this with TRAIN_DATA or TEST_DATA if you want to separate the dataset
                            base_name = os.path.basename(filename),
                            features = features,
                            labels = labels
                        )
                        os.rename(f.name,os.path.join(SUCCEED_FOLDER,filename)) # move the file to SUCCEED_FOLDER, so Preprocess can resume after interrupted
                        print('{} succeeded'.format(filename))
                    except Exception as e:
                        os.rename(f.name,os.path.join(FAIL_FOLDER,filename)) # move the file to FAIL_FOLDER, so Preprocess can resume after interrupted
                        print('{} failed dues to {}'.format(filename, e))



