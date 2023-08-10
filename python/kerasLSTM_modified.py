import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.layers.wrappers import TimeDistributed
import random
import pickle

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

all_recipes = pickle.load(open('dataset/cleaned_recipes.p', 'rb'))
char2id = pickle.load(open('dataset/char2id.p', 'rb'))
id2char = pickle.load(open('dataset/id2char.p', 'rb'))
max_recipe_length = 500
max_name_length = 50
maxSequenceLength = 501
maxNameSequenceLength = 51

def booze_permuter(recipes):
    while True:
        for rec in recipes:
            ings = rec['ingredients']
            i = 0
            while i < len(ings) and is_number(ings[i][0]):
                i += 1

            ing_list = ings[:i]
            garn_list = ings[i:]

            random.shuffle(ing_list)
            random.shuffle(garn_list)

            ing_list.extend(garn_list)
            yield '\n'.join(ing_list)
        
def training_set_generator(num_recipes):
    recipe_generator = booze_permuter(all_recipes)
    while True:
        recipe_list = []
        while len(recipe_list) < num_recipes:
            recipe_list.append(next(recipe_generator))
            
        maxSequenceLength = max_recipe_length + 1
        inputChars = np.zeros((len(recipe_list), maxSequenceLength, len(char2id)), dtype=np.bool)
        nextChars = np.zeros((len(recipe_list), maxSequenceLength, len(char2id)), dtype=np.bool)

        for i in range(0, len(recipe_list)):
            inputChars[i, 0, char2id['S']] = 1
            nextChars[i, 0, char2id[recipe_list[i][0]]] = 1
            for j in range(1, maxSequenceLength):
                if j < len(recipe_list[i]) + 1:
                    inputChars[i, j, char2id[recipe_list[i][j - 1]]] = 1
                    if j < len(recipe_list[i]):
                        nextChars[i, j, char2id[recipe_list[i][j]]] = 1
                    else:
                        nextChars[i, j, char2id['E']] = 1
                else:
                    inputChars[i, j, char2id['E']] = 1
                    nextChars[i, j, char2id['E']] = 1
        
        yield (inputChars, nextChars)



def name_booze_permuter(recipes):
    while True:
        for rec in recipes:
            if len(rec['name']) <= max_name_length:
                ings = rec['ingredients']
                i = 0
                while i < len(ings) and is_number(ings[i][0]):
                    i += 1

                ing_list = ings[:i]
                garn_list = ings[i:]

                random.shuffle(ing_list)
                random.shuffle(garn_list)

                ing_list.extend(garn_list)
                yield ('\n'.join(ing_list), rec['name'].lower())

def name_training_set_generator(num_recipes):
    recipe_generator = name_booze_permuter(all_recipes)
    while True:
        recipe_list = []
        name_list = []
        while len(recipe_list) < num_recipes:
            example = next(recipe_generator)
            recipe_list.append(example[0])
            name_list.append(example[1])
            
        maxSequenceLength = max_recipe_length + 1
        maxNameSequenceLength = max_name_length + 1

        recipeChars = np.zeros((num_recipes, maxSequenceLength, len(char2id)), dtype=np.bool)
        inNameChars = np.zeros((num_recipes, maxNameSequenceLength, len(char2id)), dtype=np.bool)
        nextNameChars = np.zeros((num_recipes, maxNameSequenceLength, len(char2id)), dtype=np.bool)

        for i in range(0, num_recipes):
            recipeChars[i, 0, char2id['S']] = 1
            nextNameChars[i, 0, char2id[name_list[i][0]]] = 1
            for j in range(1, maxSequenceLength):
                if j < len(recipe_list[i]) + 1:
                    recipeChars[i, j, char2id[recipe_list[i][j - 1]]] = 1
                else:
                    recipeChars[i, j, char2id['E']] = 1
            inNameChars[i, 0, char2id['S']] = 1
            for j in range(1, maxNameSequenceLength):
                if j <= len(name_list[i]):
                    if name_list[i][j - 1] not in char2id:
                        inNameChars[i, j, char2id[' ']] = 1
                    else:
                        inNameChars[i, j, char2id[name_list[i][j - 1]]] = 1
                        
                    if j < len(name_list[i]):
                        if name_list[i][j] not in char2id:
                            nextNameChars[i, j, char2id[' ']] = 1
                        else:
                            nextNameChars[i, j, char2id[name_list[i][j]]] = 1
                    else:
                        nextNameChars[i, j, char2id['E']] = 1
                        
                else:
                    inNameChars[i, j, char2id['E']] = 1
                    nextNameChars[i, j, char2id['E']] = 1
        yield ([recipeChars, inNameChars], nextNameChars)

        
gen = name_training_set_generator(128)
inputs, nextChars = next(gen)

print(len(inputs))
print(len(inputs[0][0]))
print(len(inputs[1][0]))

# Compute a char2id and id2char vocabulary.
# test_set = get_new_recipes()
# charIndex = 0
# for recipe in test_set:
#     for char in recipe:
#         if char not in char2id:
#             char2id[char] = charIndex
#             id2char[charIndex] = char
#             charIndex += 1

# # Add a special starting and ending character to the dictionary.
# char2id['S'] = charIndex; id2char[charIndex] = 'S'  # Special sentence start character.
# char2id['E'] = charIndex + 1; id2char[charIndex + 1] = 'E'  # Special sentence ending character.
# pickle.dump(char2id, open('char2id_2.p', 'wb'), protocol = 2)
# pickle.dump(id2char, open('id2char_2.p', 'wb'), protocol = 2)
# pickle.dump(all_recipes, open('cleaned_recipes_2.p', 'wb'), protocol = 2)



# print("input:")
# print(inputChars.shape)  # Print the size of the inputCharacters tensor.
# print("output:")
# print(nextChars.shape)  # Print the size of the nextCharacters tensor.
# print("char2id:")
# print(char2id)  # Print the character to ids mapping.

inputChars = inputs[1]
trainCaption = inputChars[25, :, :]  # Pick some caption
labelCaption = nextChars[25, :, :]  # Pick what we are trying to predict.

def printCaption(sampleCaption):
    charIds = np.zeros(sampleCaption.shape[0])
    for (idx, elem) in enumerate(sampleCaption):
        charIds[idx] = np.nonzero(elem)[0].squeeze()
    print(np.array([id2char[x] for x in charIds]))

printCaption(trainCaption)
printCaption(labelCaption)

print('Building training model...')
hiddenStateSize = 128
hiddenLayerSize = 128
model = Sequential()
# The output of the LSTM layer are the hidden states of the LSTM for every time step. 
model.add(LSTM(hiddenStateSize, return_sequences = True, input_shape=(maxSequenceLength, len(char2id))))
# Two things to notice here:
# 1. The Dense Layer is equivalent to nn.Linear(hiddenStateSize, hiddenLayerSize) in Torch.
#    In Keras, we often do not need to specify the input size of the layer because it gets inferred for us.
# 2. TimeDistributed applies the linear transformation from the Dense layer to every time step
#    of the output of the sequence produced by the LSTM.
model.add(TimeDistributed(Dense(hiddenLayerSize)))
model.add(TimeDistributed(Activation('relu'))) 
model.add(TimeDistributed(Dense(len(char2id))))  # Add another dense layer with the desired output size.
model.add(TimeDistributed(Activation('softmax')))
# We also specify here the optimization we will use, in this case we use RMSprop with learning rate 0.001.
# RMSprop is commonly used for RNNs instead of regular SGD.
# See this blog for info on RMSprop (http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop)
# categorical_crossentropy is the same loss used for classification problems using softmax. (nn.ClassNLLCriterion)
model.compile(loss='categorical_crossentropy', optimizer = RMSprop(lr=0.001))

print(model.summary()) # Convenient function to see details about the network model.

# Test a simple prediction on a batch for this model.
print("Sample input Batch size:"),
print(inputChars[0:32, :, :].shape)
print("Sample input Batch labels (nextChars):"),
print(nextChars[0:32, :, :].shape)
outputs = model.predict(inputChars[0:32, :, :])
print("Output Sequence size:"),
print(outputs.shape)

model.fit(inputChars, nextChars, batch_size = 128, nb_epoch = 10)

# model.save_weights('cocktail_weights.h5')
model.load_weights('cocktail_weights.h5')

print("Sample input Batch size:"),
print(inputChars[0:32, :, :].shape)
print("Sample input Batch labels (nextChars):"),
print(nextChars[0:32, :, :].shape)
outputs = model.predict(inputChars[0:32, :, :])
print("Output Sequence size:"),
print(outputs.shape)
# Test a simple prediction on a batch for this model.
captionId = 132

inputCaption = inputChars[captionId:captionId+1, :, :]
outputs = model.predict(inputCaption)
# printCaption(inputCaption[0])
print(''.join([id2char[x.argmax()] for x in outputs[0, :, :]]))

# The only difference with the "training model" is that here the input sequence has 
# a length of one because we will predict character by character.
print('Building Inference model...')
inference_model = Sequential()
# Two differences here.
# 1. The inference model only takes one sample in the batch, and it always has sequence length 1.
# 2. The inference model is stateful, meaning it inputs the output hidden state ("its history state")
#    to the next batch input.
inference_model.add(LSTM(hiddenStateSize, batch_input_shape=(1, 1, len(char2id)), stateful = True))
# Since the above LSTM does not output sequences, we don't need TimeDistributed anymore.
inference_model.add(Dense(hiddenLayerSize))
inference_model.add(Activation('relu'))
inference_model.add(Dense(len(char2id)))
inference_model.add(Activation('softmax'))

# Copy the weights of the trained network. Both should have the same exact number of parameters (why?).
# inference_model.set_weights(model.get_weights())
inference_model.load_weights('gpu_weights.h5')

# Given the start Character 'S' (one-hot encoded), predict the next most likely character.
startChar = np.zeros((1, 1, len(char2id)))
startChar[0, 0, char2id['S']] = 1
nextCharProbabilities = inference_model.predict(startChar)

# print the most probable character that goes next.
print(id2char[nextCharProbabilities.argmax()])

charProbs = [(id2char[i], p) for i, p in enumerate(nextCharProbabilities.squeeze())]
charProbs.sort(key=lambda i: i[1], reverse=True)
charProbs[:10]

print(id2char)
print(char2id)



for i in range(0, 10):
    inference_model.reset_states()  # This makes sure the initial hidden state is cleared every time.
    startChar = np.zeros((1, 1, len(char2id)))
    startChar[0, 0, char2id['S']] = 1
    end = False
    sent = ""
    for i in range(0, maxSequenceLength):
        nextCharProbs = inference_model.predict(startChar)

        # In theory I should be able to input nextCharProbs to np.random.multinomial.
        nextCharProbs = np.asarray(nextCharProbs).astype('float64') # Weird type cast issues if not doing this.
        nextCharProbs = nextCharProbs / nextCharProbs.sum()  # Re-normalize for float64 to make exactly 1.0.

        nextCharId = np.random.multinomial(1, nextCharProbs.squeeze(), 1).argmax()
        if id2char[nextCharId] == 'E':
            if not end:
                print("~~~~~")
            end = True
        else:
            sent = sent + id2char[nextCharId] # The comma at the end avoids printing a return line character.
        startChar.fill(0)
        startChar[0, 0, nextCharId] = 1
    print(sent)

