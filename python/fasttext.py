import fasttext
from gensim.models.wrappers.fasttext import FastText

fasttext.cbow('./data/test4.txt', './model/ft.vec')
model = FastText.load_fasttext_format('./model/ft.vec')

model.most_similar(positive=['마법'])

test_word = '마법학교와'
print("{0} 존재함".format(test_word)) if test_word in model.wv.index2word else print("{0} 존재않함".format(test_word))
model.most_similar(positive=[test_word])

