from src.model import Py2Vec

json_file = "./data/blog_model.json"
model = Py2Vec(json_file)

WORD = "if"

model[WORD]

model.closest_words(WORD, 5)

target_vector = model.null_vector

model.closest_words(target_vector, 5)

target_vector = model['for'] - model['continue']

model.closest_words(target_vector, 5)

