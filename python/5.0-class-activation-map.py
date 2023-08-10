get_ipython().run_cell_magic('capture', '', 'import matplotlib.pyplot as plt\nimport numpy as np\n\n%matplotlib inline\n\nfrom src.visualization.heatmap import generate_heatmap\nfrom src.visualization.heatmap import generate_wordcloud\nfrom keras.preprocessing.text import text_to_word_sequence')

positive_review = ("Simple, Durable, Fun game for all ages."
                   "This is an AWESOME game!"
                   "Almost everyone know tic-tac-toe so "
                   "it is EASY to learn and quick to play.")

heatmap = generate_heatmap(positive_review)
words = text_to_word_sequence(positive_review)

image = np.ones((1, 10)) * heatmap.reshape(-1, 1)

plt.figure(figsize=(15, 10))
plt.imshow(image)
plt.xticks([])
plt.yticks(range(len(words)), words)
plt.show()

wordcloud = generate_wordcloud(words, heatmap)

plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

