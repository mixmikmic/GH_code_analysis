import pandas as pd
df = pd.read_csv('./data/java_questions_including_topics.csv')

df.Topic.value_counts()

df[df.Topic == 13.0]

print(df[df['Topic'].isnull()])



