text = [
    "Because I could not stop for Death -",
    "He kindly stopped for me -",
    "The Carriage held but just Ourselves -",
    "and Immortality"
]

text

import pandas as pd

text_df = pd.DataFrame({'text': text, 'line': range(1,5)})

text_df

import spacy
nlp = spacy.load('en')

# example of using SpaCy to tokenize a simple string

def tokenize(sent):
    doc = nlp.tokenizer(sent)
    return [token.text for token in doc]
        
tokenize(text[0])

# The example in the book also lowercases everything
# and strips punctuation, so let's also do that:

def tokenize(sent):
    doc = nlp.tokenizer(sent)
    return [token.lower_ for token in doc if not token.is_punct]

print(tokenize(text[0]))

# Now we can use our `tokenize` function in combination
# with Pandas operations to expand the dataframe above into a tidy df

# First, how do we expand into tokens?
text_df['text'].apply(tokenize)

# Now we want each of those in its own row - two steps

new_df = (text_df['text'].apply(tokenize)
                         .apply(pd.Series))
new_df

# now use `stack` to reshape into a single column

new_df = new_df.stack()

new_df

new_df = (new_df.reset_index(level=0)
                .set_index('level_0')
                .rename(columns={0: 'word'}))

new_df

# Now we use a `join` to get the information from the other associated columns

new_df = new_df.join(text_df.drop('text', 1), how='left')

new_df

new_df = new_df.reset_index(drop=True)

new_df

def unnest_tokens(df, # line-based dataframe
                  column_to_tokenize, # name of the column with the text
                  new_token_column_name='word', # what you want the column of words to be called
                  tokenizer_function=tokenize): # what tokenizer to use
    
    return (df[column_to_tokenize]
              .apply(tokenizer_function)
              .apply(pd.Series)
              .stack()
              .reset_index(level=0)
              .set_index('level_0')
              .rename(columns={0: new_token_column_name})
              .join(text_df.drop(column_to_tokenize, 1), how='left')
              .reset_index(drop=True))

text_df = unnest_tokens(text_df, 'text')
text_df

