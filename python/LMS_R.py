import pandas as pd, numpy as np, ast, re

def parse_np_array(array_string):
    pattern = r'''# Match (mandatory) whitespace between...
              (?<=\]) # ] and
              \s+
              (?= \[) # [, or
              |
              (?<=[^\[\]\s]) 
              \s+
              (?= [^\[\]\s]) # two non-bracket non-whitespace characters
           '''
    fixed_string = re.sub(pattern, ',', array_string, flags=re.VERBOSE)
    return np.array(ast.literal_eval(fixed_string))

LMSR_df = pd.read_csv("datasets/LMSR_rev2vec.csv")
LMSR_df["rev_vec"] = LMSR_df["rev_vec"].apply(lambda x: parse_np_array(x) if type(x) == str and "[" in x else None)
LMSR_df.head(5)

def merging_function(frame):
    return np.mean(frame["rev_vec"])

LMS_r = LMSR_df.copy()
# LMS_r = LMS_r.groupby(["Movie_ID","Language","Score"])
LMS_r = LMS_r.groupby(["Movie_ID","Language","Score"], as_index=False).apply(merging_function)
LMS_r.to_frame().head(5)

LMS_r_df = LMS_r.reset_index().rename({0:"merged_reviews_vector"}, axis=1)
LMS_r_df.head(5)

LMS_r_df.to_csv("datasets/LMS_r_merged_reviews_per_movie_language_score.csv", index=False)



