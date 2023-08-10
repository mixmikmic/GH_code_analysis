import graphlab as gl

sf = gl.SFrame({'srch_destination_id':[1, 1, 1, 1, 2, 2, 2], 
                'relevance':[4.95, 3.9, 0.05, 1.1, 0.2, 1.3, 0.05], 
                'hotel':[30, 63, 60, 25, 48, 82, 58]})

# For each srch_destination_id create a dictonary of relevance:hotel usign CONCAT
summary = sf.groupby('srch_destination_id',{'hotel_cluster': gl.aggregate.CONCAT('relevance', 'hotel')})
summary

# Create a function to return the top k items in the diction by relevance
def arg_max(d, k=3):
    topk = sorted(d.keys(), reverse=True)[:k]
    return [d[k] for k in topk]

arg_max({1.3: 82, 0.2: 48, 0.05:58})

# Apply this function to each row to create the top 5 resules by releance, with the most relevant first.
# NOTE - for some reason this change the hotel id from an int() to a float()
summary['hotel_cluster'] = summary['hotel_cluster'].apply(lambda d: arg_max(d, k=5))
summary



