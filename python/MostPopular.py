# Set full page width
from IPython.core.display import HTML
HTML("""
<style>
.container {
    width: 100%;
}
</style>
""")

import graphlab as gl
gl.canvas.set_target('ipynb')
import datetime

train = gl.SFrame('Data/train.csv')
print "Train:", len(train)

summary = train.groupby(['srch_destination_id', 'hotel_cluster'], 
                        {'bookings':gl.aggregate.SUM('is_booking'), 
                         'clicks': gl.aggregate.COUNT()} )
summary

BOOKING_WEIGHT = 100
summary['clicks'] = summary['clicks'] - summary['bookings'] 
summary['relevance'] = summary['bookings']*BOOKING_WEIGHT + summary['clicks']
summary

def arg_max(d, k=3):
    topk = sorted(d.keys(), reverse=True)[:k]
    
    return  [str(d[k]) for k in topk]

most_popular = summary.groupby('srch_destination_id',
                       {'hotel_cluster': gl.aggregate.CONCAT('relevance', 'hotel_cluster')})

most_popular['hotel_cluster'] = most_popular['hotel_cluster'].apply(lambda d: arg_max(d, k=5), list)
most_popular.print_rows(5, max_column_width=40)

test=gl.SFrame('Data/test.csv')

test = test.join(most_popular, how='left',on='srch_destination_id')
test

test['hotel_cluster'].num_missing()

most_popular_all = summary.groupby('hotel_cluster',
                       {'relevance': gl.aggregate.SUM('relevance')})

most_popular_all = list(most_popular_all.topk('relevance', k=5)['hotel_cluster'].astype(str))
most_popular_all

test = test.fillna('hotel_cluster', most_popular_all)

test['hotel_cluster'] = test['hotel_cluster'].apply(lambda lst: ' '.join(lst))

test['srch_destination_id', 'hotel_cluster']

test['id', 'hotel_cluster'].save('Submissions/MostPopular.csv')
# submission scored 0.30253

