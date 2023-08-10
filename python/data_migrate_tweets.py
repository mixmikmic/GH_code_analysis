import mapd_renderer
import ibis
import altair as alt

host = dict(
    host='metis.mapd.com', user='mapd', password='HyperInteractive',
    port=443, database='mapd', protocol= 'https'
)

def sample(n=1000):
    mapd_cli = ibis.mapd.connect(**host)
    t = mapd_cli.table('tweets_nov_feb')
    expr = t[t.goog_x, t.goog_y, t.tweet_id]
    sample = expr.filter([
        expr.goog_x < 7413325.514451755,
        expr.goog_x > -3650484.1235206556, 
        expr.goog_y < 10471808.487466192,
        expr.goog_y > -5778161.9183506705, 
    ])
    return mapd_cli, sample, sample.limit(1000).execute()

def main():
    prev, expr, df = sample()

    new = ibis.mapd.connect(
        host='qs-dev.mapd.com', port='9091', 
        user='mapd', password='HyperInteractive', 
        database='mapd')

    if not 'tweet' in new.list_tables():
        new.load_data('tweet', df)

if __name__ == '__main__': main()

