import mapd_renderer
import ibis
import altair as alt

mapd_cli = ibis.mapd.connect(
    host='metis.mapd.com', user='mapd', password='HyperInteractive',
    port=443, database='mapd', protocol= 'https'
)

t = mapd_cli.table('tweets_nov_feb')

expr = t[
    t.goog_x.name('x'),
    t.goog_y.name('y'),
    t.tweet_id.name('rowid')
]

print(expr.compile())

df = expr.execute()
df.head()

alt.data_transformers.enable('json')
alt.Chart(df, width=384, height=564).mark_square(
    color='green',
    size=2,
    clip=True
).encode(
    alt.X(
        'x:Q',
        scale=alt.Scale(domain=[-3650484.1235206556, 7413325.514451755], range='width')
    ),
    alt.Y(
        'y:Q',
        scale=alt.Scale(domain=[-5778161.9183506705, 10471808.487466192], range='height')
    ),
)

alt.renderers.enable('mapd', conn=mapd_cli)
alt.Chart(expr.compile(), width=384, height=564).mark_square(
    color='green',
    size=2,
    clip=True
).encode(
    alt.X(
        'x:Q',
        scale=alt.Scale(domain=[-3650484.1235206556, 7413325.514451755], range='width')
    ),
    alt.Y(
        'y:Q',
        scale=alt.Scale(domain=[-5778161.9183506705, 10471808.487466192], range='height')
    ),
)

