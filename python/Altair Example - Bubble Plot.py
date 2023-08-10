import mapd_renderer
import ibis
import altair as alt

TABLE_NAME = "cars"

conn = ibis.mapd.connect(host='qs-dev.mapd.com', port='9092', 
    user='mapd', password='HyperInteractive', 
    database='mapd', protocol='http')

conn.table(TABLE_NAME)

t = conn.table(TABLE_NAME)
expr = t[
    t.Horsepower,
    t.Miles_per_Gallon,
    t.Acceleration,
    ibis.row_id()
]

def chart(source):
    return alt.Chart(source).mark_circle(fill='blue', opacity=0.1).encode(
        alt.X(
            'Horsepower:Q',
            scale=alt.Scale(domain=[0, 240], range='width')
        ),
        alt.Y(
            'Miles_per_Gallon:Q',
            scale=alt.Scale(domain=[0, 50], range='height')
        ),
        alt.Size(
            'Acceleration:Q',
            scale=alt.Scale(domain=[0, 20])
        )
    )

alt.renderers.enable('default')
chart(expr.execute())

alt.renderers.enable('mapd', conn=conn)
chart(expr.compile())

