import folium

carte = folium.Map(location=[45.5236, -122.6750], zoom_start=12)
marker = folium.Marker([45.5, -122.7], popup='Un marker')
marker.add_to(carte)
carte

carte = folium.Map(location=[45.5236, -122.6750], zoom_start=12)
circle = folium.CircleMarker(
    [45.5, -122.7],
    radius=1000, 
    popup='Un cercle', 
    color="#e74c3c",       # rouge
    fill_color="#27ae60",  # vert
    fill_opacity=0.9
)
circle.add_to(carte)
carte

carte = folium.Map(location=[45.5236, -122.6750], zoom_start=12)

# add firt marker with bootstrap icon
icone1 = folium.Icon(icon="asterisk", icon_color="#9b59b6", color="lightblue")
marker1 = folium.Marker([45.5, -122.7], popup='Un icone', icon=icone1)
marker1.add_to(carte)

# add second marker with font-awesome icon
icone1 = folium.Icon(icon="globe", icon_color="#e67e22", color="lightgreen", prefix="fa")
marker1 = folium.Marker([45.5, -122.6], popup='Un icone', icon=icone1)
marker1.add_to(carte)

carte

import matplotlib
import matplotlib.pyplot as plt

color = plt.cm.winter(22)
color

matplotlib.colors.rgb2hex(color)

