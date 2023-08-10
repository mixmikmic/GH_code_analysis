get_ipython().magic("config InlineBackend.figure_format='retina'")
get_ipython().magic('matplotlib inline')

from urllib import request
import numpy as np
import pandas as pd

#pd.read_html('http://services.datasport.com/2012/lauf/lucernemarathon/rang091.htm')
url = 'https://services.datasport.com/2012/lauf/lucernemarathon/rang091.htm'

x = request.urlopen(url)
html = '\n'.join([l.decode('utf8') for l in x.readlines()])

text12 ="""Rang Name                    Jg   Land/Ort                 Zeit   Rückstand    Stnr         Kat/Rang       Schnitt ¦    Start-Horw ¦      Horw-HM ¦      HM-Horw ¦    Horw-Ziel ¦
     Team                                                                                                          ¦               ¦     - 21.1km ¦     - 33.8km ¦   - 42.195km ¦
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  1. Jeanneret Patrick       1971 Fribourg            2:36.02,1       -----    (52) Diplom  M-M40    1.       3.41 ¦   45.06     1.¦  31.23     1.¦  46.37     1.¦  32.54     1.¦
     Swisscom Marathon Team                                                                                        ¦               ¦1:16.30     1.¦2:03.07     1.¦2:36.02     1.¦
  2. Gisler Steve            1964 Erstfeld            2:39.07,9      3.05,8   (688) Diplom  M-M40    2.       3.46 ¦   45.13     2.¦  32.08     6.¦  47.38     2.¦  34.07     2.¦
     Imholz-Sport & Steve-Events                                                                                   ¦               ¦1:17.22     4.¦2:05.00     2.¦2:39.07     2.¦
  3. Weiss Alexander         1977 A-Kirchdorf an de   2:41.30,1      5.28,0   (788) Diplom  M-M30    1.       3.49 ¦   45.30     5.¦  31.33     3.¦  48.27     3.¦  35.58     8.¦
     Team Zisser Enns                                                                                              ¦               ¦1:17.03     2.¦2:05.31     3.¦2:41.30     3.¦
  4. Burkhard Beat           1975 Gutenswil           2:41.56,5      5.54,4  (1455) Diplom  M-M30    2.       3.50 ¦   45.29     4.¦  31.51     4.¦  49.19     6.¦  35.16     7.¦
     LC Uster                                                                                                      ¦               ¦1:17.20     3.¦2:06.40     4.¦2:41.56     4.¦
  5. Frei Rolf               1970 Uznach              2:41.56,6      5.54,5  (1817) Diplom  M-M40    3.       3.50 ¦   45.28     3.¦  31.59     5.¦  49.23     7.¦  35.06     5.¦
     sport trend shop                                                                                              ¦               ¦1:17.27     5.¦2:06.50     5.¦2:41.56     5.¦
  6. Wyss Roman              1976 Niederbipp          2:43.09,8      7.07,7  (2002) Diplom  M-M30    3.       3.52 ¦   46.30    11.¦  32.48    10.¦  48.44     4.¦  35.05     4.¦
                                                                                                                   ¦               ¦1:19.19     8.¦2:08.04     6.¦2:43.09     6.¦
  7. Eggenberger Michael     1977 Zug                 2:43.49,7      7.47,6  (1253) Diplom  M-M30    4.       3.52 ¦   45.44     7.¦  32.34     8.¦  50.23    10.¦  35.07     6.¦
     VELORADO Racing Team                                                                                          ¦               ¦1:18.19     7.¦2:08.42     7.¦2:43.49     7.¦
  8. Elmer Rico              1969 Elm                 2:46.29,6     10.27,5   (833) Diplom  M-M40    4.       3.56 ¦   46.43    13.¦  32.51    11.¦  50.37    12.¦  36.17    11.¦
     Central Garage Glarus                                                                                         ¦               ¦1:19.35    12.¦2:10.12    10.¦2:46.29     8.¦
  9. Kacir Vlastimil         1979 CZ-Praha            2:47.08,2     11.06,1   (413) Diplom  M-M30    5.       3.57 ¦   46.39    12.¦  32.43     9.¦  49.54     8.¦  37.50    28.¦
                                                                                                                   ¦               ¦1:19.22     9.¦2:09.17     8.¦2:47.08     9.¦
 10. Willcock Patrick        1976 Müswangen           2:47.20,7     11.18,6  (1562) Diplom  M-M30    6.       3.57 ¦   48.25    24.¦  33.49    19.¦  50.17     9.¦  34.48     3.¦
     Pilatus Ski Club                                                                                              ¦               ¦1:22.14    23.¦2:12.32    15.¦2:47.20    10.¦
"""

text15 = """Rang Name                    Jg   Land/Ort                 Zeit   Rückstand     Stnr               Kat/Rang       Schnitt ¦    Start-Horw ¦      Horw-HM ¦      HM-Horw ¦    Horw-Ziel ¦
       Team                                                                                                                 ¦               ¦     - 21.1km ¦     - 33.8km ¦              ¦
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    1. Kuert Fabian            1983 Langenthal          2:26.18,1       -----   (1161) Diplom Video  M-M30    1.       3.28 ¦   42.30     2.¦  30.16     2.¦  43.24     1.¦  30.06     1.¦
                                                                                                                            ¦               ¦1:12.47     2.¦1:56.11     1.¦              ¦
    2. Wenk Stephan            1982 Greifensee          2:29.09,3      2.51,2     (20) Diplom Video  M-M30    2.       3.32 ¦   42.22     1.¦  30.11     1.¦  43.38     2.¦  32.56     5.¦
       TV Oerlikon/Scott                                                                                                    ¦               ¦1:12.34     1.¦1:56.12     2.¦              ¦
    3. Bodenmann Heinz         1988 Gais                2:32.10,7      5.52,6    (229) Diplom Video  M-M20    1.       3.36 ¦   43.45     5.¦  31.13     3.¦  44.38     3.¦  32.33     2.¦
       LG Rheintal                                                                                                          ¦               ¦1:14.59     3.¦1:59.37     3.¦              ¦
    4. Jenni Walter            1968 Oberwil b. Büren    2:35.38,6      9.20,5   (1808) Diplom Video  M-M45    1.       3.41 ¦   44.19     8.¦  32.31    10.¦  45.54     4.¦  32.52     4.¦
                                                                                                                            ¦               ¦1:16.51     9.¦2:02.46     5.¦              ¦
    5. Frieden Thomas          1969 Kollbrunn           2:36.04,9      9.46,8    (240) Diplom Video  M-M45    2.       3.41 ¦   44.51    12.¦  32.18     7.¦  46.14     5.¦  32.40     3.¦
       LSV Winterthur                                                                                                       ¦               ¦1:17.09    10.¦2:03.24     7.¦              ¦
    6. Eggenschwiler Bernhard  1985 Büsserach           2:37.56,2     11.38,1    (438) Diplom Video  M-M30    3.       3.44 ¦   44.52    13.¦  32.33    11.¦  46.51     7.¦  33.39     6.¦
       mega-joule.ch                                                                                                        ¦               ¦1:17.26    13.¦2:04.17    10.¦              ¦
    7. Jeanneret Patrick       1971 Bern                2:38.23,7     12.05,6     (56) Diplom Video  M-M40    1.       3.45 ¦   44.21    10.¦  31.36     4.¦  46.59     8.¦  35.27    17.¦
       Trilogie Running Team - Mizuno                                                                                       ¦               ¦1:15.57     6.¦2:02.56     6.¦              ¦
    8. Arnold Philipp          1987 Cham                2:39.02,4     12.44,3    (218) Diplom Video  M-M20    2.       3.46 ¦   43.51     6.¦  32.18     8.¦  47.36    11.¦  35.15    14.¦
       LAC TV Unterstrass                                                                                                   ¦               ¦1:16.10     7.¦2:03.47     8.¦              ¦
    9. Schmauder Stefan        1985 Diepoldsau          2:40.00,7     13.42,6    (232) Diplom Video  M-M30    4.       3.47 ¦   44.20     9.¦  32.52    14.¦  47.51    14.¦  34.56    11.¦
       LG Rheintal                                                                                                          ¦               ¦1:17.13    11.¦2:05.04    11.¦              ¦
   10. Perino José Manuel      1980 Luzern              2:40.02,3     13.44,2   (1734) Diplom Video  M-M35    1.       3.47 ¦   44.56    15.¦  33.02    16.¦  48.17    16.¦  33.45     7.¦
       erdbeergold.ch                                                                                                       ¦               ¦1:17.59    15.¦2:06.16    14.¦              ¦

"""

text = text12
lines = text.split('\n')
for n,line in enumerate(lines[3:]):
    #
    fields = [f.strip() for f in line.strip().split('  ')]
    fields = [f for f in fields if f]
    if n%2: #odd lines
        pass
    else:
        name, place, total, _,_, cat, _,_, split1,_,split2,_,split3,_,split4,_ = fields
        pos, name = name.split(maxsplit=1)
        yob, city = place.split(maxsplit=1)
        print('|'.join((pos, name, city, total, cat, split1, split2, split3, split4)))
        print('='*20)

fields
pos,first,last, _,_, total,_,_,_,cat,_,_,_,split1,_,split2,_,split3,_,split4,_ = fields
print(pos, first, last, total, cat, split1, split2, split3, split4)

