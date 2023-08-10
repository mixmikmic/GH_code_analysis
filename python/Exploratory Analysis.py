#Initial imports
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
from matplotlib import pyplot as plt
from pandas import DataFrame
get_ipython().magic('matplotlib inline')

file_path = "data/enamin-2012.csv"
enamin12 = pd.read_csv(file_path, sep=',', encoding="latin-1")
enamin12.head()

data = enamin12[['ENT', 'P9_C', 'P27', 'P27_1', 'P27_2', 'P29', 'P34', 'P35']]
data.columns = ['state_code', 'activity_code', 'founder_code', 'year_start', 'month_start', 'reason_code',
                'owner_code', 'type_assoc_code']
data.head(10)

state_codes = {
    1:'Aguascalientes',
    2:'Baja California Norte',
    3:'Baja California Sur',
    4:'Campeche',
    5:'Coahuila',
    6:'Colima',
    7:'Chiapas',
    8:'Chihuahua',
    9:'Ciudad de México',
    10:'Durango',
    11:'Guanajuato',
    12:'Guerrero',
    13:'Hidalgo',
    14:'Jalisco',
    15:'Estado de México',
    16:'Michoacán',
    17:'Morelos',
    18:'Nayarit',
    19:'Nuevo león',
    20:'Oaxaca',
    21:'Puebla',
    22:'Querétaro',
    23:'Quintana roo',
    24:'San Luis Posotí',
    25:'Sinaloa',
    26:'Sonora',
    27:'Tabasco',
    28:'Tamaulipas',
    29:'Tlaxcala',
    30:'Veracruz',
    31:'Yucatán',
    32:'Zacatecas'
}

activity_codes = {
    1110:'Agricultura',
    1121:'Ganadería',
    1130:'Aprovechamiento forestal',
    1141:'Pesca',
    2361:'Edificación residencial',
    2370:'Construcción de obras de ingeniería civil u obra pesada',
    2382:'Industria alimentaria',
    3110:'Industria de las bebidas y del tabaco',
    3120:'Fabricación de insumos textiles',
    3130:'Confección de productos textiles, excepto prendas de vestir',
    3140:'Confección de productos textiles, excepto prendas de vestir',
    3150:'Fabricación de prendas y accesorios de vestir',
    3160:'Fabricación de productos de cuero, piel y materiales sucedáneos, excepto prendas de vestir',
    3210:'Industria de la madera',
    3220:'Industria de papel',
    3230:'Impresión e industria conexas',
    3270:'Fabricación de productos a base de minerales no metálicos',
    3310:'Industrias metálicas básicas',
    3320:'Fabricación de productos metálicos',
    3370:'Fabricación de muebles y productos relacionados (colchones, cortineros)',
    3380:'Otras industrias manufactureras',
    4310:'Comercio al por mayor de alimentos, bebidas y tabaco',
    4320:'Comercio al por mayor de productos textiles y calzado',
    4340:'Comercio al por mayor de materias primas agropecuarias, para la industria y materiales de desecho',
    4611:'Comercio al por menor de alimentos, bebidas y tabaco',
    4612:'Comercio ambulante de productos alimenticios y bebidas',
    4620:'Comercio al por menor en tiendas de autoservicio y departamentales',
    4631:'Comercio al por menor de productos textiles, ropa nueva, accesorios de vestir y calzado',
    4632:'Comercio ambulante de productos textiles y ropa nueva, accesorios de vestir y calzado',
    4641:'Comercio al por menor de artículos para el cuidado de la salud',
    4642:'Comercio ambulante de artículos para el cuidado de la salud',
    4651:'Comercio al por menor de artículos de papelería, para el esparcimiento y otros artículos de uso personal',
    4652:'Comercio ambulante de artículos de papelería, para el esparcimiento y otros artículos de uso personal',
    4661:'Comercio al por menor de enseres domésticos, computadoras y artículos para la decoración de interiores',
    4662:'Comercio ambulante de muebles para el hogar y otros enseres domésticos',
    4671:'Comercio al por menor de artículos de ferretería, tlapalería y vidrios',
    4672:'Comercio ambulante al por menor de artículos de ferretería y tlapalería',
    4681:'Comercio al por menor de vehículos de motor, refacciones, combustibles y lubricantes',
    4682:'Comercio ambulante de partes y refacciones para automóviles, camionetas y camiones',
    4690:'Intermediación y comercio al por menor por medios masivos de comunicación y otros medios',
    4840:'Autotransporte de carga',
    4881:'Servicios relacionados con el transporte',
    5110:'Edición de publicaciones y de software, excepto a través de internet',
    5322:'Servicios de alquiler y centros de alquiler de bienes muebles, excepto equipo de transporte terrestre',
    5411:'Servicios profesionales, científicos y técnicos',
    5611:'Servicios de administración de negocios, de empleo, apoyo secretarial y otros servicios de apoyo a los negocios',
    5613:'Servicios de limpieza y de instalación y mantenimiento de áreas verdes',
    5620:'Manejo de desechos y servicios de remediación',
    6141:'Otros servicios educativos pertenecientes al sector privado',
    6252:'Guarderías del sector público',
    7111:'Compañías y grupos de espectáculos artísticos',
    7114:'Artistas y técnicos independientes',
    7131:'Servicios de entretenimiento en instalaciones recreativas y otros servicios recreativos',
    7210:'Servicios de alojamiento temporal',
    7221:'Servicios de preparación de alimentos y bebidas',
    7222:'Servicios de preparación de alimentos y bebidas por trabajadores ambulantes',
    8111:'Reparación y mantenimiento de automóviles y camiones',
    8112:'Reparación y mantenimiento de equipo, maquinaria, artículos para el hogar y personales',
    8121:'Servicios personales',
    8122:'Estacionamientos y pensiones para automóviles',
    8123:'Servicios de cuidado y de lavado de automóviles por trabajadores ambulantes',
    8130:'Asociaciones y organizaciones',
    8140:'Hogares con empleado domésticos',
    9700:'Trabajador de otro trabajador'
}

reason_codes = {
    1:'Por tradición familiar o lo heredó',
    2:'Para complementar el ingreso familiar',
    3:'Para mejorar el ingreso',
    4:'Tenía dinero y encontró una buena oportunidad',
    5:'Para ejercer su oficio, carrera o profesión',
    6:'Fue la única manera que tuvo para obtener ingreso',
    7:'No tenía experiencia requerida para un empleo',
    8:'No tenía escolaridad o capacitación requerida para un empleo',
    9:'Estaba sobre capacitado',
    10:'Los empleos que encontró estaban mal pagados',
    11:'Requería un horario flexible',
    12:'No había oportunidades de empleo',
    13:'Otra razón'
}

owner_codes = {
    1:'Un solo dueño',
    2:'Varios dueños'
}

type_assoc_codes = {
    1:'Familiar',
    2:'No familiar',
    3:'Familiar y no familiar'
}

founder_codes = {
    1:'Usted solo (a)',
    2:'Su pareja o cónyuge',
    3:'Usted y su pareja o cónyuge (u otro familiar)',
    4:'Usted y otra (s) personas (s), no familiares',
    5:'Otro(s) familiares(s)',
    6:'Otras(s) persona(s)'
}

data['state'] = data['state_code'].map(state_codes)
data['activity'] = data['activity_code'].map(activity_codes)
data['founder'] = data['founder_code'].map(founder_codes)
data['reason'] = data['reason_code'].map(reason_codes)
data['owner'] = data['owner_code'].map(owner_codes)
data['type_assoc'] = data['type_assoc_code'].map(type_assoc_codes)
data.head(10)

data['year_start'][data['year_start'] == 9999] = np.nan

data['comp_age'] = datetime.now().year - data['year_start']

data.head(10)

# Most prosperous business nation wide
plt.bar(data.comp_age, data.activity_code, bin)

max(data.comp_age)



