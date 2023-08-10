import pystq.widget as widget
from pystq.widget import spaceTimeQuest
import pystq.materials as materials
import pystq.sites as sites

aLIGO = {
    'depth' : 0,
    'pumps' : 6,
    'sus_stages' : 4,
    'sus_length' : 0.5,
    'mirror_mass' : 40,
    'power' : 125,
    'roughness' : 100,
    'site' : sites.Desert,
    'material' : materials.Silica,
    'temperature' : 290,
}

stq = spaceTimeQuest(aLIGO)
stq.tabs

Voyager = {
    'depth' : 0,
    'pumps' : 8,
    'sus_stages' : 4,
    'sus_length' : 0.5,
    'mirror_mass' : 80,
    'power' : 150,
    'roughness' : 10,
    'site' : sites.Desert,
    'material' : materials.Silicon,
    'temperature' : 295,
}

