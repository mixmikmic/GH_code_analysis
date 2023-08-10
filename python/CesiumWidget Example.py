import json
from IPython import display
from CesiumWidget import CesiumWidget

cesium = CesiumWidget()

cesium

cesium.class_own_traits()

cesium.enable_lighting = True

from CesiumWidget.examples.iss import ISS
cesium.czml = ISS

