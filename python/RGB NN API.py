get_ipython().run_cell_magic('writefile', 'rgb_nn.py', '\nimport urllib.request\nimport json\nimport numpy as np\nimport pandas as pd\nimport colorsys\nfrom pprint import pprint\n\nclass RGB_NN():\n# class for displaying NN weights on RGB NN of shape (2:3:2)\n\n    _server_loc = \'http://192.168.1.162:5000\'\n    _url = _server_loc +\'/change_leds\'\n    \n    _blank_led_json = {"red":0, "green":0, "blue":0, "led_num":0}\n    _led_num = 44\n    _blank_leds_json = {}\n    \n    _brightness = 1\n    _scale = (-1,1)\n    _verbose = False\n    _dummy_api = False\n    \n    def __init__(self, server_loc, scale=None):\n        self._server_loc = server_loc\n        self._url = self._server_loc +\'/change_leds\'\n        self._blank_leds_json = {"leds":[{**self._blank_led_json, "led_num":i} for i in range(self._led_num)]}\n        self._scale = scale if scale else self._scale\n        \n    def display_weights(self, clf):\n        led_json = {"leds": self._make_led_coef_vals(clf)}\n        return self._send_json(led_json)\n    \n    \n    def _send_json(self, led_json):\n        if self._dummy_api:\n            return led_json\n        params = json.dumps(led_json).encode(\'utf8\')\n        req = urllib.request.Request(self._url, data=params,\n                                  headers={\'content-type\': \'application/json\'})\n        response = urllib.request.urlopen(req)\n        return True\n        \n    def _turn_off(self):\n        _send_json(_blank_leds_json)\n        \n    def _cap_val(self, val):\n        if val > self._scale[1]:\n            return self._scale[1]\n        elif val < self._scale[0]:\n            return self._scale[0]\n        else:\n            return val\n    \n    def _norm_val(self, val):\n        capped = self._cap_val(val)\n        shifted = capped - self._scale[0]\n        scaled = shifted / (self._scale[1] - self._scale[0])\n        return scaled\n    \n    def _val_to_rgb(self, val):\n        \'\'\' \n            return [RED, GREEN, BLUE] for coef weight based on scale\n        \'\'\'\n        i = self._norm_val(val)\n        R2B_hue_range = i * (1/3)*2\n        c = colorsys.hsv_to_rgb(R2B_hue_range,1,1)\n        rgb = [int((color*255)*self._brightness) for color in c]\n        if self._verbose:\n            print(\'val {0} normed to {1} to color {2}\'.format(val,i,rgb))\n        return rgb\n    \n    def _led_json_with_rgb_value(self, led_num, val):\n        r, b, g = self._val_to_rgb(val)\n        if self._verbose:\n            print(\'mapping led {0}\'.format(led_num))\n        return {"led_num":led_num, "red":r, "green":g, "blue":b}\n    \n    \n    # led mappings\n\n    _input_layer_bias = [\n        [8,9],\n        [3,4]\n    ]\n\n    _input_layer_weights = [\n        [[5,11],[6,20],[7,28]],\n        [[0,10],[1,18],[2,27]]\n    ]\n\n    _hidden_layer_bias = [\n        [13,14],\n        [21,22],\n        [29,30]\n    ]\n\n    _hidden_layer_weights = [\n        [[16,41],[17,36]],\n        [[23,40],[25,35]],\n        [[31,39],[32,34]]\n    ]\n\n    _output_layer_bias = [\n        [42,43],\n        [37,38]\n    ]\n    \n    def _map_weights_to_leds(self, coefs, led_map):\n        leds = []\n        for i,v in enumerate(led_map):\n            for i2,v2 in enumerate(v):\n                for led in v2:\n                    led_json = self._led_json_with_rgb_value(led, coefs[i][i2])\n                    leds.append(led_json)\n        if self._verbose:\n            print(\'did weights {0}\'.format(leds))\n        return leds\n\n    def _map_biases_to_leds(self,bias, led_map):\n        leds = []\n        for i,v in enumerate(led_map):\n            for led in v:\n                led_json = self._led_json_with_rgb_value(led, bias[i])\n                leds.append(led_json)\n        if self._verbose:\n            print(\'did biases {0}\'.format(leds))\n        return leds\n\n    def _make_led_coef_vals(self, clf):\n        leds = []\n        leds += self._map_weights_to_leds(clf.coefs_[0], self._input_layer_weights)\n        leds += self._map_weights_to_leds(clf.coefs_[1], self._hidden_layer_weights)\n        leds += self._map_biases_to_leds(clf.intercepts_[0], self._hidden_layer_bias)\n        leds += self._map_biases_to_leds(clf.intercepts_[1], self._output_layer_bias)\n        return leds')

con = RGB_NN(1892018)

con.display_weights()

import colorsys

colorsys.hsv_to_rgb(0.9,1,1)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

r =[]
g =[]
b =[]
i = []
for i in range(100):
    ri, gi, bi = colorsys.hsv_to_rgb(i/100,1,1)
    r.append(ri)
    g.append(gi)
    b.append(bi)

plt.plot(list(range(100)), r, c='r')
plt.plot(list(range(100)), g, c='g')
plt.plot(list(range(100)), b, c='b')
plt.show()

colorsys.hsv_to_rgb((1/3)*2,1,1)



