get_ipython().run_cell_magic('javascript', '', "\nlet myVar = 'test'\nconsole.log(myVar)")

get_ipython().run_cell_magic('javascript', '', "\nelement.text('Hello')")

get_ipython().run_cell_magic('javascript', '', "require.config({\n    paths: {\n        'p5': 'https://cdnjs.cloudflare.com/ajax/libs/p5.js/0.6.0/p5.min',\n        'lodash': 'https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.4/lodash.min'\n    }\n});")

get_ipython().run_cell_magic('javascript', '', '\nwindow.defineModule = function (name, dependencies, module) {\n    // force the recreation of the module\n    // (when re-executing a cell)\n    require.undef(name);\n    \n    define(name, dependencies, module);\n};')

get_ipython().run_cell_magic('javascript', '', "\nwindow.createSketchView = function (name, dependencies, module) {\n    \n    require.undef(name);\n    \n    define(name,\n           ['@jupyter-widgets/base', 'p5', 'lodash'].concat(dependencies),\n           (widgets, p5, _, ...deps) => {\n\n        let viewName = `${name}View`;\n        \n        let View = widgets.DOMWidgetView.extend({\n            initialize: function () {\n                this.el.setAttribute('style', 'text-align: center;');\n            },\n\n            render: function () {\n                // pass the model as the last dependency so it can\n                // be accessed in the sketch\n                let sketch = module(...deps, this.model);\n                setTimeout(() => {\n                    this.sketch = new p5(sketch, this.el);                    \n                }, 0);\n            },\n\n            remove: function () {\n                // stop the existing sketch when the view is removed\n                // so p5.js can cancel the animation frame callback and free up resources\n                if (this.sketch) {\n                    this.sketch.remove();\n                    this.sketch = null;\n                }\n            }\n        });\n        \n        return {\n            [viewName] : View,\n        };\n    });\n}")

get_ipython().run_cell_magic('javascript', '', "\n// Test module defining a few constants, for example purposes\n// Such constants should ideally be defined directly in the model\n// and directly accessed by the view\n\ndefineModule('testModule', [], () => {\n    const [W, H] = [500, 500];\n    return {W, H};\n})")

get_ipython().run_cell_magic('javascript', '', "\ncreateSketchView('Sketch2D', ['testModule'], (TestModule, model) => {\n    return function(p) {\n        const {W, H} = TestModule;\n        const [CX, CY] = [W / 2, H / 2];\n        \n        p.setup = function(){\n            p.createCanvas(W, H);\n            p.rectMode(p.CENTER);\n        }\n\n        p.draw = function () {\n            p.background('#ddd');\n            p.translate(CX, CY);\n            let n = model.get('n_squares');\n            _.range(n).forEach(i => {\n                p.push();\n                p.rotate(p.frameCount / 200 * (i + 1));\n                p.fill(i * 5, i * 100, i * 150);\n                p.rect(0, 0, 200, 200);\n                p.pop();\n            });\n        }\n    };\n})")

import ipywidgets as widgets
from traitlets import Unicode, Int


class Sketch2D(widgets.DOMWidget):
    _view_name = Unicode('Sketch2DView').tag(sync=True)
    _view_module = Unicode('Sketch2D').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)
    n_squares = Int(1).tag(sync=True)

sketch_2d = Sketch2D()
sketch_2d

sketch_2d.n_squares = 4

get_ipython().run_cell_magic('javascript', '', "\ncreateSketchView('Sketch3D', ['testModule'], (Settings, model) => {\n    return function(p) {\n        const {W, H} = Settings;\n        \n        p.setup = function(){\n            p.createCanvas(W, H, p.WEBGL);\n        }\n\n        p.draw = function () {\n            p.background('#ddd');\n            let t = p.frameCount;\n            let n = model.get('n_cubes');\n            p.randomSeed(42);\n            _.range(n).forEach(i => {\n                const R = 180 //+ 30 * p.sin(t * 0.2 + i);\n                const x = R * p.cos(i * p.TWO_PI / n);\n                const y = R * p.sin(i* p.TWO_PI / n);\n                p.push();\n                p.translate(x, y);\n                p.fill(p.random(255), p.random(255), p.random(255));\n                p.rotateY(t * 0.05 + i);\n                p.box(50);\n                p.pop();\n            });\n        } \n    };\n})")

class Sketch3D(widgets.DOMWidget):
    _view_name = Unicode('Sketch3DView').tag(sync=True)
    _view_module = Unicode('Sketch3D').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)
    n_cubes = Int(4).tag(sync=True)

sketch_3d = Sketch3D()
sketch_3d

sketch_3d.n_cubes = 10

N_TRIANGLES = 16
IDS = list(range(N_TRIANGLES))
N_COLORS = 6
WHITE, BLUE, YELLOW, GREEN, BLACK, RED = range(N_COLORS)

# list the pieces, turning anti-clockwise for the colors
# The number at the last position of the tuple indicates the number
# of identical pieces (cf photo above)
triangles_count = [
    (WHITE, BLUE, BLUE, 1),
    (WHITE, YELLOW, GREEN, 2),  # 2 of these
    (WHITE, BLACK, BLUE, 2),  # 2 of these
    (WHITE, GREEN, RED, 1),
    (WHITE, RED, YELLOW, 1),
    (WHITE, WHITE, BLUE, 1),
    (BLACK, GREEN, RED, 1),
    (BLACK, RED, GREEN, 2),  # 2 of these
    (BLACK, BLACK, GREEN, 1),
    (BLACK, GREEN, YELLOW, 1),
    (BLACK, YELLOW, BLUE, 1),
    (GREEN, RED, YELLOW, 1),
    (BLUE, GREEN, YELLOW, 1)
]

assert N_TRIANGLES == sum(t[-1] for t in triangles_count)

triangles = tuple([t[:-1] for t in triangles_count for times in range(t[-1])])
print(triangles)

assert N_TRIANGLES == len(triangles)

from traitlets import List, Tuple, Dict, validate, default

class Board(widgets.DOMWidget):    
    TRIANGLES = List(triangles).tag(sync=True)
    LEFT = Tuple((WHITE, RED, WHITE, YELLOW)).tag(sync=True)
    RIGHT = Tuple((BLUE, RED, GREEN, BLACK)).tag(sync=True)
    BOTTOM = Tuple((GREEN, GREEN, WHITE, GREEN)).tag(sync=True)
    
    positions = List().tag(sync=True)
    permutation = List([]).tag(sync=True)
    
    @default('positions')
    def _default_positions(self):
        triangle_id, positions = 0, []
        for row in range(4):
            n_row = 2 * row + 1
            for col in range(n_row):
                flip = (triangle_id + row) % 2
                positions.append({
                    'id': triangle_id,
                    'flip': flip,
                    'row': row,
                    'col': col,
                    'n_row': n_row
                })
                triangle_id += 1
        return positions
    
    @default('permutation')
    def _default_permutation(self):
        return self.random()
    
    def random(self):
        return [[i, 0] for i in range(N_TRIANGLES)]

b = Board(permutation=[[i, 0] for i in range(N_TRIANGLES)])
b.positions

get_ipython().run_cell_magic('javascript', '', "\n// define a list of constant such as the size of the base canvas,\n// the size of the triangles, colors...\ndefineModule('settings', [], () => {\n    const ANIM_W = 800;\n    const ANIM_H = ANIM_W / 1.6;\n    const N_TRIANGLES = 16;\n    const COLORS = ['#fff', '#00f', '#ff0', '#0f0', '#000', '#f00'];\n    const WOOD_COLOR = '#825201';\n    const R = 50;\n    const r = R / 2;\n    const CR = r / 2;\n    const OFFSET_X = R * Math.sqrt(3) * 0.5;\n    const OFFSET_Y = 1.5 * R;\n    \n    return {ANIM_W, ANIM_H, N_TRIANGLES, WOOD_COLOR, COLORS, R, r, CR, OFFSET_X, OFFSET_Y};\n})")

get_ipython().run_cell_magic('javascript', '', "\ndefineModule('triangles', ['settings'], (settings) => {\n    const {COLORS, WOOD_COLOR, R, r, CR, OFFSET_X, OFFSET_Y} = settings;\n    \n    function _getPoints(n, startAngle, radius) {\n        let points = [];\n        const da = 2 * Math.PI / n;\n        for (let i = 0; i < n; i++) {\n            const angle = startAngle - i * da;\n            const x = radius * Math.cos(angle);\n            const y = radius * Math.sin(angle);\n            points.push(x);\n            points.push(y);\n        }\n        return points;\n    }\n    \n    return (p) => {\n        return {\n            getTrianglePoints: _getPoints,\n            getTriangleCoordinates: function (flip, row, col) {\n                const x = (col - row) * OFFSET_X;\n                const y = row * OFFSET_Y + ((flip === 1) ? -R/2 : 0);\n                return {x, y};\n            },\n            drawTriangle: function (colors, x, y, rotation, flip=0) {\n                const n = colors.length;\n                \n                p.fill(WOOD_COLOR);\n                p.push();\n                p.translate(x, y);\n                p.rotate(-rotation * p.TWO_PI / 3 + flip * p.PI);\n                \n                p.triangle(..._getPoints(n, Math.PI / 6, R));\n\n                let circles = _getPoints(n, Math.PI / 2, 1.25 * CR);\n                for (let i = 0; i < n; i++) {\n                    const xx = circles[2*i];\n                    const yy = circles[2*i+1];\n                    const color = COLORS[colors[i]];\n                    p.fill(color);\n                    p.ellipse(xx, yy, CR);\n                }\n                p.pop();\n            }\n        };\n    };\n});")

get_ipython().run_cell_magic('javascript', '', "\ndefineModule('staticBoard', ['settings', 'triangles'], (Settings, Triangles) => {\n    let {COLORS, R, CR, OFFSET_X, OFFSET_Y} = Settings;\n    \n    return (p) => {\n        let triangles = Triangles(p);\n        \n        function _drawStaticColors (left, right, bottom, positions) {\n            for (let {flip, row, col, n_row} of positions) {\n                const {x, y} = triangles.getTriangleCoordinates(flip, row, col);\n                if (col === 0) {\n                    const colorLeft = COLORS[left[row]];\n                    const colorRight = COLORS[right[row]];\n                    p.fill(colorLeft);\n                    p.ellipse(x - OFFSET_X, y - R / 2, CR);\n                    p.fill(colorRight);\n                    p.ellipse(x + n_row * OFFSET_X, y - R / 2, CR);\n                }\n                          \n                if (row === 3 && col % 2 == 0) {\n                    p.fill(COLORS[bottom[parseInt(col / 2, 10)]]);\n                    p.ellipse(x, 3.75 * OFFSET_Y, CR);\n                }\n            }\n        }\n        \n        function _drawFrame (positions) {\n            const {flip, row, col} = positions[6];\n            const {x, y} = triangles.getTriangleCoordinates(flip, row, col);\n            p.push();\n            p.noFill();\n            p.stroke(0);\n            p.strokeWeight(2);\n            p.translate(x, y);\n            p.triangle(...triangles.getTrianglePoints(3, Math.PI / 6, 4 * R));\n            p.pop();\n        }\n        \n        function _drawTriangles(permutation, triangle_list, positions) {\n            for (let {id, row, col, flip} of positions) {\n                const {x, y} = triangles.getTriangleCoordinates(flip, row, col);\n                let [a, b, c] = triangle_list[permutation[id][0]];\n                let rot = permutation[id][1];\n                p.push();\n                triangles.drawTriangle([a, b, c], x, y, rot, flip);\n                p.pop();\n            }\n        }\n        \n        return {\n            drawStaticColors: _drawStaticColors,\n            drawFrame: _drawFrame,\n            drawTriangles: _drawTriangles\n        };\n    };\n});")

get_ipython().run_cell_magic('javascript', '', "\ncreateSketchView('StaticBoard', ['staticBoard'], (StaticBoard, model) => {\n    return function(p) {\n        const W = 400;\n        const H = 400;\n        const LEFT = model.get('LEFT');\n        const RIGHT = model.get('RIGHT');\n        const BOTTOM = model.get('BOTTOM');\n        const TRIANGLES = model.get('TRIANGLES');\n        let staticBoard = StaticBoard(p);\n\n        p.setup = function() {\n            p.createCanvas(W, H);\n        }\n\n        p.draw = function () {\n            p.background('#ddd');\n            p.push();\n            p.translate(W / 2, H / 4);\n            let permutation = model.get('permutation');\n            let positions = model.get('positions');\n            staticBoard.drawFrame(positions);\n            staticBoard.drawTriangles(permutation, TRIANGLES, positions);\n            staticBoard.drawStaticColors(LEFT, RIGHT, BOTTOM, positions);\n            p.pop();\n        }\n    };\n})")

from random import sample

class StaticBoard(Board):
    _view_name = Unicode('StaticBoardView').tag(sync=True)
    _view_module = Unicode('StaticBoard').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def shuffle(self):
        self.permutation = sample(self.permutation, N_TRIANGLES) 

testStaticBoard = StaticBoard()
testStaticBoard

testStaticBoard.shuffle()

get_ipython().run_cell_magic('javascript', '', "\nrequire.config({\n    paths: {\n        tween: 'https://cdnjs.cloudflare.com/ajax/libs/tween.js/17.1.1/Tween.min'\n    }\n});")

get_ipython().run_cell_magic('javascript', '', '\ncreateSketchView(\'RotateDemo\', [\'tween\', \'triangles\'], (Tween, Triangles, model) => {\n    const [W, H] = [300, 150];\n    \n    return (p) => {\n        let obj = { angle: 0 };\n        let T = Triangles(p);\n        \n        let tweenGroup = new Tween.Group();\n        let t = new Tween.Tween(obj, tweenGroup)\n                    .to({angle: "+" + (p.TWO_PI / 3)}, 500)\n                    .easing(Tween.Easing.Quadratic.InOut)\n                    .onStart(() => t.running = true)\n                    .onComplete(() => t.running = false)\n        \n        function rotate () {\n            if (t.running) return;\n            t.start();\n        }\n        \n        model.on(\'change:rotations\', rotate);\n\n        p.setup = function(){\n            p.createCanvas(W, H);\n        }\n\n        p.draw = function () {\n            tweenGroup.update();\n            p.background(\'#ddd\');\n            p.translate(W / 3, H / 2);\n            p.push();\n            p.rotate(obj.angle);\n            T.drawTriangle([0, 1, 2], 0);\n            p.pop();\n            p.push();\n            p.translate(W / 3, 0);\n            p.rotate(-obj.angle);\n            T.drawTriangle([3, 4, 5], 0);\n            p.pop();\n        }\n    };\n});')

class RotateDemo(Board):
    _view_name = Unicode('RotateDemoView').tag(sync=True)
    _view_module = Unicode('RotateDemo').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)
    rotations = Int(0).tag(sync=True)

rotate_button = widgets.Button(description="Rotate the triangles")

def on_button_clicked(b):
    rotate_demo.rotations += 1

rotate_button.on_click(on_button_clicked)
rotate_demo = RotateDemo()

widgets.VBox([rotate_button, rotate_demo])

get_ipython().run_cell_magic('javascript', '', "\ndefineModule('animatedBoard',\n             ['settings', 'staticBoard', 'triangles', 'tween', 'lodash'],\n             (Settings, StaticBoard, Triangles, Tween, _) => {\n    \n    const {ANIM_W, ANIM_H, N_TRIANGLES} = Settings;\n    \n    return (p, model) => {\n        let tweenGroup = new Tween.Group();\n        let [it, globalTime, paused] = [0, 0, false];\n        \n        let staticBoard = StaticBoard(p);\n        let triangles = Triangles(p);\n        \n        const TRIANGLES = model.get('TRIANGLES');\n        const LEFT = model.get('LEFT');\n        const RIGHT = model.get('RIGHT');\n        const BOTTOM = model.get('BOTTOM');\n        \n        const states = model.get('states');\n        const positions = model.get('positions');\n        const out = _.range(N_TRIANGLES).map(i => {\n            return {\n                x: ANIM_W * 0.3 + (i % 4) * 100,\n                y: Math.floor(i / 4) * 100,\n                r: 0,\n                f: 0,\n            };\n        })\n        const pos = positions.map(({flip, row, col}) => {\n            const {x, y} = triangles.getTriangleCoordinates(flip, row, col);\n            return {x, y, flip};\n        });\n        \n        // arrays of positions to create the tweens (animations)\n        let [start, end] = [_.cloneDeep(out), _.cloneDeep(out)];\n        \n        // store the triangles moving at each turn to display them on top of the others\n        let moving = [];\n        \n        function findPos(triangleId, state) {\n            return state.findIndex(e => (e && e[0] === triangleId));\n        }\n\n        function transitionState(curr) {\n            let [from, to] = [states[curr-1], states[curr]];\n            to = to || from;\n            _.range(N_TRIANGLES).forEach(i => {\n                \n                const [startPos, endPos] = [findPos(i, from), findPos(i, to)];\n                \n                // on the board\n                if (startPos > -1 && endPos > -1) {\n                    _.assign(start[i], {x: pos[startPos].x, y: pos[startPos].y, r: from[startPos][1], f: pos[startPos].flip});\n                    _.assign(end[i], {x: pos[endPos].x, y: pos[endPos].y, r: to[endPos][1], f: pos[endPos].flip});\n                    return;\n                }\n                \n                // not in current state but in the next one\n                if (startPos < 0 && endPos > -1) {\n                    _.assign(start[i], {x: out[i].x, y: out[i].y, r: out[i].r, f: out[i].f});\n                    _.assign(end[i], {x: pos[endPos].x, y: pos[endPos].y, r: to[endPos][1], f: pos[endPos].flip});\n                    return;\n                }\n                \n                // in current state but not in the next one, bring back\n                if (startPos > -1 && endPos < 0) {\n                    _.assign(start[i], {x: pos[startPos].x, y: pos[startPos].y, r: from[startPos][1], f: pos[startPos].flip});\n                    _.assign(end[i], {x: out[i].x, y: out[i].y, r: out[i].r, f: out[i].f});\n                    return;\n                }\n                \n                // out, no movement\n                if (startPos < 0 && endPos < 0) {\n                    _.assign(start[i], {x: out[i].x, y: out[i].y, r: out[i].r, f: out[i].f});\n                    _.assign(end[i], start[i]);\n                    return;\n                }\n            });\n\n            moving = [];\n            start.forEach((a, i) => {\n                const b = end[i];\n                if (a.x != b.x || a.y != b.y || a.r != b.r) {\n                    moving.push(i);\n                    new Tween.Tween(a, tweenGroup)\n                        .to({x: b.x, y: b.y, r: b.r, f: b.f}, model.get('speed') * 0.8)\n                        .easing(Tween.Easing.Quadratic.InOut)\n                        .start(globalTime)\n                }\n            });   \n        }\n        \n        model.on('change:frame', () => {\n            let frame = model.get('frame');\n            tweenGroup.removeAll();\n            it = Math.max(1, Math.min(frame, states.length - 1));\n            transitionState(it);\n        });\n        \n        return {\n            draw: () => {\n                tweenGroup.update(globalTime);  \n                globalTime += Math.min(1000 / p.frameRate(), 33);\n                \n                p.fill(0);\n                p.textSize(24);\n                p.text(`Iteration: ${it}`, 10, 30);\n                \n                p.translate(ANIM_W / 4, ANIM_H / 4);\n                \n                staticBoard.drawFrame(positions);\n                \n                let allTriangles = _.range(N_TRIANGLES);\n                let staticTriangles = _.difference(allTriangles, moving);\n                [staticTriangles, moving].forEach(bucket => {\n                    bucket.forEach(triangleId => {\n                        const [a, b, c] = TRIANGLES[triangleId];\n                        const {x, y, r, f} = start[triangleId];\n                        p.push();\n                        triangles.drawTriangle([a, b, c], x, y, r, f);\n                        p.pop();\n                    });\n                });\n                staticBoard.drawStaticColors(LEFT, RIGHT, BOTTOM, positions);\n            }\n        };\n    };\n});")

get_ipython().run_cell_magic('javascript', '', "\ncreateSketchView('AnimatedBoard', ['animatedBoard', 'settings'], (AnimatedBoard, Settings, model) => {\n    const {ANIM_W, ANIM_H} = Settings;\n\n    return function(p) {\n        let board = AnimatedBoard(p, model);\n\n        p.setup = function () {\n            p.createCanvas(ANIM_W, ANIM_H);\n        }\n\n        p.draw = function () {\n            p.background('#ddd');\n            board.draw();\n        }\n    };\n});")

import time
from threading import Thread
from traitlets import Bool, observe

class AnimatedBoard(Board):
    _view_name = Unicode('AnimatedBoardView').tag(sync=True)
    _view_module = Unicode('AnimatedBoard').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)
    
    # animation running automatically
    running = Bool(False).tag(sync=True)
    # list of states to animate
    states = List([]).tag(sync=True)
    # current frame (= current iteration)
    frame = Int(0).tag(sync=True)
    # speed of the animation
    speed = Int(1000).tag(sync=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def next_frame(self):
        self.frame = min(self.frame + 1, len(self.states))
        
    def prev_frame(self):
        self.frame = max(0, self.frame - 1)
    
    @observe('running')
    def _on_running_change(self, change):
        if change['new']:
            # start the animation if going from 
            # running == False to running == True
            self._run()
        
    def _run(self):
        def work():
            while self.running and self.frame < len(self.states):
                # update the frame number according
                # to the speed of the animation
                self.frame += 1
                time.sleep(self.speed / 1000)    
            self.running = False

        thread = Thread(target=work)
        thread.start()

animated_board = AnimatedBoard(
    permutation=[[i, 0] for i in range(N_TRIANGLES)],
    states=[
        [None] * 16,
        [[7, 0]] + [None] * 15,
        [[7, 1]] + [None] * 15,
        [[7, 2]] + [None] * 15,
        [[7, 2], [0, 0]] + [None] * 14,
        [[7, 2], [0, 1]] + [None] * 14,
        [[7, 2], [0, 2]] + [None] * 14,
    ]
)
animated_board

from ipywidgets import Layout, Button, Box, VBox, ToggleButton, IntSlider
from traitlets import link


def create_animation(animated_board):
    items_layout = Layout(flex='flex-stretch', width='auto')

    iteration_slider = IntSlider(max=len(animated_board.states), description='Iteration', layout=Layout(width='100%'))
    speed_slider = IntSlider(min=100, max=5000, step=100, description='Speed (ms)')
    prev_button = Button(description='◄ Previous', button_style='info')
    next_button = Button(description='Next ►', button_style='info')
    play_button = ToggleButton(description='Play / Pause', button_style='success', value=False)

    # interactions
    link((play_button, 'value'), (animated_board, 'running'))
    link((iteration_slider, 'value'), (animated_board, 'frame'))
    link((speed_slider, 'value'), (animated_board, 'speed'))
    speed_slider.value = 2500
    
    def on_click_next(b):
        animated_board.next_frame()

    def on_click_prev(b):
        animated_board.prev_frame()

    next_button.on_click(on_click_next)
    prev_button.on_click(on_click_prev)

    box_layout = Layout(display='flex', flex_flow='row', align_items='stretch', width='100%')
    items = [play_button, prev_button, next_button, iteration_slider]
    box = VBox([
        Box(children=items, layout=box_layout), 
        Box(children=(speed_slider,), layout=box_layout),
        animated_board
    ])
    display(box)

create_animation(animated_board)

from copy import deepcopy


class RecursiveSolver(AnimatedBoard):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset_state()
        
    def reset_state(self):
        self.board = [None] * N_TRIANGLES
        self.used = [False] * N_TRIANGLES
        self.logs = [deepcopy(self.board)]
        self.it = 0
        
    def _log(self):
        self.logs.append(deepcopy(self.board))
        
    def _is_valid(self, i):
        ts = self.TRIANGLES
        permutation, positions = self.board, self.positions[i]
        row, col, n_col = positions['row'], positions['col'], positions['n_row']
        triangle_id, triangle_rotation = permutation[i]
            
        # on the left edge
        if col == 0 and ts[triangle_id][2-triangle_rotation] != self.LEFT[row]:
            return False

        # on the right edge
        if col == n_col - 1 and ts[triangle_id][1-triangle_rotation] != self.RIGHT[row]:
            return False

        # on the bottom edge
        if row == 3 and col % 2 == 0 and ts[triangle_id][0-triangle_rotation] != self.BOTTOM[col//2]:
            return False
        
        if col > 0:
            left_pos = i - 1
            left_triangle_id, left_triangle_rotation = permutation[left_pos]

            # normal orientation (facing up)
            if col % 2 == 0 and ts[triangle_id][2-triangle_rotation] != ts[left_triangle_id][2-left_triangle_rotation]:
                return False

            if col % 2 == 1:
                # reverse orientation (facing down)
                # match with left triangle
                if ts[triangle_id][1-triangle_rotation] != ts[left_triangle_id][1-left_triangle_rotation]:
                    return False
                
                # match with line above
                above_pos = i - (n_col - 1)
                above_triangle_id, above_triangle_rotation = permutation[above_pos]
                if ts[triangle_id][0-triangle_rotation] != ts[above_triangle_id][0-above_triangle_rotation]:
                    return False

        return True
    
    def _place(self, i):
        self.it += 1
        if i == N_TRIANGLES:
            return True
        
        for j in range(N_TRIANGLES - 1, -1, -1):
            if self.used[j]:
                # piece number j already used
                continue
                
            self.used[j] = True
            
            for rot in range(3):
                # place the piece on the board
                self.board[i] = (j, rot)
                self._log()

                # stop the recursion if the current configuration
                # is not valid or a solution has been found
                if self._is_valid(i) and self._place(i + 1):
                    return True

            # remove the piece from the board
            self.board[i] = None
            self.used[j] = False
            self._log()
            
        return False

    def solve(self):
        self.reset_state()
        self._place(0)
        return self.board
    
    def found(self):
        return all(slot is not None for slot in self.board)
    
    def save_state(self):
        self.permutation = self.board
        self.states = self.logs

get_ipython().run_cell_magic('time', '', "\nsolver = RecursiveSolver()\nres = solver.solve()\nif solver.found():\n    print('Solution found!')\n    print(f'{len(solver.logs)} steps')\n    solver.save_state()\nelse:\n    print('No solution found')")

solver.permutation

static_solution = StaticBoard(permutation=solver.permutation)
static_solution

create_animation(solver)

from IPython.display import HTML

HTML('<iframe width="800" height="500" src="https://www.youtube.com/embed/lW7mo-9TqEQ?rel=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>')

