get_ipython().run_cell_magic('writefile', 'f1.template', '<!DOCTYPE html>\n<html>\n    <meta http-equiv="Content-Type" content="text/html;charset=utf-8"/>\n    <script type="text/javascript" src="https://mbostock.github.io/d3/talk/20111018/d3/d3.js"></script>\n    <script type="text/javascript" src="https://mbostock.github.io/d3/talk/20111018/d3/d3.geom.js"></script>\n    <script type="text/javascript" src="https://mbostock.github.io/d3/talk/20111018/d3/d3.layout.js"></script>\n    <style type="text/css">\n\ncircle {\n  stroke: #000;\n  stroke-opacity: .5;\n}\n\n    </style>\n  <body>\n    <div id="body">\n    <script type="text/javascript">\n\nvar w = {width},\n    h = {height};\n\nvar nodes = d3.range({ball_count}).map(function() { return {radius: Math.random() * {rad_fac} + {rad_min}}; }),\n    color = d3.scale.category10();\n\nvar force = d3.layout.force()\n    .gravity(0.1)\n    .charge(function(d, i) { return i ? 0 : -2000; })\n    .nodes(nodes)\n    .size([w, h]);\n\nvar root = nodes[0];\nroot.radius = 0;\nroot.fixed = true;\n\nforce.start();\n\nvar svg = d3.select("#body").append("svg:svg")\n    .attr("width", w)\n    .attr("height", h);\n\nsvg.selectAll("circle")\n    .data(nodes.slice(1))\n  .enter().append("svg:circle")\n    .attr("r", function(d) { return d.radius - 2; })\n    .style("fill", function(d, i) { return color(i % {color_count}); });\n\nforce.on("tick", function(e) {\n  var q = d3.geom.quadtree(nodes),\n      i = 0,\n      n = nodes.length;\n\n  while (++i < n) {\n    q.visit(collide(nodes[i]));\n  }\n\n  svg.selectAll("circle")\n      .attr("cx", function(d) { return d.x; })\n      .attr("cy", function(d) { return d.y; });\n});\n\nsvg.on("mousemove", function() {\n  var p1 = d3.svg.mouse(this);\n  root.px = p1[0];\n  root.py = p1[1];\n  force.resume();\n});\n\nfunction collide(node) {\n  var r = node.radius + 16,\n      nx1 = node.x - r,\n      nx2 = node.x + r,\n      ny1 = node.y - r,\n      ny2 = node.y + r;\n  return function(quad, x1, y1, x2, y2) {\n    if (quad.point && (quad.point !== node)) {\n      var x = node.x - quad.point.x,\n          y = node.y - quad.point.y,\n          l = Math.sqrt(x * x + y * y),\n          r = node.radius + quad.point.radius;\n      if (l < r) {\n        l = (l - r) / l * .5;\n        node.x -= x *= l;\n        node.y -= y *= l;\n        quad.point.x += x;\n        quad.point.y += y;\n      }\n    }\n    return x1 > nx2\n        || x2 < nx1\n        || y1 > ny2\n        || y2 < ny1;\n  };\n}\n\n    </script>\n  </body>\n</html>')

from IPython.display import IFrame
import re

def replace_all(txt,d):
    rep = dict((re.escape('{'+k+'}'), str(v)) for k, v in d.items())
    pattern = re.compile("|".join(rep.keys()))
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], txt)    

count=0
def serve_html(s,w,h):
    import os
    global count
    count+=1
    fn= '__tmp'+str(os.getpid())+'_'+str(count)+'.html'
    with open(fn,'w') as f:
        f.write(s)
    return IFrame('files/'+fn,w,h)

def f1(w=500,h=400,ball_count=150,rad_min=2,rad_fac=11,color_count=3):
    d={
       'width'      :w,
       'height'     :h,
       'ball_count' :ball_count,
       'rad_min'    :rad_min,
       'rad_fac'    :rad_fac,
       'color_count':color_count
       }
    with open('f1.template','r') as f:
        s=f.read()
    s= replace_all(s,d)        
    return serve_html(s,w+30,h+30)

f1(ball_count=50, color_count=17, rad_fac=10, rad_min=3, w=600)



