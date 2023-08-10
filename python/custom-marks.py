import io
import xml.etree.ElementTree as xml

import numpy

import toyplot.html
import toyplot.mark

import logging
logging.basicConfig(level=logging.DEBUG)

toyplot.log.setLevel(logging.DEBUG)

# Custom marks must derive from toyplot.mark.Mark
class Circle(toyplot.mark.Mark):
    def __init__(self, x, y, radius):
        super(Circle, self).__init__()
        self._coordinates = {
            "x": x,
            "y": y,
        }
        self._radius = radius
        
    # Marks should return their domain (their minimum and maximum values along a given axis).
    def domain(self, axis):
        return numpy.array(self._coordinates[axis]), numpy.array(self._coordinates[axis])
    
    # Marks should return their extents, which are a combination of domain coordinates and
    # range extents relative to those coordinates.  This is so the domain and range of an axis can
    # be adjusted to account for the size of the mark (in range coordinates)
    def extents(self, axes):
        coordinates = tuple([numpy.array([self._coordinates[axis]]) for axis in axes])
        extents = (
            numpy.array([-self._radius]),
            numpy.array([self._radius]),
            numpy.array([-self._radius]),
            numpy.array([self._radius]),
            )
        return coordinates, extents

# Custom marks must define a _render() function and register it using dispatch() so it can be
# called at render time.  Note that _render() is registered for a given combination of coordinate
# system and mark.  This allows marks to adapt their visual representation to the coordinate
# system (for example, a scatterplot mark would be rendered using lines if it was part of a
# hypothetical parallel coordinate system).
@toyplot.html.dispatch(toyplot.coordinates.Cartesian, Circle, toyplot.html.RenderContext)
def _render(axes, mark, context):
    x = axes.project("x", mark._coordinates["x"])
    y = axes.project("y", mark._coordinates["y"])
    xml.SubElement(
        context.parent,
        "circle",
        id=context.get_id(mark),
        cx=repr(x),
        cy=repr(y),
        r=str(mark._radius),
        stroke="black",
        fill="lightgray",
        )
    
    circle_ml = "<circle x=%s y=%s radius=%s></circle>" % (mark._coordinates["x"], mark._coordinates["y"], mark._radius)
    
    context.require(
        dependencies=["toyplot/menus/context", "toyplot/io"],
        arguments=[context.get_id(mark), circle_ml],
        code="""function(context_menu, io, mark_id, content)
        {
            var owner = document.querySelector("#" + mark_id);
            function show_item(e)
            {
                return owner.contains(e.target);
            }

            function choose_item()
            {
                io.save_file("text/xml+cml", "utf-8", content, "test.cml");
            }
            context_menu.add_item("Save circle as CircleML", show_item, choose_item);
        }""",
    )

canvas = toyplot.Canvas()
axes = canvas.cartesian()
axes.scatterplot(numpy.linspace(2, 3, 1000), numpy.random.normal(2.5, 0.1, size=1000))
axes.add_mark(Circle(2, 3, 20))
axes.add_mark(Circle(3, 2, 50));

class DOTOutput(toyplot.mark.Mark):
    def __init__(self, graph):
        super(DOTOutput, self).__init__()
        self._graph = graph
        
@toyplot.html.dispatch(toyplot.coordinates.Cartesian, DOTOutput, toyplot.html.RenderContext)
def _render(axes, mark, context):
    context.require(
        dependencies=["toyplot/menus/context", "toyplot/tables", "toyplot/io"],
        arguments=[context.get_id(mark._graph)],
        code="""function(context_menu, tables, io, mark_id)
        {
            var owner = document.querySelector("#" + mark_id);
            function show_item(e)
            {
                return owner.contains(e.target);
            }

            function choose_item()
            {
                var edges = tables.get(mark_id, "edge_data");
                var source = null;
                var target = null;
                for(var i = 0; i != edges.names.length; ++i)
                {
                    if(edges.names[i] == "source")
                        source = edges.columns[i];
                    else if(edges.names[i] == "target")
                        target = edges.columns[i];
                }
                var content = "digraph toyplot {\\n";
                for(var i = 0; i != source.length; ++i)
                    content += "  \\"" + source[i] + "\\" -> \\"" + target[i] + "\\"\\n";
                content += "}\\n";
                io.save_file("text/vnd.graphviz", "utf-8", content, "toyplot.dot");
            }
            context_menu.add_item("Save graph as GraphViz .dot", show_item, choose_item);
        }""",
    )    

# Random graph
vertices = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
edges = numpy.random.choice(vertices, (20, 2))

canvas, axes, mark = toyplot.graph(edges)
axes.add_mark(DOTOutput(mark));

