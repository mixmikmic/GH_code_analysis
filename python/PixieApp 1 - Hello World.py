# uncomment the following line to install/upgrade the PixieDust library
# ! pip install pixiedust --user --upgrade
import pixiedust

from pixiedust.display.app import *
@PixieApp
class HelloWorldPixieApp:
    @route()
    def main(self):
        return"""
            <input pd_options="clicked=true" type="button" value="Click Me">
        """
    @route(clicked="true")
    def _clicked(self):
        return """
            <input pd_options="clicked=false" type="button" value="You Clicked, Now Go back">
        """
#run the app
HelloWorldPixieApp().run(runInDialog='false')

from pixiedust.display.app import *
@PixieApp
class HelloWorldPixieAppWithData:
    @route()
    def main(self):
        return"""
        <div class="row">
           <div class="col-sm-2">
              <input pd_options="handlerId=dataframe"
                     pd_entity
                     pd_target="target{{prefix}}"
                     type="button" value="Preview Data">
           </div>
           <div class="col-sm-10" id="target{{prefix}}"/>
        </div>
        """
#Create dataframe
df = SQLContext(sc).createDataFrame(
[(2010, 'Camping Equipment', 3, 200),(2010, 'Camping Equipment', 10, 200),(2010, 'Golf Equipment', 1, 240),
 (2010, 'Mountaineering Equipment', 1, 348),(2010, 'Outdoor Protection',2,200),(2010, 'Personal Accessories', 2, 200),
 (2011, 'Camping Equipment', 4, 489),(2011, 'Golf Equipment', 5, 234),(2011, 'Mountaineering Equipment',2, 123),
 (2011, 'Outdoor Protection', 4, 654),(2011, 'Personal Accessories', 2, 234),(2012, 'Camping Equipment', 5, 876),
 (2012, 'Golf Equipment', 5, 200),(2012, 'Mountaineering Equipment', 3, 156),(2012, 'Outdoor Protection', 5, 200),
 (2012, 'Personal Accessories', 3, 345),(2013, 'Camping Equipment', 8, 987),(2013, 'Golf Equipment', 5, 434),
 (2013, 'Mountaineering Equipment', 3, 278),(2013, 'Outdoor Protection', 8, 134),(2013,'Personal Accessories',4, 200)],
 ["year","zone","unique_customers", "revenue"])

#run the app
HelloWorldPixieAppWithData().run(df, runInDialog='false')

