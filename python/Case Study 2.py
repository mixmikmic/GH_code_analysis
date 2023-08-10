import vtk

class ssLayer:
    def __init__ (self, interactor, data):
        self.interactor = interactor
        self.data = data
        
        self.properties = vtk.vtkProperty()
        
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.data)
        
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.SetProperty(self.properties)
    
        self.box = vtk.vtkBoxWidget()
        self.box.SetInteractor(self.interactor)
        self.box.SetPlaceFactor(1.00)
        self.box.SetInputData(self.data)
        self.box.InsideOutOn()
        self.box.PlaceWidget()
        self.box.EnabledOff()
        
        self.planes = vtk.vtkPlanes()
        self.box.GetPlanes(self.planes)
        
        self.mapper.SetClippingPlanes(self.planes)
        
        self.box.AddObserver("InteractionEvent", self.boxClip)
        
    def boxClip (self, widget, event_string):
        self.box, self.mapper
        self.box.GetPlanes(self.planes)
        self.mapper.SetClippingPlanes(self.planes)

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
        
interactionStyle = vtk.vtkInteractorStyleTrackballCamera()
renderWindowInteractor.SetInteractorStyle(interactionStyle)

reader = vtk.vtkPolyDataReader()
reader.SetFileName("data/vtk/Subcortical_Structure__1_1.vtk")
reader.Update()
data = reader.GetOutput()

dataLayerA = ssLayer(renderWindowInteractor, data)

renderer.AddActor(dataLayerA.actor)
renderWindow.Render()
renderWindowInteractor.Start()

import webcolors

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
        
interactionStyle = vtk.vtkInteractorStyleTrackballCamera()
renderWindowInteractor.SetInteractorStyle(interactionStyle)

reader = vtk.vtkPolyDataReader()
reader.SetFileName("data/vtk/Subcortical_Structure__1_1.vtk")
reader.Update()
data = reader.GetOutput()

dataLayerA = ssLayer(renderWindowInteractor, data)
dataLayerA.properties.SetColor(webcolors.name_to_rgb('red'))
dataLayerA.properties.SetOpacity(0.4)
dataLayerA.box.EnabledOn()
renderer.AddActor(dataLayerA.actor)

reader = vtk.vtkPolyDataReader()
reader.SetFileName("data/vtk/Subcortical_Structure__2_2.vtk")
reader.Update()
data = reader.GetOutput()

dataLayerB = ssLayer(renderWindowInteractor, data)
dataLayerB.properties.SetColor(webcolors.name_to_rgb('green'))
dataLayerB.properties.SetOpacity(0.4)
renderer.AddActor(dataLayerB.actor)

reader = vtk.vtkPolyDataReader()
reader.SetFileName("data/vtk/Subcortical_Structure__3_3.vtk")
reader.Update()
data = reader.GetOutput()

dataLayerC = ssLayer(renderWindowInteractor, data)
renderer.AddActor(dataLayerC.actor)

renderWindow.Render()
renderWindowInteractor.Start()

