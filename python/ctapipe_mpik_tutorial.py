def simple_generator():
    for x in range(10):
        yield x*2
source = simple_generator()
print(source)
print(type(source))
print()

entry = next(source)
print(entry)
entry = next(source)
print(entry)
print()

source = simple_generator()
for entry in source:
    print(entry)

from ctapipe.io.hessio import hessio_event_source
from ctapipe.utils.datasets import get_datasets_path

filename = get_datasets_path("gamma_test.simtel.gz")
source = hessio_event_source(filename)
event = next(source)

event

event.r0

event.r0.tels_with_data

event.r0.tel[38]

event.inst

from ctapipe.core import Component
from ctapipe.io import CameraGeometry
from ctapipe.visualization import CameraDisplay
from traitlets import CaselessStrEnum
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
from IPython import display

class ImagePlotter(Component):
    name = 'ImagePlotter'
    
    color = CaselessStrEnum(["viridis", "jet", "flag"], "viridis",
                           help="The colormap to use for the intensity in each pixel").tag(config=True)

    def __init__(self, config, tool, **kwargs):
        """
        Plotter for camera images.

        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs
        """
        super().__init__(config=config, parent=tool, **kwargs)

        self.fig = plt.figure(figsize=(16, 7))
        self.ax = self.fig.add_subplot(1, 1, 1)
        
        self.cm = getattr(plt.cm, self.color)
        self.colorbar = None

    def plot(self, event, telid):
        # Get the camera geometry (for plotting the camera)
        geom = CameraGeometry.guess(*event.inst.pixel_pos[telid],
                                    event.inst.optical_foclen[telid])

        # Obtain the photoelectrons in each pixel
        image = event.dl1.tel[telid].image[0]

        # Clear axis
        self.ax.cla()

        # Redraw Camera
        camera = CameraDisplay(geom, image=image, cmap=self.cm, ax=self.ax)

        # Draw colorbar
        if not self.colorbar:
            camera.add_colorbar(ax=self.ax, label='Intensity (p.e.)')
            self.colorbar = camera.colorbar
        else:
            camera.colorbar = self.colorbar
            camera.update(True)

        self.fig.suptitle("Event_index={}  Event_id={}  Telescope={}"
                          .format(event.count, event.r0.event_id, telid))
        
        display.clear_output(wait=True)
        display.display(self.fig)

from ctapipe.core import Tool
from ctapipe.io.eventfilereader import EventFileReaderFactory
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.charge_extractors import ChargeExtractorFactory
from traitlets import Int, Dict, List

class DisplayDL1Calib(Tool):
    name = "DisplayDL1Calib"
    description = "Calibrate dl0 data to dl1, and plot the photoelectron images."

    telescope = Int(None, allow_none=True,
                    help='Telescope to view. Set to None to display all '
                         'telescopes.').tag(config=True)

    aliases = Dict(dict(f='EventFileReaderFactory.input_path',
                        r='EventFileReaderFactory.reader',
                        max_events='EventFileReaderFactory.max_events',
                        extractor='ChargeExtractorFactory.extractor',
                        window_width='ChargeExtractorFactory.window_width',
                        window_start='ChargeExtractorFactory.window_start',
                        window_shift='ChargeExtractorFactory.window_shift',
                        sig_amp_cut_HG='ChargeExtractorFactory.sig_amp_cut_HG',
                        sig_amp_cut_LG='ChargeExtractorFactory.sig_amp_cut_LG',
                        lwt='ChargeExtractorFactory.lwt',
                        clip_amplitude='CameraDL1Calibrator.clip_amplitude',
                        T='DisplayDL1Calib.telescope',
                        color='ImagePlotter.color'
                        ))

    classes = List([EventFileReaderFactory,
                    ChargeExtractorFactory,
                    CameraDL1Calibrator,
                    ImagePlotter
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_reader = None
        self.r1 = None
        self.dl0 = None
        self.dl1 = None
        self.plotter = None

    def setup(self):
        self.log_format = "%(levelname)s: %(message)s [%(name)s.%(funcName)s]"
        kwargs = dict(config=self.config, tool=self)

        reader_factory = EventFileReaderFactory(**kwargs, max_events=3)
        reader_class = reader_factory.get_class()
        self.file_reader = reader_class(**kwargs)

        extractor_factory = ChargeExtractorFactory(**kwargs)
        extractor_class = extractor_factory.get_class()
        extractor = extractor_class(**kwargs)

        r1_factory = CameraR1CalibratorFactory(origin=self.file_reader.origin,
                                               **kwargs)
        r1_class = r1_factory.get_class()
        self.r1 = r1_class(**kwargs)

        self.dl0 = CameraDL0Reducer(**kwargs)

        self.dl1 = CameraDL1Calibrator(extractor=extractor, **kwargs)

        self.plotter = ImagePlotter(**kwargs)

    def start(self):
        source = self.file_reader.read()
        for event in source:
            self.r1.calibrate(event)
            self.dl0.reduce(event)
            self.dl1.calibrate(event)

            tel_list = event.r0.tels_with_data

            if self.telescope:
                if self.telescope not in tel_list:
                    continue
                tel_list = [self.telescope]
            for telid in tel_list:
                self.plotter.plot(event, telid)

    def finish(self):
        pass

exe = DisplayDL1Calib()
exe.run(argv=['-h'])

exe = DisplayDL1Calib()
exe.run(argv=[])

exe = DisplayDL1Calib()
exe.run(argv=['--color', 'flag'])



