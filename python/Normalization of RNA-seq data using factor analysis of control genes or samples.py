from traitlets.config.manager import BaseJSONConfigManager
path = "/home/saket/anaconda2/etc/jupyter/nbconfig"
cm = BaseJSONConfigManager(config_dir=path)
cm.update('livereveal', {
              'transition': 'zoom',
              'start_slideshow_at': 'selected',
})



