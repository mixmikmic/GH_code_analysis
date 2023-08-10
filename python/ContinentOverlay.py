overlay = Image.open("ContinentOverlay.png")
fitOverlay = ImageOps.fit(overlay, MapClimate.size, Image.BILINEAR)
MapClimate.paste(fitOverlay, (0, 0), fitOverlay)

