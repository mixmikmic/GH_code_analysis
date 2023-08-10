import lsst.daf.persistence as dafPersist
import lsst.afw.geom as afwGeom

import os
#os.environ["SUPRIME_DATA_DIR"] = "/tigress/HSC/HSC"
dataPath = os.path.join(os.environ["SUPRIME_DATA_DIR"], "rerun", "production-20151224")

butler = dafPersist.Butler(dataPath)

for ccd in range(100):
    raw = butler.get('raw', visit=23692, ccd=ccd)
    det = raw.getDetector()
    
    xp, yp = 0, 0
    xfp, yfp = [15e-3*_ for _ in det.getPositionFromPixel(afwGeom.PointD(xp, yp)).getMm()]
    print ccd, det.getId().getName(), xfp, yfp

