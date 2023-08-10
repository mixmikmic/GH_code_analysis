# Imports for this Python3 notebook
import numpy
import matplotlib.pyplot as plt

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from rios import rat
from rios import ratapplier

from tpot import TPOTRegressor

# Read Biomass library data from the csv file 
fieldBiomass=numpy.loadtxt("biolib_sitelist_auscover.csv",delimiter=',', skiprows=1)

# Open Height Map dataset
keaFile = "alpsbk_aust_y2009_sc5a2.kea"

heightDataset = gdal.Open(keaFile, gdal.GA_Update)


# Set up the reprojection transform from WGS84 (biomass library) to Australian Albers (height data)
source = osr.SpatialReference()
source.ImportFromEPSG(4326)
target = osr.SpatialReference()
target.ImportFromEPSG(3577)
transform = osr.CoordinateTransformation(source, target)


# Open the raster band with the segment IDs
heightBand=heightDataset.GetRasterBand(1)

# Get the Albers to pixel transform
geoTransform=heightDataset.GetGeoTransform()

# Find the segmentID for all the field sites
print("Linking field observations to segment IDs\n")
segmentIDs = []
for record in fieldBiomass:
    # Make up a site OGR point
    site = ogr.Geometry(ogr.wkbPoint)
    site.AddPoint(record[0], record[1])
    # Transform the site to EPSG3577
    site.Transform(transform)
    # Get the pixel location of the site
    mx,my=site.GetX(), site.GetY()  #coord in map units
    #Convert from map to pixel coordinates.
    #Only works for geotransforms with no rotation.
    px = int((mx - geoTransform[0]) / geoTransform[1]) #x pixel
    py = int((my - geoTransform[3]) / geoTransform[5]) #y pixel
    # Extract the segmentID for the location
    segmentIDs.append(heightBand.ReadAsArray(px,py,1,1)[0][0])

# Get the RAT column names
colNames = rat.getColumnNames(heightDataset)

# Select the columns used for the training/prediction
trainingColumns = [5,6,7,8,9,10,15,19,22,23,25,26,27,28,29,30,31,32,33,34]
trainingNames = [colNames[i] for i in trainingColumns]

# Now we have the segmentIDs, pull the image data from the RAT that corresponds to the segment IDs
imageData = []
# Iterate for all the RAT columns
for name in trainingNames:
    print("Extracting sites from " + name)
    # Extract the array of values corresponding to the field site segments
    imageData.append(rat.readColumnFromBand(heightBand,name).astype('float')[segmentIDs])

# Convert the list of arrays to an array
imageData = numpy.transpose(numpy.array(imageData))

# Remove nodata from the couple of segments too small to get statistics in the image data
goodDataIDX = imageData.min(axis=1)>0
imageData = imageData[goodDataIDX]
fieldBiomass = fieldBiomass[goodDataIDX]


print("\nTraining data has %d observations and %d columns" % imageData.shape)

# Total amount of time we allow for the training
optTime = 600

# Number of CPUs to use for training and cross validation
nCPUs = 4

# What function to minimise
scoring = 'mean_squared_error'

# Number of subsamples from the Biomass Library for model training
nSubsets = 9999

# We select the tb_drymass_ha column to train on
totalBiomass = fieldBiomass[:,10]
# This is the standard error of the site level estimates
totalBiomassSE = fieldBiomass[:,11]

# Select a subsample to improve the model search speed
subSample = numpy.random.choice(len(totalBiomass),nSubsets,replace=False)
biomass=totalBiomass[subSample]
biomassSE=totalBiomassSE[subSample]
trainData=imageData[subSample]

# Use the proportion of the error in the estimates as fitting weights
biomassWeights=biomass/biomassSE

# Setup the TPOT regression options
tpot = TPOTRegressor(max_time_mins=optTime,  
                     n_jobs = nCPUs, 
                     scoring=scoring, 
                     verbosity=2, 
                     cv=10, 
                     max_eval_time_mins=1,
                     population_size=100)

# Start testing models using 10 fold cross validation and 100 models per generation
tpot.fit(trainData, biomass, sample_weight=biomassWeights)

# Export the best model to a file
tpot.export('tpot_biomass_pipeline.py')

# Build the biomass predictive model
biomassModel = tpot._fitted_pipeline.fit(imageData, totalBiomass)

# Predict the full dataset
predBiomass = biomassModel.predict(imageData)

# Print some RMSE Statistics for various ranges
print("\nTotal RMSE = %f\n" % numpy.sqrt(numpy.mean((totalBiomass-predBiomass)**2)))
stopPoints=[0,100,500,1000,2000,5000,10000]
print("Start"," Stop","Count"," RMSE")
for i in range(len(stopPoints)-1):
    idx=numpy.logical_and(totalBiomass>stopPoints[i],totalBiomass<stopPoints[i+1])
    rmse=numpy.sqrt(numpy.mean((totalBiomass[idx]-predBiomass[idx])**2))
    print('{0:5d} {1:5d} {2:5d} {3:5.0f}'.format(stopPoints[i],stopPoints[i+1],idx.sum(),rmse))



# Plot the Output in a LogLog figure
fig = plt.figure(figsize=(10,10))
plt.loglog(totalBiomass,predBiomass, 'g.',[10,10000], [10,10000],'r-')
plt.xlabel('Observed (Mg/ha)', fontsize=18)
plt.ylabel('Predicted (Mg/ha)', fontsize=18)
plt.title('Total Biomass Estimate', fontsize=32)
plt.xlim([10,10000])
plt.ylim([10,10000])
plt.grid(which='minor', alpha=0.4)                                                
plt.grid(which='major', alpha=0.8)                                                

get_ipython().run_cell_magic('time', '', '\ndef _ratapplier_calc_biomass(info, inputs, outputs):\n    """\n    Calculate Biomass from RAT.\n    Called by ratapplier below\n    """\n    ratArray = []\n    # Iterate for all the RAT columns\n    for name in trainingNames:\n        # Extract the array of values corresponding to the field site segments\n        ratArray.append(getattr(inputs.inrat, name).astype(\'float\'))\n\n    # Convert the list of arrays to an array\n    ratArray = numpy.transpose(numpy.array(ratArray))\n    # Predict Biomass\n    biomass = biomassModel.predict(ratArray)\n    # Make the weird inputs nodata\n    biomass[ratArray.min(axis=1) < numpy.finfo(numpy.float32).eps] = 0\n\n    # Save to \'totalBiomass\' column (will create if doesn\'t exist)\n    setattr(outputs.outrat,"totalBiomass", biomass)\n\n\n# Set up ratapplier for input / output\ninFile = ratapplier.RatAssociations()\noutFile = ratapplier.RatAssociations()\n\n# Pass in clumps file for the input and output as we\'ll be updating the existing RAT\ninFile.inrat = ratapplier.RatHandle(keaFile)\noutFile.outrat = ratapplier.RatHandle(keaFile)\n\n# Apply function to all rows in chunks\nratapplier.apply(_ratapplier_calc_biomass, inFile, outFile)\n    \n    ')

# Additional Imports
import rsgislib
from rsgislib.rastergis import exportCol2GDALImage

# Setup the export parameters
outimage='/home/jovyan/work/Temp/totalBiomass.tif'
gdalformat = 'GTIFF'
datatype = rsgislib.TYPE_16UINT
field = 'totalBiomass'

# Run the export
exportCol2GDALImage(keaFile, outimage, gdalformat, datatype, field)

