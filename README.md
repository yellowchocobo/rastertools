# rastertools

Rastertools is a package that gathers a collection of tools to manipulate rasters and extract metadata from them. A large portion of this package was written for its application in the pre- and post-processing of planetary images so that he can be ingested 



for its use as input in the deep learning 



 use in the automatic detection of boulders on planetary surfaces, and therefore some of the functions are tailored to this study. 

----------------------

**It provides:**

+ functions to read, save, clip rasters.
+ extract metadata.
+ practical tools for converting between grayscale and rgb(a).



**It can be :**

```python
import sys
sys.path.append(<path-to-rastertools-library>)

import numpy as np
import raster
import rasterio as rio

from pathlib import Path

## Create a dummy raster for the rest of the tutorial
array = (np.random.rand(512,512) * 255).astype('uint8')
array = np.expand_dims(array,2) # expand to third dimension

"""
rasterio expect (bands, rows, columns). whereas image processing software like scikit-image, pillow and matplotlib expect (rows, columns, bands). See https://rasterio.readthedocs.io/en/stable/topics/image_processing.html for more details. reshape_as_raster and reshape_as_image from rasterio.plot can be used to change between one and the other (or numpy can be used to swap axes). 
"""
## Create a dummy profile
profile = {'driver': 'GTiff',
 'dtype': 'uint8',
 'nodata': 0,
 'width': 512,
 'height': 512,
 'count': 1,
 'crs': None,
 'tiled': False,
 'interleave': 'band'}

## save dummy raster
r = Path("/home/nilscp/tmp/raster-tmp/dummy_raster.tif")
raster.save_raster(r, array, profile, is_image=True) # convert to raster (bands, rows, columns)

## if you want to extract all of the metadata of a raster 
raster.get_raster_profile(r)

## or other image 


```

