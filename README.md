# rastertools

(last updated the 13th of June 2023).

Rastertools is a package that gathers a collection of tools to manipulate rasters and extract metadata from them. A large portion of this package was written for  the pre- and post-processing of planetary images so that it can be easily ingested in deep learning algorithms. Because of that, some of the functions are a bit, sometimes, too specific and repetitive (sorry for that!). I will try over time to improve this GitHub repository. Please contribute if you are interested. 

This GitHub repository is written following the functional programming paradigm. 

## To do

---

- [ ] Add generate graticule from raster.
- [ ] Delete generalutils (keep listLayers, move rasterize_gdf and random_point_in_polys but move to shptools).
- [ ] Delete geobox (I don't think there is anything of interest there).
- [ ] Move some of the create_annotations functions to rastertools, and some to shptools. 
- [ ] Create metadata.py
- [ ] Create convert.py 
- [ ] Merge `clip_advanced`, `clip` and `clip_from_bbox` into one single clip function. 

## Functionalities

----------------------

+ functions to manipulate rasters, e.g., read, save, clip and tile rasters (`./raster.py`) 
+ extract metadata (`./metadata.py`) 
+ practical tools for converting between grayscale and rgb(a) (`./convert.py`) 
+ Include basic coordinate systems for the Moon and Mars (`./crs.py`) 

## Getting Started

---

I will now walk you through some of the most important functionalities. 

## Manipulation of rasters

```python
import sys
sys.path.append(<path-to-rastertools-library>)

import geopandas as gpd
import numpy as np
import raster
import rasterio as rio

from pathlib import Path
from affine import Affine

"""
Note: rasterio expect (bands, rows, columns). whereas image processing software like scikit-image, pillow and matplotlib expect (rows, columns, bands). See https://rasterio.readthedocs.io/en/stable/topics/image_processing.html for more details. reshape_as_raster and reshape_as_image from rasterio.plot can be used to change between one and the other (or numpy can be used to swap axes). 
"""
```

### Reading a raster

```python
r = Path("/media/nilscp/pampa/BOULDERING/completed_mapping/moon/crater0394/raster/M1221383405.tif")
```



In order to read a raster, you can use the `read_raster` function:

```python
array = raster.read_raster(r) # to read the whole raster with all bands
```

(*raster in QGIS below*)

![image-20230613160540583](/home/nilscp/.config/Typora/typora-user-images/image-20230613160540583.png)

But you can include options if needed, such as selecting only the `bands` you are interested in:

```python
array = raster.read_raster(r, bands=[1,2,3]) # bands starting from 1, in our case, the example raster has only one band...
array = raster.read_raster(r, bands=[1]) 
```

You can also choose if you want to have your array output with the rasterio format (bands, rows, columns) or the image format  (rows, columns, bands) with the `as_image` flag. 

```python
# image format
array = raster.read_raster(r, bands=[1], as_image=True) 
array.shape
(55680, 12816, 1)

# rasterio format 
array = raster.read_raster(r, bands=[1], as_image=False) 
array.shape
(1, 55680, 12816)
```

If you don't want to load the whole raster, you can specify the bounding box of a portion of the image, and only the data within this portion will be loaded. Let's say you are only interested in the area around the very fresh impact crater in the middle of the original raster, and we have a polygon shapefile that constrain the boundary. 

```python
poly = Path("/home/nilscp/tmp/raster-tmp/ROM.shp")
gdf_poly = gpd.read_file(poly) # load a rectangular box covering the fresh impact crater
bounds_poly = list(gdf_poly.bounds.values[0])
array = raster.read_raster(r, bands=[1], bbox=bounds_poly, as_image=True) 
array.shape
(4500, 4194, 1)
```

If you want to save it as a new raster to avoid the use of the large original raster, which may slow down your computer, you can "clip" your raster. In order to save the new raster, the metadata (see Metadata section at the bottom of this file) of the new raster need to be created. We can use:

```python
original_raster_profile = raster.get_raster_profile(r)
new_raster_profile = original_raster_profile.copy() 
```

The width, height and transform metadata need to be updated.

```python
new_raster_profile["transform"]
Affine(0.6339945614856195, 0.0, 10559291.7031,0.0, -0.6339945671695403, -428407.4778)
```

See https://en.wikipedia.org/wiki/Affine_transformation for more info about Affine transformation or write `Affine?`. But long story short, you need to specify the following: 

```python
raster_resolution = raster.get_raster_resolution(r)[0]
# Affine(raster_resolution, 0.0, xmin, -raster_resolution, ymax) # xmin, ymax corresponds to the top left corner of the image
new_transform = Affine(raster_resolution, 0.0, bounds_poly[0], 0.0, -raster_resolution, bounds_poly[3])
```

Let's update the metadata:

```python
new_raster_profile.update({
         "width": array.shape[1],
         "height": array.shape[0],
         "transform": new_transform})
```

### Save the new raster

```python
out_raster = Path("/home/nilscp/tmp/raster-tmp/crater.tif")
raster.save_raster(out_raster, array, new_raster_profile, is_image=True)
```

(*new raster in QGIS*) 

![image-20230613165613689](/home/nilscp/.config/Typora/typora-user-images/image-20230613165613689.png)

NB! Using this workflow, was only for tutorial purpose as it introduces the user to basic functions such as `read_raster` and `save_raster` and the use of metadata-related functions. This pipeline actually introduce some shifts between the original and the new raster because of the coordinate of the top left extent of the polygon shapefile do not fall on the top left edge of a pixel. 

### Clipping

For a correct behavior for the clipping of rasters, please use the following function:

``` python
out_raster_good = Path("/home/nilscp/tmp/raster-tmp/crater2.tif")
raster.clip_from_bbox(r, bounds_poly, out_raster_good)
```

### Reprojection

### Resampling

### Tiling

### Extract footprint and true footprint

### Polygonize









## Metadata

```python
## if you want to extract all of the metadata of a raster 
raster.get_raster_profile(r)
```

You should get a dictionary as output with all of the metadata:

```python
{'driver': 'GTiff', 
 'dtype': 'uint8', 
 'nodata': 0.0, 
 'width': 12816, 
 'height': 55680, 
 'count': 1, 'crs': 
 CRS.from_wkt('PROJCS["EQUIRECTANGULAR MOON",GEOGCS["GCS_MOON",DATUM["D_MOON",SPHEROID["MOON_localRadius",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",-14.59],PARAMETER["central_meridian",-10.48],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'), 
 'transform': Affine(0.6339945614856195, 0.0, 10559291.7031, 0.0, -0.6339945671695403, -428407.4778), 
 'tiled': False, 'interleave': 'band'}
```

Or you can extract directly specific metadata of interest with the functions `get_raster_crs` , `get_raster_resolution`, `get_raster_types`, `get_raster_height` and so forth... (I let you have a look at `metadata.py`) For example, if you want to quickly get the raster resolution. 

```python
raster.get_raster_resolution(r)
```

Be careful as it returns the resolution along the x- and y-axes 

```python
(0.6339945614856195, 0.6339945671695403)
```



