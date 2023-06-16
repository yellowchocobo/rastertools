# rastertools

(last updated the 13th of June 2023).

Rastertools is a package that gathers a collection of tools to manipulate rasters and extract metadata from them. A large portion of this package was written for  the pre- and post-processing of planetary images so that it can be easily ingested in deep learning algorithms. Because of that, some of the functions are a bit, sometimes, too specific and repetitive (sorry for that!). I will try over time to improve this GitHub repository. Please contribute if you are interested. 

This GitHub repository is written following the functional programming paradigm. 

## To do

---

- [ ] Add generate graticule from raster.
- [ ] Move some of the create_annotations functions to rastertools, and some to shptools. 
- [ ] Merge `clip_advanced`, `clip` and `clip_from_bbox` into one single clip function. 
- [ ] Define coordinate systems (in `crs.py`) with different lonlat range (-180, 180) and (0, 360). Right now, there are some issues due to this problem. 
- [ ] `raster.projection` is currently not working for projection from Equirectangular to Moon2000. 

## Installation

---

