"""Utils and commonly used routines for working with GDAL/OGR data sets."""

import os
import math
import numpy as np
import pathlib
import pyproj
import rasterio as rio
import sys
import geopandas as gpd
import json
import fiona
import glob


sys.path.append("/home/nilscp/GIT/rastertools")
import crs

from itertools import product

import rasterio.mask as mask

from affine import Affine
from pathlib import Path
from PIL import Image
from rasterio import features
from rasterio.plot import reshape_as_raster, reshape_as_image
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import fourier_shift
from shapely.geometry import Polygon, box
from shapely.affinity import translate


def normalize_uint8(in_raster, out_raster):
    """
    Min-Max Normalization and conversion to byte
    equivalent to
    gdal_translate -ot Byte -scale -a_nodata 0 src_dataset dst_dataset
    but it is better to drop the no_data equal to 0.
    :param in_raster:
    :param out_raster:
    :return:
    """
    array = read_raster(in_raster)
    out_meta = get_raster_profile(in_raster)

    # for RED_MRDR (I don't really like those min-max stretching...)
    # a very dark image could end up having a weird stretching
    # highest value encountered 0.155219, lowest value 0.0
    if array.dtype == np.float32: # nan = -3.4028226550889045e+38
        array[array < 0] = 0.0
    array_norm = (array - array.min()) / (array.max() - array.min())
    array_uint8 = np.round(array_norm * 255, decimals=0).astype('uint8')


    out_meta.update({
             "count": 1,
             "dtype": "uint8",
             "nodata": 0})

    save_raster(out_raster, array_uint8, out_meta, False)


def rgb_to_grayscale(in_raster, out_raster):

    """
    Takes RGB or RGBA raster and convert it to grayscale.

    :param in_raster:
    :param out_raster:
    :return:
    """

    in_raster = Path(in_raster)
    array = Image.open(in_raster).convert("L")
    array = np.array(array)
    array = np.expand_dims(array, axis=0)

    if out_raster:
        None
    else:
        out_raster = in_raster.with_name(in_raster.stem + "_grayscale" + in_raster.suffix)

    out_meta = get_raster_profile(in_raster)
    out_meta.update({"count": 1})

    with rio.open(out_raster, "w", **out_meta) as dst:
        dst.write(array)

def rgb_fake_batch(folder):
    folder = Path(folder)
    for in_raster in folder.glob('*.png'):
        fake_RGB(in_raster)

def tiff_to_png_batch(folder, is_hirise=False):
    folder = Path(folder)
    for in_raster in folder.glob('*.tif'):
        tiff_to_png(in_raster, is_hirise)

def tiff_to_png(in_raster, out_png=False, is_hirise=False):
    in_raster = Path(in_raster)
    png = in_raster.with_name(in_raster.name.split(".tif")[0] + ".png")
    array = read_raster(in_raster, as_image=True)
    h, w, c = array.shape
    array = array.reshape((h,w))
    if is_hirise:
        array = np.round(array * (255.0 / 1023.0)).astype('uint8')
    im = Image.fromarray(array)

    if out_png:
        None
    else:
        out_png = in_raster.with_name(in_raster.stem + "_fakergb" + in_raster.suffix)
    im.save(png)

def get_raster_crs(in_raster):
    with rio.open(in_raster) as rio_dataset:
        crs = rio_dataset.crs
    return crs

def get_raster_resolution(in_raster):
    with rio.open(in_raster) as rio_dataset:
        res = rio_dataset.res
    return res

def get_raster_types(in_raster):
    with rio.open(in_raster) as rio_dataset:
        dtypes = rio_dataset.dtypes
    return dtypes

def get_raster_height(in_raster):
    with rio.open(in_raster) as rio_dataset:
        height = rio_dataset.height
    return height

def get_raster_width(in_raster):
    with rio.open(in_raster) as rio_dataset:
        width = rio_dataset.width
    return width

def get_raster_bbox(in_raster):
    with rio.open(in_raster) as rio_dataset:
        bbox =rio_dataset.bounds
    return list(bbox)

def get_raster_shape(in_raster):
    with rio.open(in_raster) as rio_dataset:
        shape = rio_dataset.shape
    return shape

def get_raster_nbands(in_raster):
    with rio.open(in_raster) as rio_dataset:
        count = rio_dataset.count
    return count

def get_raster_profile(in_raster):
    with rio.open(in_raster) as rio_dataset:
        profile = rio_dataset.profile.copy()
    return profile

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def parse_srid(rio_dataset):    
    """
    Parse the SRID (EPSG code) from a raster open with rasterio    
    :param rio_dataset: geodataframe with polygons    
    :returns: coordinate systems as EPSG code (integer)
    
    Examples:
        with rio.open(raster) as rio_dataset:
            crs = parse_srid(rio_dataset)
            
        # UTM 32 will for example return
        32632
    """
    return rio_dataset.crs.to_epsg()


def srid_to_wkt(crs_epsg):
    """Convert the SRID (EPSG) to GDAL-compatible projection metadata.
    :param crs_epsg: coordinate systems as EPSG code (integer)  
    :returns: coordinate systems as wkt string
    
    Examples:
        with rio.open(raster) as rio_dataset:
            crs_wkt = srid_to_wkt(parse_srid(rio_dataset))
            
        # UTM 32 will for example return
        'PROJCS["WGS 84 / UTM zone 32N",
        GEOGCS["WGS 84",DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
        AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
        AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]],
        PROJECTION["Transverse_Mercator"],
        PARAMETER["latitude_of_origin",0],
        PARAMETER["central_meridian",9],
        PARAMETER["scale_factor",0.9996],
        PARAMETER["false_easting",500000],
        PARAMETER["false_northing",0],
        UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
        AXIS["Easting",EAST],
        AXIS["Northing",NORTH],
        AUTHORITY["EPSG","32632"]]'
    """
    srs = rio.crs.CRS.from_epsg(crs_epsg)
    return srs.to_wkt()


def pix2world(rio_dataset, xy_pix):
    """Convert rasterio dataset pixel coordinates to world coordinates (WGS84).
    :param rio_dataset: rasterio dataset
    :param xy_pix: numpy array [x, y] pixel coordinates
    :returns: lon and lat in Moon coordinates (Moon2000)
    
    Examples:
        with rio.open(raster) as rio_dataset:
            xy_pix = (1000,1000)
            lon, lat = pix2world(rio_dataset, xy_pix)
    """
    row, col = xy_pix
    
    # coordinate system rasterio dataset
    crs_proj4_raster = rio_dataset.crs.to_proj4()
    crs_raster = pyproj.Proj(crs_proj4_raster)
    
    # get xy coordinates in the coordinate system of the rasterio dataset
    x, y = rio_dataset.xy(row, col)
        
    # World geodetic system 84 (WGS84)
    #wgs_crs = pyproj.Proj(init='epsg:4326')

    # Moon2000
    Moon2000_crs = pyproj.Proj('+proj=longlat +a=1737400 +b=1737400 +no_defs')
    
    lon,lat = pyproj.transform(crs_raster, Moon2000_crs, x, y)
        
    return ((lon,lat))


def world2pix(rio_dataset, xy_world):
    """Convert from world coordinates to pixel coordinates (based on coordinates
     of the rasterio dataset).
    :param rio_dataset:  rasterio dataset
    :param xy_world: numpy array [x, y] world coordinates in Moon2000
    :returns: numpy array [x, y] pixel coordinates
    
    Examples:
        with rio.open(raster) as rio_dataset:
            xy_pix = (11.05,61.22)
            row, col = world2pix(rio_dataset, xy_world)
    """
    
    lon, lat = xy_world
    
    # World geodetic system 84 (WGS84)
    #wgs_crs = pyproj.Proj(init='epsg:4326')

    # Moon2000
    Moon2000_crs = pyproj.Proj('+proj=longlat +a=1737400 +b=1737400 +no_defs')
    
    # coordinate system rasterio dataset
    crs_proj4_raster = rio_dataset.crs.to_proj4()
    crs_raster = pyproj.Proj(crs_proj4_raster)
    
    east,north = pyproj.transform(Moon2000_crs, crs_raster, lon, lat)
    
    # get row and cols with the help of rasterio
    row, col = rio_dataset.index(east, north)
    
    return ((row,col))

def world2crs(rio_dataset, xy_world):
    """Convert from world coordinates to rio_dataset coordinates
    (based on coordinates of the rasterio dataset).
    :param rio_dataset:  rasterio dataset
    :param xy_world: numpy array [x, y] world coordinates in Moon2000
    :returns: numpy array [x, y] pixel coordinates

    Examples:
        with rio.open(raster) as rio_dataset:
            xy_pix = (11.05,61.22)
            x, y = world2crs(rio_dataset, xy_world)
    """

    lon, lat = xy_world

    # World geodetic system 84 (WGS84)
    # wgs_crs = pyproj.Proj(init='epsg:4326')

    # Moon2000
    Moon2000_crs = pyproj.Proj('+proj=longlat +a=1737400 +b=1737400 +no_defs')

    # coordinate system rasterio dataset
    crs_proj4_raster = rio_dataset.crs.to_proj4()
    crs_raster = pyproj.Proj(crs_proj4_raster)

    east, north = pyproj.transform(Moon2000_crs, crs_raster, lon, lat)

    return ((east, north))

def bbox_world2crs(rio_dataset, bbox_xy_world):
    """Convert bbox from world coordinates to bbox with pixel coordinates
    (based on coordinates of the rasterio dataset).
    :param rio_dataset:  rasterio dataset
    :param bbox_xy_world: list [lon_min, lat_min, lon_max, lat_max] in world coordinates Moon2000
    :returns: numpy array [x, y] pixel coordinates

    Examples:
        with rio.open(raster) as rio_dataset:
            xy_pix = (11.05,61.22, 11.10, 61.32)
            (lon_min, lat_min, lon_max, lat_max) = world2pix(rio_dataset, xy_world)
    """
    lon_min, lat_min, lon_max, lat_max = bbox_xy_world

    x_min, y_min = world2crs(rio_dataset, [lon_min, lat_min])
    x_max, y_max = world2crs(rio_dataset, [lon_max, lat_max])

    return [x_min, y_min, x_max, y_max]


def boundary(rio_dataset):
    """Get boundary of gdal tif.
    :param rio_dataset: rasterio dataset
    :returns: bounding box of the raster (rasterio dataset)
    """
    
    ulx, xres, _, uly, _, yres = rio_dataset.transform.to_gdal()
    lrx = ulx + (rio_dataset.width * xres)
    lry = uly + (rio_dataset.height * yres)
    
    bbox = [ulx, lry, lrx, uly]
    
    return bbox

def get_extent(rio_dataset, bbox):
    
    """get extent (column_upper_left, row_upper_left, ncols, nrows) from a 
     bounding box. The bounding box must have the same coordinate systems and
     within the bounds of the rio_dataset raster.
    :param rio_dataset: rio dataset (e.g., raster)
    :param bbox: bounding box of the raster
    :returns: extent (column_upper_left, row_upper_left, ncols, nrows)
    
    This can be used as input in the rio.Windows.window function
    """
    
    row_ul, col_ul = rio_dataset.index(bbox[0], bbox[3])
    
    row_lr, col_lr = rio_dataset.index(bbox[2], bbox[1])
    
    extent = (col_ul, 
              row_ul, 
              col_lr - col_ul,
              row_lr - row_ul)
    
    return extent

def read_raster(in_raster, bands=None, bbox=None, as_image=False):
    """Read a raster. If bbox is specified, then only the specified bbox
    within the raster is read.

    Parameters
    ----------
    raster : path,
        absolute path to raster.
    bands : list of int or int. optional
        band(s) to read (remember band count starts from 1)
    bbox : list of int, optional
        bounding box of an area within the raster [xmin, ymin, xmax, ymax]
    as_image : boolean, optional
    """

    with rio.open(in_raster) as rio_dataset:
        if bands:
            if type(bands) == int:
                bands = [bands]
            else:
                None
        else:
            bands = list(np.arange(rio_dataset.count) + 1)

        if bbox:
            # if bbox is provided as indexes
            if type(bbox) == rio.windows.Window:
                array = rio_dataset.read(bands, window=bbox)

            else:
                # if a bbox with coordinates are specified, convert to pixel
                win = rio.windows.from_bounds(*bbox, rio_dataset.transform)

                # let's round to the closest pixel
                new_col_off = np.int32(np.round(win.col_off))
                new_row_off = np.int32(np.round(win.row_off))
                new_width = np.int32(np.round(win.width))
                new_height = np.int32(np.round(win.height))

                new_win = rio.windows.Window(new_col_off, new_row_off,
                                             new_width, new_height)

                array = rio_dataset.read(bands, window=new_win)
        else:
            # if none of the above, just read the whole array
            array = rio_dataset.read(bands)

    # reshape to (rows, columns, bands) from (bands, rows, columns)
    if as_image:
        return reshape_as_image(array)
    else:
        return (array)

def save_raster(fpath, arr, profile, is_image=True):
    with rio.open(fpath, "w", **profile) as dst:
        if is_image:
            dst.write(reshape_as_raster(arr))
        else:
            dst.write(arr)

def fake_RGB(in_raster, out_raster=None):

    """
    For some reasons the fake RGB can not be plotted in QGIS (maybe need RGBA!)
    :param in_raster:
    :param out_raster:
    :return:
    """
    in_raster = Path(in_raster)
    array = Image.open(in_raster).convert("RGB")
    if out_raster:
        None
    else:
        out_raster = in_raster.with_name(in_raster.stem + "_fakergb" + in_raster.suffix)
    array.save(out_raster)

def clip_from_bbox(in_raster, bbox, clipped_raster):
    
    """
    Clip a raster using the window functionality of rasterio and a specified
    bounding box and return a numpy array containing the clipped data as 
    (rows, columns, bands). Only the data within the window is read. If 
    clipped_raster is defined, then the array is written to the absolute path
    contained in clipped_raster.
    
    :param raster: absolute path to the input raster
    :param bbox: bounding box of an area within the raster
    :param clipped_raster: absolute path to where to save the clipped raster.
    if not defined, the clipped raster will not be saved.
    :returns: array: numpy array corresponding to the clipped area shaped as 
    (rows, columns, bands)

    Note: I should always make a copy of the in_meta to avoid changing the
    meta of the original file
    """
    
    with rio.open(in_raster) as rio_dataset:
        in_meta = rio_dataset.meta
        out_meta = in_meta.copy()
        
        # if we get a rio.windows.Windows directly
        if type(bbox) == rio.windows.Window:
            
            # read array for window
            array = rio_dataset.read(window=bbox)
            
            # get new transform
            win_transform = rio_dataset.window_transform(bbox)
            
        else:
            # get window (can be done with rio_dataset.index and indexes too)
            #extent = get_extent(rio_dataset, bbox)
            #win = rio.windows.Window(*extent)
            win = rio.windows.from_bounds(*bbox, rio_dataset.transform)

            # let's round to the closest pixel
            new_col_off = np.int32(np.round(win.col_off))
            new_row_off = np.int32(np.round(win.row_off))
            new_width = np.int32(np.round(win.width))
            new_height = np.int32(np.round(win.height))

            new_win = rio.windows.Window(new_col_off, new_row_off,
                                         new_width, new_height)

            # read array for window
            array = rio_dataset.read(window=new_win)
            
            # get new transform
            win_transform = rio_dataset.window_transform(new_win)
        
    # shape of array
    dst_channel, dst_height, dst_width = np.shape(array)

    # update meta information
    if in_meta['driver'] == 'VRT':
        try:
            out_meta = removekey(out_meta, "blockysize")
            out_meta = removekey(out_meta, "blockxsize")
        except:
            None
        out_meta.update({"tiled": False})
    else:
        None

    out_meta.update({"driver": "GTiff",
             "height": dst_height,
             "width": dst_width,
             "transform": win_transform})

    with rio.open(clipped_raster, "w", **out_meta) as dst:
        dst.write(array)


def clip(in_raster, in_polygon, out_raster=False):
    """
    Clip a raster using either a:
        i) polygon shape file (.shp)) - cliptype = 'shp'
        ii) bounding box ([left,bottom,right,top])  - cliptype = 'bbox'
        iii) geojson polygon - cliptype = 'geojson'

    The advanced clip function takes an input polygon under various formats
    (see above) and generate a clipped raster from it. It returns a numpy array
    containing the clipped data as (rows, columns, bands) and save the clipped
    raster if wanted.

    :param raster: absolute path to the input raster (to be clipped)
    :param in_polygon: input polygon (format depends on cliptype)
    :param cliptype: type of input polygon ('shp', 'bbox' or 'geojson')
    :param clipped_raster: absolute path to saved clipped raster* (*if wanted)
    :returns: array: numpy array corresponding to the clipped area shaped as
    (rows, columns, bands)
    """

    with rio.open(in_raster) as rio_dataset:
        out_meta = rio_dataset.meta

        # get clip shape from polygon shape
        if (type(in_polygon) == str) or (type(in_polygon) == pathlib.PosixPath):

            with fiona.open(in_polygon, "r") as polygon:
                shapes = [feature["geometry"] for feature in polygon]

                # clipping of raster
                out_array, out_transform = mask.mask(rio_dataset, shapes,
                                                     all_touched=False, crop=True)

                # if out_raster is specified, we save the data to a tif
                if out_raster:
                    out_meta.update({"driver": "GTiff",
                                     "height": out_array.shape[1],
                                     "width": out_array.shape[2],
                                     "transform": out_transform})

                    with rio.open(out_raster, "w", **out_meta) as dst:
                        dst.write(out_array)

                else:
                    None

        # if bounding box with coordinates (list or tuple)
        elif (type(in_polygon) == tuple or type(in_polygon) == list):

            out_array = clip_from_bbox(in_raster, in_polygon, out_raster)

        else:
            raise Exception('Input polygon format is not recognized. Please ' +
                            'specify a bounding box (as a tuple or a list with ' +
                            'xmin, ymin, xmax, ymax) or the absolute path to a ' +
                            'polygon shapefile (.shp)')
    return reshape_as_image(out_array)

def clip_advanced(in_raster, in_polygon, cliptype, clipped_raster = ""):
    
    """
    Args:
        in_raster: raster to be clipped
        in_polygon: as 
        cliptype : either 'bbox', 'shp', 'geojson'
        clipped_raster: raster to be saved if not it will .....
    """
    
    with rio.open(in_raster) as src:
        out_meta = src.meta
    
    
        # get clip shape from polygon shape
        if cliptype == "shp":
            
            with fiona.open(in_polygon, "r") as polygon:
                shapes = [feature["geometry"] for feature in polygon]
                
        # get clip shape from bounding box       
        elif cliptype == "bbox":
            
            # convert it to a polygon bbox
            bbox_pol = box(in_polygon[0], in_polygon[1], in_polygon[2], in_polygon[3]) 
            
            # use geopandas to transform the bounding box into a json polygon
            geo = gpd.GeoDataFrame({'geometry': bbox_pol}, index=[0], crs=src.crs)
                
            # convert it to a json polygon
            shapes = [json.loads(geo.to_json())['features'][0]['geometry']]
        
        # or getting directly a geojson string (as above)   
        elif cliptype == "geojson":
            shapes = in_polygon
                    
        else:
            None
            # print error 
                        
        # clipping of raster
        out_image, out_transform = mask.mask(src, shapes, crop=True)
        
    # if clipped raster, we save the data to a tif (otherwise just return out_image)    
    if clipped_raster:          
        out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})
        
        with rio.open(clipped_raster, "w", **out_meta) as dst:
            dst.write(out_image)
            
    else:
        None
        
    return reshape_as_image(out_image)


def reproject_raster(in_raster, dst_crs, filename):

    with rio.open(in_raster) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rio.open(filename, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.cubic)

def resample(raster_path, resolution):
    """
    Resample input raster based on the specified new_resolution.
    Cubic interpolation is used during the resampling step.
    Parameters
    ----------
    raster_path : str
        Path to raster
    resolution : int
        Resolution of target/output array

    Returns
    -------
    resampled_array: array
        Resampled array to chosen resolution

    """
    raster_path = Path(raster_path)

    with rio.open(raster_path) as src:
        array = src.read()
        profile = src.profile.copy()
        bounds = src.bounds # might have to calculate bounds based on 60 m?
        original_resolution = profile["transform"][0]
        resampling_factor = original_resolution / resolution

        transform, width, height = rio.warp.calculate_default_transform(
            profile["crs"],
            profile["crs"],
            int(profile["width"] * resampling_factor),
            int(profile["height"] * resampling_factor),
            *bounds
        )

        bands = array.shape[0]
        resampled_array = np.zeros((bands, height, width),
                                   dtype=profile['dtype'])

        rio.warp.reproject(
            source=array,
            destination=resampled_array,
            rio_dataset_transform=profile["transform"],
            rio_dataset_crs=profile["crs"],
            src_transform=profile["transform"],
            dst_transform=transform,
            src_crs=profile["crs"],
            dst_crs=profile["crs"],
            resampling=Resampling.cubic,
        )

        profile.update(
            {"transform": transform, "width": width, "height": height}
        )

        return (resampled_array)

            
            
def read_gtiff_bbox(dtype, in_raster, bbox, resampling_factor = 1.0, destination_clip = "", destination_resample= ""):
    
    """Read a given bounding box, in the source reference system, from
    a GeoTiff. The bounding box must be within the bounds of the data set. A 
    resampling factor can be specified if the source raster needs to be up- or 
    down-scaled. The resampled and clipped rasters can be saved (if wanted).

    :param dtype: type of values in raster (e.g., np.uint16)
    :param raster: absolute path to the input raster to be read
    :param bbox: bounding box of an area within the raster
    :param resampling_factor: resampling factor (float, default values equal to 1.0)
    :param clipped_raster: absolute path to the clipped raster (if specified will be saved)
    :param resampled_raster: absolute path to clipped and resampled raster (if specified will be saved)
    :returns: array: numpy array corresponding to the clipped and resampled area 
    shaped as (rows, columns, bands)
    """
    
    # define name of rasters (if not defined)
    if destination_clip:
        clipped_raster_name = destination_clip
    else:
        clipped_raster_name = './tmp_clipped_raster.tif'
    
    
    if destination_resample:
        resampled_raster_name = destination_resample
    else:
        resampled_raster_name = './tmp_resampled_raster.tif'
    
    # clip array
    clipped_array = clip_from_bbox(in_raster, bbox, clipped_raster_name)
    
    # resampling of clip array
    if resampling_factor == 1.0:
        array = clipped_array
    else:
        array = resample(clipped_raster_name, resampling_factor, resampled_raster_name)
    
    # remove temporary file
    files_to_be_removed = glob.glob("./tmp_*raster.tif")
    
    for f in files_to_be_removed:
        os.remove(f)
        
    return array

def tile_windows(in_raster, block_width = 512, block_height = 512, stride=False, add_together=False):
    
    """
    Tile the rio_dataset raster (rasterio format) to a desired number of blocks
    having specified width and height.
    
    :param rio_dataset: rasterio raster dataset
    :param width: width of blocks (multiplier of 16 is advised, e.g., 512)
    :param height: height of blocks (multiplier of 16 is advised, e.g., 512)
    :returns: tile_window, tile_transform: list of rasterio windows and transforms
    """

    nwidth = get_raster_width(in_raster)
    nheight = get_raster_height(in_raster)

    offsets = product(range(0, nwidth, block_width),
                      range(0, nheight, block_height))

    # I want to allow for three different scenarios
    # (1) tiling with no stride (only)
    # (2) tiling with stride (only)
    # (3) (1) and (2) together

    if stride:
        offsets_stride = product(range(stride, nwidth, block_width),
                          range(stride, nheight, block_height))
    else:
        None

    if stride & add_together:
        offsets = list(offsets) + list(offsets_stride)

    elif stride & ~add_together:
        offsets = offsets_stride

    else:
        None

    tile_window = []
    tile_transform = []
    tile_bounds = []

    with rio.open(in_raster) as src:
        src_transform = src.transform

        # added rounding to avoid varying height, width of tiles
        # maybe redundant
        for col_off, row_off in offsets:
            window =rio.windows.Window(col_off=col_off,
                                       row_off=row_off,
                                       width=block_width,
                                       height=block_height)

            new_col_off = np.int32(np.round(window.col_off))
            new_row_off = np.int32(np.round(window.row_off))
            new_width = np.int32(np.round(window.width))
            new_height = np.int32(np.round(window.height))

            new_win = rio.windows.Window(new_col_off, new_row_off,
                                         new_width, new_height)


            win_transform = src.window_transform(new_win)
            tile_window.append(new_win)
            tile_transform.append(win_transform)

            tile_bounds.append(rio.windows.bounds(new_win, src_transform,
                                              new_win.height,
                                              new_win.width))
        
    return tile_window, tile_transform, tile_bounds

def true_footprint(in_raster, out_shapefile=False):

    """
    footprint excluding nan values

    :param in_raster:
    :param out_shapefile:
    :return:

    :todo: allow for convertion to new crs?
    """

    in_raster = Path(in_raster)
    array = read_raster(in_raster, as_image=True).squeeze()
    mask = array != 0
    values = mask + 0
    values = values.astype('uint8')

    with rio.open(in_raster) as rio_dataset:
        if out_shapefile:
            gdf_true_footprint = polygonize(rio_dataset, values, mask, out_shapefile=out_shapefile)
        else:
            gdf_true_footprint = polygonize(rio_dataset, values, mask, out_shapefile=False)

    return (gdf_true_footprint)

def footprint(in_raster, crs_out=False, out_shapefile=False):

    """

    :param raster:
    :param crs_out:
    :return:

    :example:
    nac_dtm_folder1 = Path("/media/nilscp/pampa/NAC_DTM/NAC_DTM_RDR")
    nac_dtm_folder2 = Path("/media/nilscp/pampa/NAC_DTM/NAC_AMES")
    nac_dtms = sorted(list(nac_dtm_folder1.glob("*.TIF"))) + sorted(list(nac_dtm_folder2.glob("*.TIF")))
    crs_out = crs.Moon_Equidistant_Cylindrical()

    geom = [footprint(raster, crs_out) for raster in nac_dtms]

    dtm_path = []
    name = []
    for raster in nac_dtms:
        name.append(raster.stem)
        dtm_path.append(raster.as_posix())

    gdf = gpd.GeoDataFrame(np.column_stack((name, dtm_path)), columns=[
    "name", "path"], geometry=geom, crs=crs_out)

    gdf.to_file("/home/nilscp/QGIS/Moon/NAC_DTM_footprints/NAC_DTM_footprints.shp")
    """

    in_raster = Path(in_raster)
    crs_in = get_raster_crs(in_raster).to_wkt()
    bbox = get_raster_bbox(in_raster)
    bbox = box(*bbox)
    gs = gpd.GeoSeries(bbox, crs=crs_in)

    if crs_out:
        gs_proj = gs.to_crs(crs_out)
    else:
        gs_proj = gs

    if out_shapefile:
        gs_proj.to_file(out_shapefile)

    return (gs_proj.geometry.values[0])

def pad(in_raster, padding_height, padding_width):

    in_raster = Path(in_raster)
    out_raster = in_raster.with_name(in_raster.stem + "_padded" + in_raster.suffix)
    in_array = read_raster(in_raster, bands=None, bbox=None, as_image=True).squeeze()
    in_meta = get_raster_profile(in_raster)
    in_res = get_raster_resolution(in_raster)[0]
    in_bbox = get_raster_bbox(in_raster)
    padded_array = np.pad(in_array, (padding_height,padding_width), 'constant', constant_values=in_meta["nodata"])
    padded_array = np.expand_dims(padded_array, axis=2)
    out_meta = in_meta.copy()

    out_bbox = [in_bbox[0] - (in_res * padding_width[0]),
                in_bbox[1] - (in_res * padding_height[0]),
                in_bbox[2] + (in_res * padding_width[1]),
                in_bbox[3] + (in_res * padding_height[1])]

    out_meta["width"] = padded_array.shape[1]
    out_meta["height"] = padded_array.shape[0]
    out_meta["transform"] = Affine(in_res,0.0,out_bbox[0],0.0,-in_res,out_bbox[3])

    save_raster(out_raster, padded_array, out_meta, is_image=True)

def polygonize(rio_dataset, values, mask, out_shapefile=False):
    """

    :param rio_dataset:
    :param values:
    :param mask:
    :param out_shapefile:
    :return:

    :example:

    array = read_raster(raster, as_image=True).squeeze()
    mask = array > 200 # brightest region in the picture
    values = (mask + 0.0).astype('uint8')
    with rio.open(raster) as rio_dataset:
        meta = rio_dataset.profile
        polygonize(rio_dataset, values, mask, out_shapefile="/home/nilscp/tmp/shp/test.shp")
    """
    geoms = []
    meta = rio_dataset.profile
    results = ({'properties': {'raster_val': v}, 'geometry': s}
               for j, (s, v) in enumerate(
        features.shapes(values, mask=mask, transform=meta["transform"])))
    geoms.append(list(results))

    gdf = gpd.GeoDataFrame.from_features(geoms[0], crs=meta["crs"])
    if out_shapefile:
        gdf.to_file(out_shapefile)

    return gdf


def shift_with_padding(in_raster, x_shift, y_shift, out_raster):

    """

    :param in_raster:
    :param x_shift: (in m)
    :param y_shift: (in m)
    :return:
    """
    in_raster = Path(in_raster)
    in_res = get_raster_resolution(in_raster)[0]

    width_shift_px = x_shift / in_res
    height_shift_px = y_shift / in_res
    padding_width = np.abs(np.ceil(x_shift).astype('int') * 2) # padding is in m
    padding_height = np.abs(np.ceil(y_shift).astype('int') * 2) # padding is in m

    # let's pad the original raster so that we do not loose any pixels in the shifting
    in_raster_padded = in_raster.with_name(in_raster.stem + "_padded" + in_raster.suffix)
    pad(in_raster, (padding_height,padding_height), (padding_width,padding_width))
    out_meta = get_raster_profile(in_raster_padded)

    # pad raster based on shift
    shift_in_image = (-height_shift_px, width_shift_px) # in pixel resolution
    # note the minus in front of the height_shift_px in the image
    # this is because positive is downwards in array, origin is at the upper left

    # read array
    array = read_raster(in_raster_padded,as_image=True).squeeze()

    offset_corrected_image = fourier_shift(np.fft.fftn(array), shift_in_image)
    offset_corrected_image = np.fft.ifftn(offset_corrected_image)
    offset_corrected_image_uint8 = np.round(offset_corrected_image.real, decimals=0).astype('uint8')
    offset_corrected_image_uint8 = np.expand_dims(offset_corrected_image_uint8, axis=2)

    # save raster
    save_raster(out_raster, offset_corrected_image_uint8, out_meta, is_image=True)

    # crop the boundary of the raster by shifting the bbox and cropping the previous raster


def shift(in_raster, x_shift, y_shift, out_raster):

    """

    :param in_raster:
    :param x_shift: (in m)
    :param y_shift: (in m)
    :return:
    """
    in_raster = Path(in_raster)
    in_res = get_raster_resolution(in_raster)[0]

    width_shift_px = x_shift / in_res
    height_shift_px = y_shift / in_res

    out_meta = get_raster_profile(in_raster)

    # pad raster based on shift
    shift_in_image = (-height_shift_px, width_shift_px) # in pixel resolution
    # note the minus in front of the height_shift_px in the image
    # this is because positive is downwards in array, origin is at the upper left

    # read array
    array = read_raster(in_raster,as_image=True).squeeze()

    offset_corrected_image = fourier_shift(np.fft.fftn(array), shift_in_image)
    offset_corrected_image = np.fft.ifftn(offset_corrected_image)
    offset_corrected_image_uint8 = np.round(offset_corrected_image.real, decimals=0).astype('uint8')
    offset_corrected_image_uint8 = np.expand_dims(offset_corrected_image_uint8, axis=2)

    # save raster
    save_raster(out_raster, offset_corrected_image_uint8, out_meta, is_image=True)