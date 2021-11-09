"""Utils and commonly used routines for working with GDAL/OGR data sets."""

import os
import numpy as np
import pyproj
import rasterio as rio
import geopandas as gpd
import json
import fiona
import glob

from itertools import product

import rasterio.mask as mask
from rasterio.plot import reshape_as_raster, reshape_as_image
from rasterio.enums import Resampling

from affine import Affine

from shapely.geometry import Polygon, box


def get_raster_resolution(raster):
    with rio.open(raster) as rio_dataset:
        res = rio_dataset.res
    return res

def get_raster_types(raster):
    with rio.open(raster) as rio_dataset:
        dtypes = rio_dataset.dtypes
    return dtypes

def get_raster_height(raster):
    with rio.open(raster) as rio_dataset:
        height = rio_dataset.height
    return height

def get_raster_width(raster):
    with rio.open(raster) as rio_dataset:
        width = rio_dataset.width
    return width

def get_raster_bbox(raster):
    with rio.open(raster) as rio_dataset:
        bbox =rio_dataset.bounds
    return list(bbox)

def get_raster_shape(raster):
    with rio.open(raster) as rio_dataset:
        shape = rio_dataset.shape
    return shape

def get_raster_nbands(raster):
    with rio.open(raster) as rio_dataset:
        count = rio_dataset.count
    return count

def get_raster_profile(raster):
    with rio.open(raster) as rio_dataset:
        profile = rio_dataset.profile.copy()
    return profile


def crs_eqc(crs_wkt_src, lat):
        
    # standard parallel should be replaced by the latitude
    crs_wkt_dst = crs_wkt_src.replace('["standard_parallel_1",0]', '["standard_parallel_1",' + str(int(lat)) + ']')
    
    # central meridian should be replaced by the longitude
    #crs_wkt = crs_wkt.replace('["central_meridian",0]', '["central_meridian",' + str(int(lon)) + ']')
    
    return crs_wkt_dst

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


def boundary_to_polygon(bbox):
    
    """convert bbox to shapely polygon
    :param bbox: bounding box of a raster
    :returns: shapely polygon of the same projection as the bbox
    """
    
    [ulx, lry, lrx, uly] = bbox
    
    x_coords = [ulx, lrx, lrx, ulx, ulx] # will this be equivalent?
    y_coords = [uly, uly, lry, lry, uly]
    pol = Polygon(zip(x_coords, y_coords))
    
    return pol

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

def raster_extent_idxs(rio_dataset, bbox):
    """get extent (column_upper_left, row_upper_left, ncols, nrows) from a
     bounding box. The bounding box must have the same coordinate systems and
     within the bounds of the rio_dataset raster.

    This can be used as input in the rio.Windows.window function

    Parameters
    ----------
    rio_dataset : rio dataset (e.g., raster)
    bbox : list of int
        bounding box of the raster [xmin, ymin, xmax, ymax]

    Returns
    ----------
    extent (column_upper_left, row_upper_left, ncols, nrows)
    """

    row_ul, col_ul = rio_dataset.index(bbox[0], bbox[3])

    row_lr, col_lr = rio_dataset.index(bbox[2], bbox[1])

    extent = (col_ul,
              row_ul,
              col_lr - col_ul,
              row_lr - row_ul)

    return extent

def read_raster(raster, bands=None, bbox=None, as_image=False):
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

    with rio.open(raster) as rio_dataset:
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
                # if a bbox with coordinates are specified, convert to idxs
                idxs = raster_extent_idxs(rio_dataset, bbox)
                array = rio_dataset.read(bands, window=rio.windows.Window(
                    *idxs))
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

def clip_from_bbox(raster, bbox, clipped_raster = ""):
    
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
    """
    
    with rio.open(raster) as rio_dataset:
        out_meta = rio_dataset.meta
        
        # if we get a rio.windows.Windows directly
        if type(bbox) == rio.windows.Window:
            
            # read array for window
            array = rio_dataset.read(window=bbox)
            
            # get new transform
            win_transform = rio_dataset.window_transform(bbox)
            
        else:
            # get window (can be done with rio_dataset.index and indexes too)
            extent = get_extent(rio_dataset, bbox)
                
            win = rio.windows.Window(*extent)
            
            # read array for window
            array = rio_dataset.read(window=win)
            
            # get new transform
            win_transform = rio_dataset.window_transform(win)
        
        # shape of array
        dst_channel, dst_height, dst_width = np.shape(array)
        

        
        # update meta information
        if clipped_raster:          
            out_meta.update({"driver": "GTiff",
                     "height": dst_height,
                     "width": dst_width,
                     "transform": win_transform})
            
            with rio.open(clipped_raster, "w", **out_meta) as dst:
                dst.write(array)
                
        else:
            None
            
        return reshape_as_image(array)
        
        
def clip(raster, in_polygon, destination = None):
    
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
    
    with rio.open(raster) as rio_dataset:
        out_meta = rio_dataset.meta
    
    
        # get clip shape from polygon shape
        if type(in_polygon) == str:
            
            with fiona.open(in_polygon, "r") as polygon:
                shapes = [feature["geometry"] for feature in polygon]
                
                # clipping of raster
                out_array, out_transform = mask.mask(rio_dataset, shapes, crop=True)
                
                # if destination is specified, we save the data to a tif 
                if destination:          
                    out_meta.update({"driver": "GTiff",
                             "height": out_array.shape[1],
                             "width": out_array.shape[2],
                             "transform": out_transform})
                    
                    with rio.open(destination, "w", **out_meta) as dst:
                        dst.write(out_array)
                        
                else:
                    None
                
        # if bounding box with coordinates (list or tuple)  
        elif (type(in_polygon) == tuple or type(in_polygon) == list):
            
            out_array = clip_from_bbox(raster, in_polygon, destination)
                            
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


def reproject(in_raster, dest_crs_wkt, reproj_raster):
    
    
    with rio.open(in_raster) as src:
        
        # get the new coordinate system
        dest_crs = src.crs.from_wkt(dest_crs_wkt)
        
        transform, width, height = rio.warp.calculate_default_transform(
            src.crs, dest_crs, src.width, src.height, *src.bounds)
        
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dest_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
    
        with rio.open(reproj_raster, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rio.warp.reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dest_crs,
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

        transform, width, height = rasterio.warp.calculate_default_transform(
            profile["crs"],
            profile["crs"],
            int(profile["width"] * resampling_factor),
            int(profile["height"] * resampling_factor),
            *bounds
        )

        bands = array.shape[0]
        resampled_array = np.zeros((bands, height, width),
                                   dtype=profile['dtype'])

        rasterio.warp.reproject(
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

            
            
def read_gtiff_bbox(dtype, raster, bbox, resampling_factor = 1.0, destination_clip = "", destination_resample= ""):
    
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
    clipped_array = clip_from_bbox(raster, bbox, clipped_raster_name)
    
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

def tile_windows(rio_dataset, width = 512, height = 512):
    
    """
    Tile the rio_dataset raster (rasterio format) to a desired number of blocks
    having specified width and height.
    
    :param rio_dataset: rasterio raster dataset
    :param width: width of blocks (multiplier of 16 is advised, e.g., 512)
    :param height: height of blocks (multiplier of 16 is advised, e.g., 512)
    :returns: tile_window, tile_transform: list of rasterio windows and transforms
    """
    
    # number of columns and rows
    ncols, nrows = rio_dataset.meta['width'], rio_dataset.meta['height']
    
    offsets = product(range(0, ncols, width), range(0, nrows, height))
    
    big_window = rio.windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)
    
    tile_window = []
    tile_transform = []
    
    for col_off, row_off in  offsets:
        window =rio.windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = rio_dataset.window_transform(window)
        
        tile_window.append(window)
        tile_transform.append(transform)
        
    return tile_window, tile_transform

def tile_bounds(rio_dataset, tiles_window):
    
    """
    Transform rasterio windows to bounding boxes (bbox).
    
    :param rio_dataset: rasterio raster dataset
    :param tiles_window: list of rasterio windows
    :returns: tbounds: list of bounding boxes
    """
    
    tbounds = []
    
    for tile in tiles_window:
                
        tbounds.append(rio.windows.bounds(tile, rio_dataset.transform, tile.height, tile.width))
                
    return tbounds
