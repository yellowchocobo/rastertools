"""Utils and commonly used routines for working with geodata sets."""

from bisect import bisect_left
import random

from affine import Affine
import fiona
import geopandas as gpd
import numpy as np
from osgeo import gdal, osr
from pyproj import Proj, transform
from rasterio.features import rasterize
from shapely.geometry import Point
import xarray


def listLayers(geopackage):
    """

    Parameters
    ----------
    geopackage: geopackage
        path

    Returns
    -------
    List all of the layers in a geopackage
    """

    return fiona.listlayers(geopackage)


def transform_bbox(bbox, srid_in, srid_out):
    """Transform srid bbox.
    :param bbox: numpy array (minx, miny, maxx, maxy)
    :param srid_in: srid code of input bbox
    :param srid_out: srid code of output bbox
    :returns: bbox numpy array (minx, miny, maxx, maxy)
    """
    in_proj = Proj(init=f'epsg:{srid_in}')
    out_proj = Proj(init=f'epsg:{srid_out}')
    x_out, y_out = transform(in_proj,
                             out_proj,
                             [bbox[0], bbox[0], bbox[2], bbox[2]],
                             [bbox[1], bbox[3], bbox[3], bbox[1]])
    return np.array([min(x_out), min(y_out), max(x_out), max(y_out)])


def rasterize_gdf(gdf, bbox, attribute, res=1):
    """Rasterize a geodataframe.
    :param gdf: geodataframe with polygons
    :param bbox: bbox (xmin, ymin, xmax, ymax) in srid of gdf
    :param attribute: name of attribute to be burned
    :param res: resolution (m) of the output raster
    :returns: np array with burned attribute"""
    window = np.rint((np.array([bbox[3] - bbox[1], bbox[2] - bbox[0]]) /
                      res)).astype('int')
    if gdf.empty:
        return np.zeros(window)
    shapes = ((geom, value)
              for geom, value in zip(gdf.geometry, gdf[attribute]))
    trans = Affine(res, 0, bbox[0], 0, -res, bbox[3])
    return rasterize(shapes=shapes, fill=np.nan, out_shape=window, transform=trans)


def random_point_in_polys(polys, acc_probs=None):
    """Pick a random point in a dataframe of polygons.
    :param polys: geopandas dataframe with polygons
    :param acc_probs: column in polys accumulated probabilities of selecting polygon
    :returns: numpy array [x,y]
    """
    # pick random poly
    if acc_probs is None:
        acc_probs = np.cumsum(polys.area.values / polys.area.sum())
    val = random.uniform(0, 1)
    index = bisect_left(acc_probs, val)
    random_poly = polys.iloc[[index]]
    # pick random point in poly
    bounds = random_poly.geometry.bounds.values[0]
    while True:
        # random point
        pt = Point(random.randint(int(bounds[0]), int(bounds[2])),
                   random.randint(int(bounds[1]), int(bounds[3])))
        # check if point in poly
        if random_poly.geometry.intersects(pt).bool():
            return np.array([pt.x, pt.y])


def get_epsg(ds):
    """Get epsg of xarray or gdal dataset
    :param ds: gdal or xarray dataset
    :returns: string with epsg
    """
    if isinstance(ds, gdal.Dataset):
        proj = osr.SpatialReference(wkt=ds.GetProjection())
        return proj.GetAttrValue('AUTHORITY', 1)
    elif isinstance(ds, gpd.geodataframe.GeoDataFrame):
        return ds.crs['init'].split(':')[-1]
    elif isinstance(ds, xarray.core.dataarray.DataArray):
        return ds.crs.split(':')[-1]
    raise TypeError("can't get epsg")


def world2pix(xy_world, origin, pix_size):
    """Convert world coordinates to pixel coordinates
    :param xy_world: numpy array [x, y] world coordinates
    :param origin: numpy array [x, y] origin (upper left) of raster in world coordinates
    :param pix_size: numpy array [dx, dy] pixel size
    :returns: numpy array [x, y] pixel coordinates
    """
    xy_pix = (xy_world - origin) / pix_size
    return np.rint(xy_pix).astype(int)


def pix2world(xy_pix, origin, pix_size):
    """Convert pixel coordinates to world coordinates
    :param xy_pix: numpy array [x, y] pixel coordinates
    :param origin: numpy array [x, y] origin (upper left) of raster in world coordinates
    :param pix_size: numpy array [dx, dy] pixel size
    :returns: numpy array [x, y] world coordinates
    """
    xy_world = origin + (xy_pix * pix_size)
    return xy_world
