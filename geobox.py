"""Manipulate and work with geometric bounding boxes.

A bounding box is defined as a tuple (xmin, ymin, xmax, ymax) and represents
the extents of a geometry (in the Cartesian plane).

North-up, i.e. upper-left corner as (xmin, ymax) and rotation/skew=0, is
assumed on all operations.
"""

import numpy as np
from osgeo import gdal, ogr
from shapely.geometry import Polygon


def from_polygon(p):
    return p.exterior.bounds

def from_vectorlayer(layer):
    xmin, xmax, ymin, ymax = layer.GetExtent()
    return (xmin, ymin, xmax, ymax)

def from_vectorfile(fp, layer=0):
    ds = ogr.Open(fp)
    return from_vectorlayer(ds.GetLayer(layer))

def from_raster(ds):
    ulx, xres, _, uly, _, yres = ds.GetGeoTransform()
    return (
        ulx,
        uly + yres*ds.RasterYSize,
        ulx + xres*ds.RasterXSize,
        uly
    )

def from_rasterfile(fp):
    ds = gdal.Open(fp, gdal.GA_ReadOnly)
    box = from_raster(ds)
    ds = None
    return box

def to_geotransform(resolution, box):
    xmin, ymin, xmax, ymax = box
    return (xmin, resolution, 0, ymax, 0, -resolution)

def to_polygon(box, clockwise=False):
    xmin, ymin, xmax, ymax = box
    coords = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
        (xmin, ymin)
    ]
    if clockwise:
        return Polygon(reversed(coords))
    return Polygon(coords)

def to_pixelextent(gt, box):
    """Convert a bounding box (xmin, ymin, xmax, ymax) to
    (xoffset, yoffset, nx, ny) in the pixel space of a raster.
    This can be given as argument to GDAL's ReadAsArray.
    """
    xmin, ymin, xmax, ymax = box
    ulx, xres, _, uly, _, yres = gt
    return (
        int((xmin-ulx)/xres),
        int((ymax-uly)/yres),
        int((xmax-xmin)/xres),
        int((ymin-ymax)/yres)
    )

def union(boxes):
    """Return the global bounds of multiple extents."""
    xmins, ymins, xmaxs, ymaxs = zip(*boxes)
    return (
        min(*xmins),
        min(*ymins),
        max(*xmaxs),
        max(*ymaxs)
    )

def intersection(boxes):
    """Return the intersecting bounds of multiple extents."""
    xmins, ymins, xmaxs, ymaxs = zip(*boxes)
    xmin, ymin, xmax, ymax = (
        max(*xmins),
        max(*ymins),
        min(*xmaxs),
        min(*ymaxs)
    )
    if xmax - xmin <= 0 or ymax - ymin <= 0:
        raise Exception("Boxes does not intersect")
    return (xmin, ymin, xmax, ymax)

def ceil(resolution, box):
    """Expand an extent to nearest multiple of <resolution>"""
    xmin, ymin, xmax, ymax = box
    return (
        int(round(xmin - xmin % resolution)),
        int(round(ymin - ymin % resolution)),
        int(round(xmax + (-xmax % resolution))),
        int(round(ymax + (-ymax % resolution)))
    )

def expand(delta, box):
    """Expand a bounding box by a given delta."""
    xmin, ymin, xmax, ymax = box
    return (xmin-delta, ymin-delta, xmax+delta, ymax+delta)


def random_subbox(box, box_size):
    """Draw a random sub bounding box.

    Args:
        box: Bounding box (xmin, ymin, xmax, ymax)
        box_size: Size of sub bounding box in meters.
    Returns:
        A sub bounding box (xmin, ymin, xmax, ymax) contained by bbox.
    """
    xmin, ymin, xmax, ymax = box
    subxmin = np.random.randint(xmin, xmax-box_size)
    subymin = np.random.randint(ymin, ymax-box_size)
    return (subxmin, subymin, subxmin+box_size, subymin+box_size)


def sliding_window(box, box_size, step):
    """Yield sub-boxes of size <box_size>*<box_size>"""
    xmin, ymin, xmax, ymax = box
    assert all([
        xmax-xmin >= box_size,
        ymax-ymin >= box_size,
        step > 0
    ])
    x = xmin
    y = ymax
    while y > ymin:
        while x < xmax:
            _x = x-max(0, x+box_size-xmax)
            _y = y+max(0, ymin-(y-box_size))
            yield(_x, _y-box_size, _x+box_size, _y)
            x += step
        y -= step
        x = xmin
