
from osgeo import gdal, ogr
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from collections import namedtuple
import argparse
import os

from rivertopo.cross_lines_z import create_perpendicular_lines

gdal.UseExceptions()
ogr.UseExceptions()

"""
This script is used to extract elevation profiles along perpendicular lines to a given polyline 
feature in a DEM raster

"""

# Create bounding box

### copied sampling.py from hydroadjust ###

BoundingBox = namedtuple(
    'BoundingBox',
    ['x_min', 'x_max', 'y_min', 'y_max'],
)

def get_raster_window(dataset, bbox):
    """
    Return a window of the input raster dataset, containing at least the
    provided bounding box.
    
    :param dataset: Source raster dataset
    :type dataset: GDAL Dataset object
    :param bbox: Window bound coordinates
    :type bbox: hydroadjust.sampling.BoundingBox object
    :returns: GDAL Dataset object for the requested window
    """
    
    input_geotransform = dataset.GetGeoTransform()
    
    if input_geotransform[2] != 0.0 or input_geotransform[4] != 0.0:
        raise ValueError("geotransforms with rotation are unsupported")
    
    input_offset_x = input_geotransform[0]
    input_offset_y = input_geotransform[3]
    input_pixelsize_x = input_geotransform[1]
    input_pixelsize_y = input_geotransform[5]
    
    # We want to find window coordinates that:
    # a) are aligned to the source raster pixels
    # b) contain the requested bounding box plus at least one pixel of "padding" on each side, to allow for small floating-point rounding errors in X/Y coordinates
    # 
    # Recall that the pixel size in the geotransform is commonly negative, hence all the min/max calls.
    raw_x_min_col_float = (bbox.x_min - input_offset_x) / input_pixelsize_x
    raw_x_max_col_float = (bbox.x_max - input_offset_x) / input_pixelsize_x
    raw_y_min_row_float = (bbox.y_min - input_offset_y) / input_pixelsize_y
    raw_y_max_row_float = (bbox.y_max - input_offset_y) / input_pixelsize_y
    
    col_min = int(np.floor(min(raw_x_min_col_float, raw_x_max_col_float))) - 1
    col_max = int(np.ceil(max(raw_x_min_col_float, raw_x_max_col_float))) + 1
    row_min = int(np.floor(min(raw_y_min_row_float, raw_y_max_row_float))) - 1
    row_max = int(np.ceil(max(raw_y_min_row_float, raw_y_max_row_float))) + 1
    
    x_col_min = input_offset_x + input_pixelsize_x * col_min
    x_col_max = input_offset_x + input_pixelsize_x * col_max
    y_row_min = input_offset_y + input_pixelsize_y * row_min
    y_row_max = input_offset_y + input_pixelsize_y * row_max
    
    # Padded, georeferenced window coordinates. The target window to use with gdal.Translate().
    padded_bbox = BoundingBox(
        x_min=min(x_col_min, x_col_max),
        x_max=max(x_col_min, x_col_max),
        y_min=min(y_row_min, y_row_max),
        y_max=max(y_row_min, y_row_max),
    )
    
    # Size in pixels of destination raster
    dest_num_cols = col_max - col_min
    dest_num_rows = row_max - row_min
    
    translate_options = gdal.TranslateOptions(
        width=dest_num_cols,
        height=dest_num_rows,
        projWin=(padded_bbox.x_min, padded_bbox.y_max, padded_bbox.x_max, padded_bbox.y_min),
        resampleAlg=gdal.GRA_NearestNeighbour,
    )
    
    # gdal.Translate() needs a destination *name*, not just a Dataset to
    # write into. Create a temporary file in GDAL's virtual filesystem as a
    # stepping stone.
    window_dataset_name = "/vsimem/temp_window.tif"
    window_dataset = gdal.Translate(
        window_dataset_name,
        dataset,
        options=translate_options
    )
    
    return window_dataset


def get_raster_interpolator(dataset):
    """
    Return a scipy.interpolate.RegularGridInterpolator corresponding to a GDAL
    raster.
    
    :param dataset: Raster dataset in which to interpolate
    :type dataset: GDAL Dataset object
    :returns: RegularGridInterpolator accepting georeferenced X and Y input
    """
    
    geotransform = dataset.GetGeoTransform()
    band = dataset.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()
    z_grid = band.ReadAsArray()
    num_rows, num_cols = z_grid.shape
    
    if geotransform[2] != 0.0 or geotransform[4] != 0.0:
        raise ValueError("geotransforms with rotation are unsupported")
    
    # X and Y values for the individual columns/rows of the raster. The 0.5 is
    # added in order to obtain the coordinates of the cell centers rather than
    # the corners.
    x_values = geotransform[0] + geotransform[1]*(0.5+np.arange(num_cols))
    y_values = geotransform[3] + geotransform[5]*(0.5+np.arange(num_rows))

    # RegularGridInterpolator requires the x and y arrays to be in strictly
    # increasing order, accommodate this
    if geotransform[1] > 0.0:
        col_step = 1
    else:
        col_step = -1
        x_values = np.flip(x_values)

    if geotransform[5] > 0.0:
        row_step = 1
    else:
        row_step = -1
        y_values = np.flip(y_values)
        
    # NODATA values must be replaced with NaN for interpolation purposes
    z_grid[z_grid == nodata_value] = np.nan
    
    # The grid must be transposed to swap (row, col) coordinates into (x, y)
    # order.
    interpolator = RegularGridInterpolator(
        points=(x_values, y_values),
        values=z_grid[::row_step, ::col_step].transpose(),
        method='linear',
        bounds_error=False,
        fill_value=np.nan,
    )
    
    return interpolator


def create_perpendicular_lines_at_interval(stream_linestring, length=20, interval=20):
    """
    Create perpendicular lines over a given polyline at specified intervals.
    
    :param stream_linestring: The input stream linestring.
    :param length: Length of the perpendicular line (default is 30 meters).
    :param interval: Interval distance along the linestring to create perpendicular lines.
    :return: List of perpendicular lines and their attributes (including stationing distance).
    """
    perpendicular_lines = []
    total_length = stream_linestring.Length()
    next_interval_distance = interval

    while next_interval_distance <= total_length:
        # Find the segment where this interval falls
        cumulative_distance = 0
        for i in range(stream_linestring.GetPointCount() - 1):
            point1 = stream_linestring.GetPoint(i)
            point2 = stream_linestring.GetPoint(i + 1)

            segment_length = np.hypot(point2[0] - point1[0], point2[1] - point1[1])
            if cumulative_distance + segment_length >= next_interval_distance:
                # Create a perpendicular line at this segment
                point1_geometry = ogr.Geometry(ogr.wkbPoint)
                point1_geometry.AddPoint(*point1)
                point2_geometry = ogr.Geometry(ogr.wkbPoint)
                point2_geometry.AddPoint(*point2)

                offset, t, x3, x4, y3, y4 = create_perpendicular_lines(point1_geometry, point2_geometry, length=length)
                perpendicular_lines.append((offset, t, x3, x4, y3, y4, next_interval_distance))
                break

            cumulative_distance += segment_length

        # Update the next interval distance
        next_interval_distance += interval

    return perpendicular_lines



def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('input_rasters', nargs='+', type=str, help= 'input DEM raster datasets to sample')
    #argument_parser.add_argument('input_raster', type=str, help= 'input DEM raster dataset to sample')
    argument_parser.add_argument('input_line', type=str, help= 'input line-object vector data source')
    argument_parser.add_argument('output_lines', type=str, help='output geometry file')

    input_arguments = argument_parser.parse_args()

    input_raster_path = input_arguments.input_rasters
    input_lines_path = input_arguments.input_line
    output_lines_path = input_arguments.output_lines

    vrt_options = gdal.BuildVRTOptions(resampleAlg='bilinear')
    input_raster_vrt = gdal.BuildVRT("/vsimem/input_raster.vrt", input_raster_path, options=vrt_options)
    
    input_raster_dataset = input_raster_vrt 

    input_lines_datasrc = ogr.Open(input_lines_path)
    input_lines_layer = input_lines_datasrc.GetLayer()
    input_lines_feature = input_lines_layer.GetNextFeature()


    for input_lines_feature in input_lines_layer:
        stream_linestring = input_lines_feature.GetGeometryRef()
        
        perpendicular_lines= create_perpendicular_lines_at_interval(stream_linestring, length=20, interval=20)

    output_lines_driver = ogr.GetDriverByName("gpkg")
    output_lines_datasrc = output_lines_driver.CreateDataSource(output_lines_path)
    output_lines_datasrc.CreateLayer(
        "rendered_lines",
        srs=input_lines_layer.GetSpatialRef(),
        geom_type=ogr.wkbLineString25D,
    )
    output_lines_layer = output_lines_datasrc.GetLayer()
 
    # Prepare data storage
    all_lines_data = {
        'line_ids': [],
        'x_coords': [],
        'y_coords': [],
        'z_values': [],
        'distances': [],
    }
    
    for perp_line in perpendicular_lines:
        offset, t, x3, x4, y3, y4, perp_line_station = perp_line     
     
        # Create an array of x and y coordinates along the line
        x_coords = np.linspace(x3, x4, num=50)
        y_coords = np.linspace(y3, y4, num=50)

        # Create bounding box encompassing the entire line
        input_line_bbox = BoundingBox(
            x_min=min(x_coords),
            x_max=max(x_coords),
            y_min=min(y_coords),
            y_max=max(y_coords)
        )

        # Get a raster window just covering this line object
        window_raster_dataset = get_raster_window(input_raster_dataset, input_line_bbox)

        # Prepare the interpolator
        window_raster_interpolator = get_raster_interpolator(window_raster_dataset)

        # Interpolate z values along the line
        z_values = window_raster_interpolator((x_coords, y_coords))
           
        # Create a new line geometry, including z values
        line_geometry = ogr.Geometry(ogr.wkbLineString25D)
        for x, y, z in zip(x_coords, y_coords, z_values):
            line_geometry.AddPoint(x, y, z)

        # Calculate the distances for each row
        distances = [0]  # Initialize with the first distance as 0
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            distance = np.sqrt(dx**2 + dy**2) + distances[-1]
            distances.append(distance)

        feature = ogr.Feature(output_lines_layer.GetLayerDefn())
        feature.SetGeometry(line_geometry)
        output_lines_layer.CreateFeature(feature)
        feature = None

        # Store the line data
        line_id = f'segment_{perp_line_station}'
        all_lines_data['line_ids'].append([line_id] * len(x_coords))
        all_lines_data['x_coords'].append(x_coords)
        all_lines_data['y_coords'].append(y_coords)
        all_lines_data['z_values'].append(z_values)
        all_lines_data['distances'].append(distances)
        
    concatenated_data = {}
    for key in all_lines_data:
        concatenated_data[key] = np.concatenate(all_lines_data[key])

    # Save to .npz
    output_file_path = r"C:\projekter\rivertopo\tests\data\vaerebro20.npz"

    np.savez_compressed(output_file_path, **concatenated_data)

    ##### Links til inspi #####
    # https://stackoverflow.com/questions/62283718/how-to-extract-a-profile-of-value-from-a-raster-along-a-given-line
    # https://gis.stackexchange.com/questions/167372/extracting-values-from-raster-under-line-using-gdal
    # https://kokoalberti.com/articles/creating-elevation-profiles-with-gdal-and-two-point-equidistant-projection/
    # https://gis.stackexchange.com/questions/50108/elevation-profile-10-km-each-side-of-line
    # https://stackoverflow.com/questions/59144464/plotting-two-cross-section-intensity-at-the-same-time-in-one-figure


if __name__ == '__main__':
    main()
