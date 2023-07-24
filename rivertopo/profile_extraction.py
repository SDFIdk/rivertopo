
from osgeo import gdal, ogr
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy import interpolate
from collections import namedtuple
import argparse
#import logging
import csv

from cross_lines_z_2 import create_perpendicular_lines, create_perpendicular_lines_on_polylines

gdal.UseExceptions()
ogr.UseExceptions()

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

    #input_raster_dataset = gdal.Open(input_raster_path)

    input_lines_datasrc = ogr.Open(input_lines_path)
    input_lines_layer = input_lines_datasrc.GetLayer()
    input_lines_feature = input_lines_layer.GetNextFeature()

    segments = []
    for input_lines_feature in input_lines_layer:
        stream_linestring = input_lines_feature.GetGeometryRef()

        linestring_points = np.array(stream_linestring.GetPoints())[:,:2]

        for i in range(len(linestring_points) - 1):
            segment = (linestring_points[i], linestring_points[i+1])
            segments.append(segment)

    output_lines_driver = ogr.GetDriverByName("gpkg")
    output_lines_datasrc = output_lines_driver.CreateDataSource(output_lines_path)
    output_lines_datasrc.CreateLayer(
        "rendered_lines",
        srs=input_lines_layer.GetSpatialRef(),
        geom_type=ogr.wkbLineString25D,
    )
    output_lines_layer = output_lines_datasrc.GetLayer()

    
    # Add a new field for the segment index
    field_defn = ogr.FieldDefn("Seg_Index", ogr.OFTInteger)
    output_lines_layer.CreateField(field_defn)

    # iterate over line and extract profiles perpendicular to the line
      # Open CSV file for writing
    with open('all_lines.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Line_ID", "X", "Y", "Z"])  # write header

        # Loop over all segments
        for segment_index, segment in enumerate(segments):
            #convert segment to LineString
            segment_linestring = ogr.Geometry(ogr.wkbLineString)
            segment_linestring.AddPoint(*segment[0])
            segment_linestring.AddPoint(*segment[1])

            perpendicular_lines, offset, t, x3, x4, y3, y4 = create_perpendicular_lines_on_polylines(segment_linestring, length=10, interval=1)
            
            # Create the LineString for the perpendicular line and add it to the output layer
            for offset, t, x3, x4, y3, y4 in perpendicular_lines:
                
                # Create an array of x and y coordinates along the line
                x_coords = np.linspace(x3, x4, num=200) # num is number of segments
                y_coords = np.linspace(y3, y4, num=200)

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
                
                feature = ogr.Feature(output_lines_layer.GetLayerDefn())
                feature.SetField("Seg_Index", segment_index)
                feature.SetGeometry(line_geometry)
                output_lines_layer.CreateFeature(feature)
                feature = None

                 # Write the line data to the csv file
                line_id = f'segment_{segment_index}'
                writer.writerows((line_id, x, y, z) for x, y, z in zip(x_coords, y_coords, z_values))


    
    ##### Links til inspi #####
    # https://stackoverflow.com/questions/62283718/how-to-extract-a-profile-of-value-from-a-raster-along-a-given-line
    # https://gis.stackexchange.com/questions/167372/extracting-values-from-raster-under-line-using-gdal
    # https://kokoalberti.com/articles/creating-elevation-profiles-with-gdal-and-two-point-equidistant-projection/
    # https://gis.stackexchange.com/questions/50108/elevation-profile-10-km-each-side-of-line
    # https://stackoverflow.com/questions/59144464/plotting-two-cross-section-intensity-at-the-same-time-in-one-figure


if __name__ == '__main__':
    main()
