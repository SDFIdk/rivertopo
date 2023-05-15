#from sampling import BoundingBox, get_raster_window, get_raster_interpolator

from itertools import groupby
from osgeo import gdal, ogr
import numpy as np
from tqdm import tqdm
import argparse
import logging
import random

gdal.UseExceptions()
ogr.UseExceptions()

def create_perpendicular_line(point1_geometry, point2_geometry, length):
    x1, y1 = point1_geometry.GetX(), point1_geometry.GetY()
    x2, y2 = point2_geometry.GetX(), point2_geometry.GetY()

    # Calculate the slope of the original line
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        m = float('inf')  # slope is infinity
    else:
        m = dy / dx

    # Calculate the slope of the line perpendicular to the original line
    if m == 0:
        m_perpendicular = float('inf')
    else:
        m_perpendicular = -1 / m

    # Calculate the coordinates of the endpoints of the perpendicular line
    dx_perpendicular = (length / 2) / ((1 + m_perpendicular**2)**0.5)
    dy_perpendicular = m_perpendicular * dx_perpendicular
    x3 = x1 - dx_perpendicular
    y3 = y1 - dy_perpendicular
    x4 = x1 + dx_perpendicular
    y4 = y1 + dy_perpendicular

    # Create the perpendicular line
    line_geometry = ogr.Geometry(ogr.wkbLineString25D)
    line_geometry.AddPoint(x3, y3, random.random())
    line_geometry.AddPoint(x4, y4, random.random())

    return line_geometry

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('input_points', type=str, nargs='+', help='input points vector data source(s)')
    argument_parser.add_argument('output_lines', type=str, help='output geometry file for lines with Z')
    input_arguments = argument_parser.parse_args()

    input_points_paths = input_arguments.input_points
    output_lines_path = input_arguments.output_lines

    # Load all points from all files into memory and sort by "id" attribute
    points = []
    for input_points_path in input_points_paths:
        input_points_datasrc = ogr.Open(input_points_path)
        input_points_layer = input_points_datasrc.GetLayer()
        for point_feature in input_points_layer:
            points.append(point_feature)

    points.sort(key=lambda f: (f.GetField("laengdepro"), f.GetField("regulativs")))

    # Now we group the points based on their ID
    grouped_points = {k: sorted(g, key=lambda f: f.GetField("regulativs")) for k, g in groupby(points, key=lambda f: f.GetField("laengdepro"))}

    # Create the output file
    output_lines_driver = ogr.GetDriverByName("gpkg")
    output_lines_datasrc = output_lines_driver.CreateDataSource(output_lines_path)
    output_lines_datasrc.CreateLayer(
        "rendered_lines",
        srs=input_points_layer.GetSpatialRef(),
        geom_type=ogr.wkbLineString25D,
    )
    output_lines_layer = output_lines_datasrc.GetLayer()

    # Iterate over each group of points and create a line for each pair of points within the group
    for point_group in grouped_points.values():
        for i in range(len(point_group) - 1):
            point1 = point_group[i]
            point2 = point_group[i + 1]

            point1_geometry = point1.GetGeometryRef()
            point2_geometry = point2.GetGeometryRef()

            # Create a LineString from the two points
            line_geometry = ogr.Geometry(ogr.wkbLineString25D)
            line_geometry.AddPoint(point1_geometry.GetX(), point1_geometry.GetY(), random.random())
            line_geometry.AddPoint(point2_geometry.GetX(), point2_geometry.GetY(), random.random())

            # Create output feature
            output_line_feature = ogr.Feature(output_lines_layer.GetLayerDefn())
            output_line_feature.SetGeometry(line_geometry)
            output_lines_layer.CreateFeature(output_line_feature)

            # Create a perpendicular line at the midpoint of the original line
            perpendicular_line_geometry = create_perpendicular_line(point1_geometry, point2_geometry, 10)  # replace 1 with the desired length
            output_perpendicular_line_feature = ogr.Feature(output_lines_layer.GetLayerDefn())
            output_perpendicular_line_feature.SetGeometry(perpendicular_line_geometry)
            output_lines_layer.CreateFeature(output_perpendicular_line_feature)

    logging.info(f"processed {len(points)} points and created lines")

if __name__ == '__main__':
    main()


