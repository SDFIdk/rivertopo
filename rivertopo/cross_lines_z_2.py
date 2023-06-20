from itertools import groupby
from osgeo import gdal, ogr
import numpy as np
import argparse
import logging

from profile import RegulativProfilSimpel, RegulativProfilSammensat, OpmaaltProfil # import interpolation classes
from snapping import snap_points
from cross_lines_z import create_perpendicular_lines

gdal.UseExceptions()
ogr.UseExceptions()

def get_profile(point, profile_type):
    if profile_type == 'RegulativProfilSimpel':
        return RegulativProfilSimpel(point)
    elif profile_type == 'RegulativProfilSammensat':
        return RegulativProfilSammensat(point)
    elif profile_type == 'OpmaaltProfil':
        return OpmaaltProfil(point)

def create_perpendicular_lines_on_polylines(stream_linestring, length=10, interval=1):
    # Get the number of points in the stream_linestring
    num_points = stream_linestring.GetPointCount()

    # Initialize list to store perpendicular lines
    perpendicular_lines = []
    
    # Iterate over each pair of consecutive points in the stream_linestring
    for i in range(num_points - 1):
        point1_coords = stream_linestring.GetPoint(i)
        point2_coords = stream_linestring.GetPoint(i + 1)
        # Create geometry objects from point coordinates
        point1_geometry = ogr.Geometry(ogr.wkbPoint)
        point2_geometry = ogr.Geometry(ogr.wkbPoint)
        point1_geometry.AddPoint(*point1_coords)
        point2_geometry.AddPoint(*point2_coords)
        # Calculate perpendicular line between the two points
        offset, t, x3, x4, y3, y4 = create_perpendicular_lines(point1_geometry, point2_geometry, length=10)
        
        perpendicular_lines.append((offset, t, x3, x4, y3, y4))

    return perpendicular_lines, offset, t, x3, x4, y3, y4

def give_profile_to_segments(segment_linestring, points_with_profiles):

    #find the closest point to the segment.
    min_distance = float('inf')
    snapped_result = None
    closest_point = None
    profile_type_line = None
    for point_with_profile in points_with_profiles:
        point_feature, profile_type = point_with_profile
        point = point_feature.GetGeometryRef().Clone()

        #snapping operation
        point_np = np.array([point.GetPoint()[:2]])
        snap_result = snap_points(point_np, segment_linestring)[0] 
        
        # assuming offset gives us the distance ?
        if abs(snap_result.offset) < min_distance:
            min_distance = abs(snap_result.offset)
            closest_point = point_feature
            profile_type_line = profile_type

    return closest_point, profile_type_line

def create_lines_from_perp_lines(line_profiles, offset, t, x3, x4, y3, y4, previous_perpendicular_line, output_lines_layer):
    # Interpolate Z values from points based on the class
    z1_values = line_profiles.interp(offset)  # returns an array of Z values for profile 
 
    # Create a perpendicular normalized line from the interpolated points
    perpendicular_points = []
    for i in range(len(t)):
        # Calculate coordinates and interpolated z-values for each point
        x_curr = t[i] * (x4-x3) + x3
        y_curr = t[i] * (y4-y3) + y3
        z_curr = z1_values[i]
        perpendicular_points.append((x_curr, y_curr, z_curr))


    if previous_perpendicular_line:  # Only draw lines if there was a previous line
        for j in range(len(t)):  
            point1 = previous_perpendicular_line[j]
            point2 = perpendicular_points[j]

            # Check if z coordinate is NaN
            #if np.isnan(point1[2]) or np.isnan(point2[2]):
                #continue

            line_geometry = ogr.Geometry(ogr.wkbLineString25D)
            line_geometry.AddPoint(point1[0], point1[1], float(point1[2]))
            line_geometry.AddPoint(point2[0], point2[1], float(point2[2]))

            # Create output feature for crosssections
            output_line_feature = ogr.Feature(output_lines_layer.GetLayerDefn())
            output_line_feature.SetGeometry(line_geometry)
            output_lines_layer.CreateFeature(output_line_feature)

    return perpendicular_points

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('input_points_simpel', type=str, help='input points vector data source for simpel')
    argument_parser.add_argument('input_points_sammensat', type=str, help='input points vector data source for sammensat')
    #argument_parser.add_argument('input_points_opmaalt', type=str, help= 'input points vector data source for opmaalt')
    argument_parser.add_argument('input_polyline', type=str, help='input polyline vector data source')
    argument_parser.add_argument('output_lines', type=str, help='output geometry file for lines with Z')

    input_arguments = argument_parser.parse_args()

    input_points_simpel_path = input_arguments.input_points_simpel
    input_points_sammensat_path = input_arguments.input_points_sammensat
    #input_points_opmaalt_path = input_arguments.input_points_opmaalt
    input_polyline_path = input_arguments.input_polyline
    output_lines_path = input_arguments.output_lines

    #load the polyline layer and get the polyline geometry
    input_polyline_datasrc = ogr.Open(input_polyline_path)
    input_polyline_layer = input_polyline_datasrc.GetLayer()
    input_polyline_feature = input_polyline_layer.GetNextFeature()

    segments = []
    for input_polyline_feature in input_polyline_layer:
        stream_linestring = input_polyline_feature.GetGeometryRef()

        linestring_points = np.array(stream_linestring.GetPoints())[:,:2]

        for i in range(len(linestring_points) - 1):
            segment = (linestring_points[i], linestring_points[i+1])
            segments.append(segment)
       
    points = []
    for input_points_path, profile_type in [(input_points_simpel_path, 'RegulativProfilSimpel'), (input_points_sammensat_path, 'RegulativProfilSammensat')]:
        input_points_datasrc = ogr.Open(input_points_path)
        input_points_layer = input_points_datasrc.GetLayer()

        for point_feature in input_points_layer:
            point_feature.profile_type = profile_type
            points.append((point_feature, profile_type))
    
    #create the output file
    output_lines_driver = ogr.GetDriverByName("gpkg")
    output_lines_datasrc = output_lines_driver.CreateDataSource(output_lines_path)
    output_lines_datasrc.CreateLayer(
        "rendered_lines",
        srs=input_points_layer.GetSpatialRef(),
        geom_type=ogr.wkbLineString25D,
    )
    output_lines_layer = output_lines_datasrc.GetLayer()

    #create lists to store perpendicular lines in
    previous_perpendicular_line = None

    # Loop over all segments
    for segment in segments:
        #convert segment to LineString
        segment_linestring = ogr.Geometry(ogr.wkbLineString)
        segment_linestring.AddPoint(*segment[0])
        segment_linestring.AddPoint(*segment[1])

        #find the closest point and its profile for each segment
        closest_point, profile_type_line = give_profile_to_segments(segment_linestring, points)

        perpendicular_lines, offset, t, x3, x4, y3, y4 = create_perpendicular_lines_on_polylines(segment_linestring, length=10, interval=1)
        
        #associate each perpendicular line with a profile
        line_profiles = []
        for perp_line in perpendicular_lines:
            line_profiles.append((perp_line, get_profile(closest_point, profile_type_line)))
        
        profiles = get_profile(closest_point, profile_type_line)

        previous_perpendicular_line = create_lines_from_perp_lines(profiles, offset, t, x3, x4, y3, y4, previous_perpendicular_line, output_lines_layer)
    
        
    logging.info(f"processed {len(points)} points and created lines")

    output_lines_datasrc = None

if __name__ == '__main__':
    main()
