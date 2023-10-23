from osgeo import gdal, ogr
import numpy as np
import argparse
import logging
from rivertopo.profile import RegulativProfilSimpel, RegulativProfilSammensat, OpmaaltProfil # import interpolation classes
from rivertopo.snapping import snap_points
from operator import itemgetter
from numpy import array_equal

"""
This script generates line objects with z-values by interpolating z-values from cross-sectional data. 
The data comes from dataforsyningen.dk and is applied over a polyline, specifically 
the middle of the stream from geodanmark. The interpolation is conducted based on given profiles which 
are associated with river segments to achieve accurate representation.

"""

gdal.UseExceptions()
ogr.UseExceptions()

def get_profile(point, profile_type):
    if profile_type == 'RegulativProfilSimpel':
        return RegulativProfilSimpel(point)
    elif profile_type == 'RegulativProfilSammensat':
        return RegulativProfilSammensat(point)
    elif profile_type == 'OpmaaltProfil':
        return OpmaaltProfil(point)

def calculate_center(geometry_ref):
    geometry_coords = np.array(geometry_ref.GetPoints())
    z_min_indices = np.argmin(geometry_coords[:,2])
    z_min_coords = geometry_coords[z_min_indices,:].reshape(-1, 3) 
    thalweg_coord = np.mean(z_min_coords, axis=0)

    return thalweg_coord

def create_perpendicular_lines(point1_geometry, point2_geometry, length=30):
    """
    Create perpendicular lines between two point geometries at a specified length.

    The perpendicular lines are created to interpolate interating over the middle 
    of a river segment, so the middle of the cross sections will be based off of these. 

    :param point1_geometry: The first point geometry. 
    :param point2_geometry: The second point geometry.
    :param length: The length of the perpendicular line (default is 30 meters).
    :return: Offsets, parameter t, and coordinates of the endpoints of the perpendicular line.
    """

    # Check the type of the two points
    if point1_geometry.GetGeometryName() == 'LINESTRING' and point2_geometry.GetGeometryName() == 'LINESTRING':
        # If they are LINESTRINGs, get the center point
        x1, y1 = calculate_center(point1_geometry)[0], calculate_center(point1_geometry)[1]
        x2, y2 = calculate_center(point2_geometry)[0], calculate_center(point2_geometry)[1]
    else:
        # If not, just get the first point as usual
        x1, y1 = point1_geometry.GetX(), point1_geometry.GetY()
        x2, y2 = point2_geometry.GetX(), point2_geometry.GetY()

    # Calculate the displacement vector for the original line
    vec = np.array([[x2 - x1,], [y2 - y1,]])

    # Rotate the vector 90 deg clockwise and 90 deg counter clockwise
    rot_anti = np.array([[0, -1], [1, 0]])
    rot_clock = np.array([[0, 1], [-1, 0]])
    vec_anti = np.dot(rot_anti, vec)
    vec_clock = np.dot(rot_clock, vec)

    # Normalize the perpendicular vectors
    len_anti = ((vec_anti**2).sum())**0.5
    vec_anti = vec_anti/len_anti
    len_clock = ((vec_clock**2).sum())**0.5
    vec_clock = vec_clock/len_clock

    # Calculate the coordinates of the endpoints of the perpendicular line
    x3 = x1 + vec_anti[0][0] * length / 2
    y3 = y1 + vec_anti[1][0] * length / 2
    x4 = x1 + vec_clock[0][0] * length / 2
    y4 = y1 + vec_clock[1][0] * length / 2

    # create multiple x,y values for the perpendicular line
    t = np.linspace(0,1, num=100)
    x = t* (x4-x3)+x3
    y = t* (y4-y3)+y3

    # find the middle
    offset = np.sqrt((x-x3)**2+ (y-y3)**2) - np.sqrt((x1-x3)**2+(y1-y3)**2)

    return offset, t, x3, x4, y3, y4


def create_perpendicular_lines_on_polylines(stream_linestring, length=30, interval=1):
    """
    Create perpendicular lines over a given polyline.
    
    :param stream_linestring: The input stream linestring.
    :param length: Length of the perpendicular line (default is 30 meters).
    :param interval: The interval between points (default is 1).
    :return: List of perpendicular lines and their attributes.
    """

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
        offset, t, x3, x4, y3, y4 = create_perpendicular_lines(point1_geometry, point2_geometry, length=30)
        
        perpendicular_lines.append((offset, t, x3, x4, y3, y4))

    return perpendicular_lines, offset, t, x3, x4, y3, y4

def give_profile_to_segments(segment_linestring, points_with_profiles):
    """
    Associate profiles with segments based on their proximity.
    
    :param segment_linestring: The segment represented as a linestring.
    :param points_with_profiles: List of points associated with profiles.
    :return: Closest point and its associated profile type.
    """
    #find the closest point to the segment.
    min_distance = float('inf')
    snapped_result = None
    closest_point = None
    profile_type_line = None
    for point_with_profile in points_with_profiles:
        point_feature, profile_type, regulativstationering = point_with_profile
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
    #return start_point, end_point, start_profile_type, end_profile_type

def create_lines_from_perp_lines(line_profiles, offset, t, x3, x4, y3, y4, previous_perpendicular_line, output_lines_layer):
    """
    Generate interpolated lines from given perpendicular lines.
    
    :param line_profiles: List of profiles.
    :param offset, t, x3, x4, y3, y4: Attributes of the perpendicular lines.
    :param previous_perpendicular_line: The last generated perpendicular line.
    :param output_lines_layer: The layer where output lines should be saved.
    :return: The list of generated perpendicular points.
    """
    
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

        # Linear interpolation between start_point and end_point
        # fraction = i / len(t)
        # z_curr = (1 - fraction) * start_z_value + fraction * end_z_value
        # perpendicular_points.append((x_curr, y_curr, z_curr))


    if previous_perpendicular_line:  # Only draw lines if there was a previous line
        for j in range(len(t)):  
            point1 = previous_perpendicular_line[j]
            point2 = perpendicular_points[j]

            # Check if z coordinate is NaN
            if np.isnan(point1[2]) or np.isnan(point2[2]):
                continue
        
            # Interpolate Z-values between the two points
            z_values = np.linspace(point1[2], point2[2], 2)
            #print(z_values)

            # Existing Z-values (start and end)
            #z_values = [point1[2], point2[2]]

            # Interpolated Z-values
            #z_interpolated = np.interp([0, 1], [0, 1], z_values)

            line_geometry = ogr.Geometry(ogr.wkbLineString25D)
            # line_geometry.AddPoint(point1[0], point1[1], float(point1[2]))
            # line_geometry.AddPoint(point2[0], point2[1], float(point2[2]))

            line_geometry.AddPoint(point1[0], point1[1], float(z_values[0]))
            line_geometry.AddPoint(point2[0], point2[1], float(z_values[1]))

            # Create output feature for crosssections
            output_line_feature = ogr.Feature(output_lines_layer.GetLayerDefn())
            output_line_feature.SetGeometry(line_geometry)
            output_lines_layer.CreateFeature(output_line_feature)

    return perpendicular_points

################ Sorting function of vandløbsmidte segments
#sortere efter koordinator



def main():
    """
    The main function 
    
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('input_points_simpel', type=str, help='input points vector data source for simpel')
    #argument_parser.add_argument('input_points_sammensat', type=str, help='input points vector data source for sammensat')
    #argument_parser.add_argument('input_points_opmaalt', type=str, help= 'input points vector data source for opmaalt')
    argument_parser.add_argument('input_polyline', type=str, help='input polyline vector data source')
    argument_parser.add_argument('output_lines', type=str, help='output geometry file for lines with Z')

    input_arguments = argument_parser.parse_args()

    input_points_simpel_path = input_arguments.input_points_simpel
    #input_points_sammensat_path = input_arguments.input_points_sammensat
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
    for input_points_path, profile_type in [(input_points_simpel_path, 'RegulativProfilSimpel')]: #, (input_points_sammensat_path, 'RegulativProfilSammensat'), (input_points_opmaalt_path, 'OpmaaltProfil')]:
        input_points_datasrc = ogr.Open(input_points_path)
        input_points_layer = input_points_datasrc.GetLayer()

        for point_feature in input_points_layer:
            point_feature.profile_type = profile_type
            regulativstationering = point_feature.GetField('regulativstationering')
            points.append((point_feature, profile_type, regulativstationering))
    
    ################ Sorting of reg profiles upstream ???????

    points_sorted = sorted(points, key=itemgetter(2))
    # print("points",points)
    # print("sorted",points_sorted)
   
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
        closest_point, profile_type_line = give_profile_to_segments(segment_linestring, points_sorted)

        perpendicular_lines, offset, t, x3, x4, y3, y4 = create_perpendicular_lines_on_polylines(segment_linestring, length=30, interval=1)
                
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