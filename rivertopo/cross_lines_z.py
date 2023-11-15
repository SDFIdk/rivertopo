from osgeo import gdal, ogr
import numpy as np
import argparse
import logging
from profile import RegulativProfilSimpel, RegulativProfilSammensat, OpmaaltProfil # import interpolation classes
from snapping import snap_points
from numpy import array_equal
from topology import *
import shapely

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

def create_snapped_points(stream_linestring, points_with_profiles):
    """
    Create new point geometries on stream linestring with profiles.
    
    :param segment_linestring: The segment represented as a linestring.
    :param points_with_profiles: List of points associated with profiles.
    :return: Snapped point to the linestring
    """

    snapped_points = []
    all_snap_results = []
    for point_with_profile in points_with_profiles:
        point_feature, profile_type, regulativstationering = point_with_profile
        point = point_feature.GetGeometryRef().Clone()

        points_np = np.array([point.GetPoint()[:2]])

        # Get the snap results for all points
        snap_results = snap_points(points_np, stream_linestring)[0]
        #segment_index = int(snap_results[0].segment)
        segment_start = np.array(stream_linestring.GetPoint(int(snap_results.segment))[:2])
        segment_end = np.array(stream_linestring.GetPoint(int(snap_results.segment + 1))[:2])
        
        closest_point_np = segment_start + snap_results.param * (segment_end - segment_start)
        
        # Create a new point geometry
        new_point_geom = ogr.Geometry(ogr.wkbPoint)
        new_point_geom.AddPoint(closest_point_np[0], closest_point_np[1])
    
        # Clone the original feature and set new geometry
        new_point_feature = point_feature.Clone()
        new_point_feature.SetGeometry(new_point_geom)

        snapped_points.append((new_point_feature, profile_type, regulativstationering))

        all_snap_results.append(snap_results)

    return snapped_points, all_snap_results


def create_perpendicular_lines(point1_geometry, point2_geometry, segment_start, segment_end, is_snapped_point=False, length=30):
    """
    Create perpendicular lines between two point geometries at a specified length.

    If a snapped point is provided, the perpendicular line is created through that point
    using the direction of the segment it snapped to.

    :param point1_geometry: The first point geometry or the snapped point geometry.
    :param point2_geometry: The second point geometry or None if is_snapped_point is True.
    :param is_snapped_point: Boolean to indicate if the first point is a snapped point.
    :param length: The length of the perpendicular line (default is 30 meters).
    :return: Offsets, parameter t, and coordinates of the endpoints of the perpendicular line.
    """
    # If it is a snapped point, handle differently
    if is_snapped_point:
        # Calculate the direction vector of the segment
        segment_vector = np.array([segment_end[0] - segment_start[0], segment_end[1] - segment_start[1]])

        # Rotate the vector 90 degrees to get the perpendicular direction
        rot_matrix = np.array([[0, -1], [1, 0]])
        perp_vector = np.dot(rot_matrix, segment_vector)

        # Normalize the perpendicular vector
        perp_vector = perp_vector / np.linalg.norm(perp_vector)

        # Get the coordinates of the snapped point
        x1, y1 = point1_geometry.GetX(), point1_geometry.GetY()

        # Calculate the endpoints of the perpendicular line
        x3 = x1 + perp_vector[0] * length / 2
        y3 = y1 + perp_vector[1] * length / 2
        x4 = x1 - perp_vector[0] * length / 2
        y4 = y1 - perp_vector[1] * length / 2

    else:
        # If not a snapped point, proceed as before
        x1, y1 = point1_geometry.GetX(), point1_geometry.GetY()
        x2, y2 = point2_geometry.GetX(), point2_geometry.GetY()

    # Calculate the displacement vector for the original line
        vec = np.array([[x2 - x1], [y2 - y1]])

        # Rotate the vector 90 degrees clockwise and 90 degrees counterclockwise
        rot_anti = np.array([[0, -1], [1, 0]])
        rot_clock = np.array([[0, 1], [-1, 0]])
        vec_anti = np.dot(rot_anti, vec)
        vec_clock = np.dot(rot_clock, vec)

        # Normalize the perpendicular vectors
        len_anti = np.linalg.norm(vec_anti)
        len_clock = np.linalg.norm(vec_clock)
        vec_anti = vec_anti / len_anti
        vec_clock = vec_clock / len_clock

        # Calculate the coordinates of the endpoints of the perpendicular line
        x3 = x1 + vec_anti[0][0] * length / 2
        y3 = y1 + vec_anti[1][0] * length / 2
        x4 = x1 + vec_clock[0][0] * length / 2
        y4 = y1 + vec_clock[1][0] * length / 2

        # # Check the type of the two points
        # if point1_geometry.GetGeometryName() == 'LINESTRING' and point2_geometry.GetGeometryName() == 'LINESTRING':
        #     # If they are LINESTRINGs, get the center point
        #     x1, y1 = calculate_center(point1_geometry)[0], calculate_center(point1_geometry)[1]
        #     x2, y2 = calculate_center(point2_geometry)[0], calculate_center(point2_geometry)[1]
        # else:
        #     # If not, just get the first point as usual
        #     x1, y1 = point1_geometry.GetX(), point1_geometry.GetY()
        #     x2, y2 = point2_geometry.GetX(), point2_geometry.GetY()

    # create multiple x,y values for the perpendicular line
    t = np.linspace(0,1, num=100)
    x = t* (x4-x3)+x3
    y = t* (y4-y3)+y3

    # find the middle
    offset = np.sqrt((x-x3)**2+ (y-y3)**2) - np.sqrt((x1-x3)**2+(y1-y3)**2)

    return offset, t, x3, x4, y3, y4

def get_segment_points(linestring, segment_index):
    """
    Retrieve the start and end points of a segment from a linestring based on the segment index.
    
    :param linestring: The linestring geometry.
    :param segment_index: The index of the segment.
    :return: The start and end points of the segment.
    """
    segment_index = int(segment_index)
    start_point = linestring.GetPoint(segment_index)
    end_point = linestring.GetPoint(segment_index + 1)

    return start_point, end_point

def create_perpendicular_lines_on_polylines(stream_linestring, snapped_points, snap_results, length=30, interval=1):
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
    snap_index = 0

    # Iterate over each pair of consecutive points in the stream_linestring
    for i in range(num_points - 1):
        point1_coords = stream_linestring.GetPoint(i)
        point2_coords = stream_linestring.GetPoint(i + 1)
        # Create geometry objects from point coordinates
        point1_geometry = ogr.Geometry(ogr.wkbPoint)
        point2_geometry = ogr.Geometry(ogr.wkbPoint)
        point1_geometry.AddPoint(*point1_coords)
        point2_geometry.AddPoint(*point2_coords)

        if snap_index < len(snap_results) and i == snap_results[snap_index].segment:
            # Get the segment's start and end points
            segment_start = stream_linestring.GetPoint(int(snap_results[snap_index].segment))
            segment_end = stream_linestring.GetPoint(int(snap_results[snap_index].segment + 1))
            # Get the geometry of the snapped point
            snapped_point_geom = snapped_points[snap_index][0].GetGeometryRef()
            # Create the perpendicular line through the snapped point
            offset, t, x3, x4, y3, y4 = create_perpendicular_lines(
                snapped_point_geom, None, segment_start, segment_end, is_snapped_point=True, length=length)
            perpendicular_lines.append((offset, t, x3, x4, y3, y4))
            snap_index += 1
     
        else:
            # Create a regular perpendicular line
            offset, t, x3, x4, y3, y4 = create_perpendicular_lines(
                point1_geometry, point2_geometry, None, None, is_snapped_point=False, length=length)
            perpendicular_lines.append((offset, t, x3, x4, y3, y4))
        
        #print(perpendicular_lines)

    return perpendicular_lines #, offset, t, x3, x4, y3, y4


def interpolate_perpendicular_lines_from_sp(snapped_points, perpendicular_lines): #, previous_perpendicular_line, output_lines_layer):
    """
    Generate interpolated lines from intersecting snapped points
    
    :param snapped_points: List of snapped points with profile data.
    :param perpendicular_lines: List of perpendicular lines and their attributes.
    :param output_lines_layer: The layer where output lines should be saved.
    :return: The list of generated perpendicular points.
    """

    offset, t, x3, x4, y3, y4, = perpendicular_lines
    # Create a LineString geometry for the perpendicular line
    perp_line_geom = ogr.Geometry(ogr.wkbLineString)
    perp_line_geom.AddPoint(x3, y3)
    perp_line_geom.AddPoint(x4, y4)


    intersecting_points_z = []

    for point_feature, profile_type, regulativstationering in snapped_points:        
        if point_feature.GetGeometryRef().Intersects(perp_line_geom):
                
            profile = get_profile(point_feature, profile_type)
            z_values = profile.interp(offset)
            intersecting_points_z.extend([(t[j] * (x4 - x3) + x3, t[j] * (y4 - y3) + y3, z_values[j]) for j in range(len(t))])

    return intersecting_points_z


def sort_segments(segments):
    # sorted_segments = get_layer_topology(segments)
    # print(sorted_segments)
    # return sorted_segments
    # Sort segments based on starting x-coordinate, then by starting y-coordinate
    return sorted(segments, key=lambda segment: (segment[0][0], segment[0][1]))


def create_lines_with_z(perpendicular_points, previous_perpendicular_line, output_lines_layer):
   
    for j, point2 in enumerate(perpendicular_points):
        #print(point2)
        if previous_perpendicular_line is not None:  # Only draw lines if there was a previous line
            point1 = previous_perpendicular_line[j]
            #print(point1)
            line_geometry = ogr.Geometry(ogr.wkbLineString25D)
            line_geometry.AddPoint(point1[0], point1[1], float(point1[2]))
            line_geometry.AddPoint(point2[0], point2[1], float(point2[2]))
            #print(line_geometry)
            # Create output feature for crosssections
            output_line_feature = ogr.Feature(output_lines_layer.GetLayerDefn())
            output_line_feature.SetGeometry(line_geometry)
            output_lines_layer.CreateFeature(output_line_feature)
            #print(output_lines_layer)

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


    points = []
    for input_points_path, profile_type in [(input_points_simpel_path, 'RegulativProfilSimpel')]: #, (input_points_sammensat_path, 'RegulativProfilSammensat'), (input_points_opmaalt_path, 'OpmaaltProfil')]:
        input_points_datasrc = ogr.Open(input_points_path)
        input_points_layer = input_points_datasrc.GetLayer()

        for point_feature in input_points_layer:
            point_feature.profile_type = profile_type
            regulativstationering = point_feature.GetField('regulativstationering')
            points.append((point_feature, profile_type, regulativstationering))
    
    # Sort points by 'regulativstationering'
    points_sorted = sorted(points, key=lambda x: x[2])

    #create the output file
    output_lines_driver = ogr.GetDriverByName("gpkg")
    output_lines_datasrc = output_lines_driver.CreateDataSource(output_lines_path)
    output_lines_datasrc.CreateLayer(
        "rendered_lines",
        srs=input_points_layer.GetSpatialRef(),
        geom_type=ogr.wkbLineString25D,
    )
    output_lines_layer = output_lines_datasrc.GetLayer()


    last_offsets = []
    last_z_values = []
    previous_perpendicular_line = None

    # Process each polyline feature
    for input_polyline_feature in input_polyline_layer:
        stream_linestring = input_polyline_feature.GetGeometryRef()

        # Snapping points to this polyline
        snapped_points, snap_results = create_snapped_points(stream_linestring, points_sorted)

        # Create perpendicular lines for this polyline
        perpendicular_lines = create_perpendicular_lines_on_polylines(stream_linestring, snapped_points, snap_results, length=30, interval=1)
 
        # Interpolating and creating lines for each perpendicular line
        for perp_line in perpendicular_lines:
            current_perpendicular_points = interpolate_perpendicular_lines_from_sp(snapped_points, perp_line)
            
            if current_perpendicular_points:
                last_offsets.append(perp_line[0])
                last_z_values.extend([point[2] for point in current_perpendicular_points])
                #print(last_z_values)
            
            else:
                # Handle non-intersecting lines
                offset, t, x3, x4, y3, y4 = perp_line
                offset_flat = np.array(offset).flatten()
                
                if last_offsets and last_z_values:
                    interpolated_z_values = np.interp(offset_flat, np.array(last_offsets).flatten(), np.array(last_z_values).flatten())
                    #print(interpolated_z_values)
                    current_perpendicular_points = [(t[j] * (x4 - x3) + x3, t[j] * (y4 - y3) + y3, interpolated_z_values[j]) for j in range(len(t))]
                
                #else:
                    # Handle case where no known Z values are available
                    #current_perpendicular_points= [(t[j] * (x4 - x3) + x3, t[j] * (y4 - y3) + y3, 999) for j in range(len(t))]  # Default Z value
                
           
            if previous_perpendicular_line:
                create_lines_with_z(current_perpendicular_points, previous_perpendicular_line, output_lines_layer)
                breakpoint()

            previous_perpendicular_line = current_perpendicular_points

    logging.info(f"processed {len(points)} points and created lines")

    output_lines_datasrc = None

if __name__ == '__main__':
    main()