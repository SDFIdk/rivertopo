from osgeo import gdal, ogr
import numpy as np
import argparse
import logging
from rivertopo.profile import RegulativProfilSimpel, RegulativProfilSammensat, OpmaaltProfil # import interpolation classes
from rivertopo.snapping import snap_points
from numpy import array_equal
from scipy.interpolate import RegularGridInterpolator

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

def calculate_stationing(linestring, snap_results):
    """
    Calculate the stationing (distance along the linestring) for each snapped point.
    
    :param linestring: The linestring geometry of the river.
    :param snap_results: List of SnapResult objects for the snapped points.
    :return: List of stationing values for each snapped point.
    """
    linestring_points = np.array(linestring.GetPoints())[:,:2]
    stationing = []

    for snap_list in snap_results:

        snap_result= snap_list[0]

        segment_index = snap_result.segment
        param = snap_result.param

        # Calculate the distance along the linestring up to the start of the snapped segment
        segment_distances = np.hypot(np.diff(linestring_points[:segment_index + 1, 0]), 
                                     np.diff(linestring_points[:segment_index + 1, 1]))
        distance_to_segment = np.sum(segment_distances)

        # Calculate the distance along the snapped segment proportional to the parametric position
        segment_length = np.hypot(linestring_points[segment_index + 1][0] - linestring_points[segment_index][0], 
                                  linestring_points[segment_index + 1][1] - linestring_points[segment_index][1])
        distance_along_segment = param * segment_length

        total_stationing = distance_to_segment + distance_along_segment
        stationing.append(total_stationing)
    #breakpoint()
    return stationing

def create_perpendicular_lines(point1_geometry, point2_geometry, length=30):
    """
    Create perpendicular lines between two point geometries at a specified length.

    :param point1_geometry: The first point geometry.
    :param point2_geometry: The second point geometry.
    :param length: The length of the perpendicular line (default is 30 meters).
    :return: Offsets, parameter t, and coordinates of the endpoints of the perpendicular line.
    """
            # # Check the type of the two points
        # if point1_geometry.GetGeometryName() == 'LINESTRING' and point2_geometry.GetGeometryName() == 'LINESTRING':
        #     # If they are LINESTRINGs, get the center point
        #     x1, y1 = calculate_center(point1_geometry)[0], calculate_center(point1_geometry)[1]
        #     x2, y2 = calculate_center(point2_geometry)[0], calculate_center(point2_geometry)[1]
        # else:
        #     # If not, just get the first point as usual
        #     x1, y1 = point1_geometry.GetX(), point1_geometry.GetY()
        #     x2, y2 = point2_geometry.GetX(), point2_geometry.GetY()


    # Get the coordinates of the two points
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

    # Create multiple x, y values for the perpendicular line
    t = np.linspace(0, 1, num=100)
    x = t * (x4 - x3) + x3
    y = t * (y4 - y3) + y3

    # Find the middle
    offset = np.sqrt((x - x3) ** 2 + (y - y3) ** 2) - np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)

    return offset, t, x3, x4, y3, y4


def create_perpendicular_lines_on_polylines(stream_linestring, stations, length=30):
    """
    Create perpendicular lines over a given polyline.
    
    :param stream_linestring: The input stream linestring.
    :param length: Length of the perpendicular line (default is 30 meters).
    :param interval: The interval between points (default is 1).
    :return: List of perpendicular lines and their attributes.
    """
    perpendicular_lines = []
    cumulative_distance = 0

    # First, create perpendicular lines at each point on the linestring
    for i in range(stream_linestring.GetPointCount() - 1):
        point1 = stream_linestring.GetPoint(i)
        point2 = stream_linestring.GetPoint(i + 1)

        point1_geometry = ogr.Geometry(ogr.wkbPoint)
        point1_geometry.AddPoint(*point1)
        point2_geometry = ogr.Geometry(ogr.wkbPoint)
        point2_geometry.AddPoint(*point2)

        offset, t, x3, x4, y3, y4 = create_perpendicular_lines(point1_geometry, point2_geometry, length=length)
        perpendicular_lines.append((offset, t, x3, x4, y3, y4, cumulative_distance))

        # Update cumulative distance
        segment_length = np.hypot(point2[0] - point1[0], point2[1] - point1[1])
        cumulative_distance += segment_length

    # Next, create additional perpendicular lines at specified stations
    for station in stations:
        # Find the segment where this station falls
        segment_distance = 0
        for i in range(stream_linestring.GetPointCount() - 1):
            point1 = stream_linestring.GetPoint(i)
            point2 = stream_linestring.GetPoint(i + 1)
            segment_length = np.hypot(point2[0] - point1[0], point2[1] - point1[1])

            if segment_distance + segment_length >= station:
                # Station falls within this segment
                point1_geometry = ogr.Geometry(ogr.wkbPoint)
                point1_geometry.AddPoint(*point1)
                point2_geometry = ogr.Geometry(ogr.wkbPoint)
                point2_geometry.AddPoint(*point2)

                offset, t, x3, x4, y3, y4 = create_perpendicular_lines(point1_geometry, point2_geometry, length=length)
                perpendicular_lines.append((offset, t, x3, x4, y3, y4, station))
                break

            segment_distance += segment_length


    perpendicular_lines = sorted(perpendicular_lines, key=lambda x: x[-1])

    return perpendicular_lines


def create_lines_with_z(current_line_data, previous_line_data, output_lines_layer):

    if previous_line_data is None or current_line_data is None:
        return

    t_curr, z_values_curr, x3_curr, x4_curr, y3_curr, y4_curr, _ = current_line_data
    t_prev, z_values_prev, x3_prev, x4_prev, y3_prev, y4_prev, _ = previous_line_data

    for i in range(len(t_curr)):
        # Calculate coordinates for point i in the current and previous lines
        x1 = x3_prev + t_prev[i] * (x4_prev - x3_prev)
        y1 = y3_prev + t_prev[i] * (y4_prev - y3_prev)
        z1 = z_values_prev[i]

        x2 = x3_curr + t_curr[i] * (x4_curr - x3_curr)
        y2 = y3_curr + t_curr[i] * (y4_curr - y3_curr)
        z2 = z_values_curr[i]

        # Create line geometry
        line_geometry = ogr.Geometry(ogr.wkbLineString25D)
        line_geometry.AddPoint(x1, y1, z1)
        line_geometry.AddPoint(x2, y2, z2)

        # Create output feature for cross sections
        output_line_feature = ogr.Feature(output_lines_layer.GetLayerDefn())
        output_line_feature.SetGeometry(line_geometry)
        output_lines_layer.CreateFeature(output_line_feature)

    # Update the previous line data for the next iteration
    previous_perpendicular_line = current_line_data


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


    # Process each polyline feature
    for input_polyline_feature in input_polyline_layer:
        stream_linestring = input_polyline_feature.GetGeometryRef()

        all_snap_results = []
        #breakpoint()
        for point in points_sorted:
            point_feature, profile_type, regulativstationering = point
            point_att = point_feature.GetGeometryRef().Clone()
            points_np = np.array([point_att.GetPoint()[:2]])
        
            snapping_results = snap_points(points_np, stream_linestring)
            all_snap_results.append(snapping_results)

        station_to_profile_map = {}
        stations = calculate_stationing(stream_linestring, all_snap_results)

        for station_value, point_info, snap_result in zip(stations, points_sorted, all_snap_results):
            point_feature, profile_type, regulativstationering = point_info
            station_to_profile_map[station_value] = {
                'profile_type': profile_type, 
                'point_feature': point_feature, 
                'regulativstationering': regulativstationering,
                'snap_result': snap_result
            }
      
        # Create perpendicular lines for this polyline
        perpendicular_lines = create_perpendicular_lines_on_polylines(stream_linestring, stations, length=30)

        known_stations = []
        known_z_values = []
        perpendicular_lines_z_values = []

        # Interpolate Z-values along each perpendicular line in point stations
        for perp_line in perpendicular_lines:
            offset, t, x3, x4, y3, y4, perp_line_station = perp_line
            
             # Check if this line's station is in the station_to_profile_map
            if perp_line_station in station_to_profile_map:
                # Get profile info for this station
                profile_info = station_to_profile_map[perp_line_station]
                profile = get_profile(profile_info['point_feature'], profile_info['profile_type'])

                # Interpolate Z-value using the profile and offset
                z_values = profile.interp(offset)
                
                # Store this station and its Z-values for later interpolation
                known_stations.append(perp_line_station)
                known_z_values.append(z_values)
                perpendicular_lines_z_values.append((perp_line, z_values))
        

        min_station = min(known_stations, default=0)
        max_station = max(known_stations, default=0)
        # Second pass: Interpolate Z-values for other lines
        for perp_line in perpendicular_lines:
            offset, t, x3, x4, y3, y4, perp_line_station = perp_line

            if perp_line_station not in known_stations:

                clipped_station = max(min(perp_line_station, max_station), min_station)

                # Perform interpolation
                x = np.array(known_stations)
                y = np.arange(100) 
                z = np.array(known_z_values)

                # Setup interpolation function
                interp_func = RegularGridInterpolator((x, y), z)

                # Create points for interpolation
                interp_points = np.array([[clipped_station, yi] for yi in y])

                # Interpolate Z-values
                interpolated_z_values = interp_func(interp_points)    
                perpendicular_lines_z_values.append((perp_line, interpolated_z_values))

    previous_perpendicular_line = None

    # Iterate through perpendicular lines to create 3D lines
    for line_data in perpendicular_lines_z_values:
        perp_line, z_values = line_data
        offset, t, x3, x4, y3, y4, perp_line_station = perp_line

        current_line_data = (t, z_values, x3, x4, y3, y4, perp_line_station)
        
        if previous_perpendicular_line is not None:
            # Call the function with the current and previous line's data
            create_lines_with_z(current_line_data, previous_perpendicular_line, output_lines_layer)

        # Update previous line data for next iteration
        previous_perpendicular_line = current_line_data


    logging.info(f"processed {len(points)} points and created lines")

    output_lines_datasrc = None

if __name__ == '__main__':
    main()