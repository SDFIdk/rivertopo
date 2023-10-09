from itertools import groupby
from osgeo import gdal, ogr
import numpy as np
import argparse
import logging
from itertools import groupby
from operator import itemgetter


from profile import RegulativProfilSimpel, RegulativProfilSammensat, OpmaaltProfil # import interpolation classes
from snapping import snap_points

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
    t = np.linspace(0,1)
    x = t* (x4-x3)+x3
    y = t* (y4-y3)+y3

    # find the middle
    offset = np.sqrt((x-x3)**2+ (y-y3)**2) - np.sqrt((x1-x3)**2+(y1-y3)**2)

    return offset, t, x3, x4, y3, y4

def create_perpendicular_lines_on_polylines(stream_linestring, length=30, interval=1):
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

    #find the closest point to the segment.
    min_distance = float('inf')
    snapped_result = None
    closest_point = None
    profile_type_line = None
    for point_with_profile in points_with_profiles:
        point_feature, profile_type, laengdeprofillokalid, regulativstationering = point_with_profile
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

    segments_dict = {}
    for input_polyline_feature in input_polyline_layer:
        stream_linestring = input_polyline_feature.GetGeometryRef()

        linestring_points = np.array(stream_linestring.GetPoints())[:,:2]
        laengdeprofillokalid = input_polyline_feature.GetField('laengdeprofillokalid')
        sort_value = input_polyline_feature.GetField('sort')
        print('printing laengdeprofilloaklid:',laengdeprofillokalid)

        if laengdeprofillokalid not in segments_dict:
            segments_dict[laengdeprofillokalid] = []
    
        # if laengdeprofillokalid not in segments:
        #     segments[laengdeprofillokalid] = []
    
        # for i in range(len(linestring_points) - 1):
        #     segment = (linestring_points[i], linestring_points[i+1])
        #     segments.append(segment)
    
        for i in range(len(linestring_points) - 1):
            segment = (linestring_points[i], linestring_points[i+1])
            segments_dict[laengdeprofillokalid].append((sort_value,segment))
        
    # Sort segments within each laengdeprofillokalid group based on the sort attribute
    for laengdeprofillokalid, segments in segments_dict.items():
        segments.sort(key=lambda x: x[0])
        segments_dict[laengdeprofillokalid] = [segment for sort_value, segment in segments]
        
    points = []
    for input_points_path, profile_type in [(input_points_simpel_path, 'RegulativProfilSimpel')]: #, (input_points_sammensat_path, 'RegulativProfilSammensat'), (input_points_opmaalt_path, 'OpmaaltProfil')]:
        input_points_datasrc = ogr.Open(input_points_path)
        input_points_layer = input_points_datasrc.GetLayer()

        for point_feature in input_points_layer:
            laengdeprofillokalid = point_feature.GetField('laengdeprofillokalid')
            regulativstationering = point_feature.GetField('regulativstationering')
            point_feature.profile_type = profile_type
            points.append((point_feature, profile_type, laengdeprofillokalid, regulativstationering))
    
    # Group points by 'laengdeprofillokalid' and sort them by 'regulativstationering'
    points_sorted = sorted(points, key=itemgetter(3))  # Sort by 'regulativstationering'
    points_grouped = groupby(points_sorted, key=itemgetter(2))  # Group by 'laengdeprofillokalid'

    
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

    # Loop over all groups
    for laengdeprofillokalid, segments in segments_dict.items():
        previous_perpendicular_line = None
    
        # Define which segments belong to which 'laengdeprofillokalid'
        if laengdeprofillokalid == '{0ED96FD8-94D4-4550-912C-2C38ADCCCF8F}':
            segment_indices = [0, 1, 2, 3]  
        elif laengdeprofillokalid == '{E368FFCE-965B-46E8-9DFE-40D682B6C7E2}':
            segment_indices = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  
        else:
            segment_indices = []  

        for segment_index in segment_indices:
            segment = segments[segment_index]


        # Loop over all segments within the current group
        for segment_index in segment_indices:
            segment = segments[segment_index]

            # Loop over all segments
            #for segment in segments:
                #convert segment to LineString
            segment_linestring = ogr.Geometry(ogr.wkbLineString)
            segment_linestring.AddPoint(*segment[0])
            segment_linestring.AddPoint(*segment[1])

            #find the closest point and its profile for each segment
            closest_point, profile_type_line = give_profile_to_segments(segment_linestring, points)

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
