#from sampling import BoundingBox, get_raster_window, get_raster_interpolator

from itertools import groupby
from osgeo import gdal, ogr
import numpy as np
from tqdm import tqdm
import argparse
import logging
import random

from profile import RegulativProfilSimpel, RegulativProfilSammensat, OpmaaltProfil  # import interpolation classes

gdal.UseExceptions()
ogr.UseExceptions()

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('input_points_simpel', type=str, help='input points vector data source for simpel')
    argument_parser.add_argument('input_points_sammensat', type=str, help='input points vector data source for sammensat')
    #argument_parser.add_argument('input_points_opmaalt', type=str, help= 'input points vector data source for opmaalt')
    argument_parser.add_argument('output_lines', type=str, help='output geometry file for lines with Z')
    input_arguments = argument_parser.parse_args()

    input_points_simpel_path = input_arguments.input_points_simpel
    input_points_sammensat_path = input_arguments.input_points_sammensat
    #input_points_opmaalt_path = input_arguments.input_points_opmaalt
    output_lines_path = input_arguments.output_lines

    # Load all points from all files into memory and sort by "id" attribute
    points = []
    for input_points_path, profile_type in [(input_points_simpel_path, 'simpel'), (input_points_sammensat_path, 'sammensat')]: #,(input_points_opmaalt_path, 'opmaalt')
        input_points_datasrc = ogr.Open(input_points_path)
        input_points_layer = input_points_datasrc.GetLayer()

        for point_feature in input_points_layer:
            point_feature.profile_type = profile_type  # Add the profile type to the point feature
            points.append(point_feature)

    points.sort(key=lambda f: (f.GetField("laengdeprofillokalid"), f.GetField("regulativstationering")))
    #sample_points = points[:10]
    # Now we group the points based on their ID
    grouped_points = {k: sorted(g, key=lambda f: f.GetField("regulativstationering")) for k, g in groupby(points, key=lambda f: f.GetField("laengdeprofillokalid"))}

    # Create the output file
    output_lines_driver = ogr.GetDriverByName("gpkg")
    output_lines_datasrc = output_lines_driver.CreateDataSource(output_lines_path)
    output_lines_datasrc.CreateLayer(
        "rendered_lines",
        srs=input_points_layer.GetSpatialRef(),
        geom_type=ogr.wkbLineString25D,
    )
    output_lines_layer = output_lines_datasrc.GetLayer()
    previous_perpendicular_line= []
    # Iterate over each group of points and create a line for each pair of points within the group
    for point_group in grouped_points.values():
        previous_perpendicular_line = None
        for i in range(len(point_group) - 1):
            point1 = point_group[i]
            point2 = point_group[i + 1]

            # Determine which interpolation method to be used for the profile type
         
            if point1.profile_type == 'simpel':
                profile1 = RegulativProfilSimpel(point1)
            elif point1.profile_type == 'sammensat':
                profile1 = RegulativProfilSammensat(point1)
            #elif point1.profile_type == 'opmaalt':
                #profile1 = OpmaaltProfil(point1)

            if point2.profile_type == 'simpel':
                profile2 = RegulativProfilSimpel(point2)
            elif point2.profile_type == 'sammensat':
                profile2 = RegulativProfilSammensat(point2)
            #elif point2.profile_type == 'opmaalt':
                #profile2 = OpmaaltProfil(point2)

            point1_geometry = point1.GetGeometryRef()
            point2_geometry = point2.GetGeometryRef()

            #-----------------------------------------------------------------#
            # the method for creating the perpendicular lines was from here: https://gis.stackexchange.com/questions/50108/elevation-profile-10-km-each-side-of-line

            #Create perpendicular lines and get x and y coordinatoes for cross section data

            # Get coordinates for each point
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

            #unsure about how to determine the lenght of the lines
            length = 10

            # Calculate the coordinates of the endpoints of the perpendicular line
            x3 = x1 + vec_anti[0][0] * length / 2
            y3 = y1 + vec_anti[1][0] * length / 2
            x4 = x1 + vec_clock[0][0] * length / 2
            y4 = y1 + vec_clock[1][0] * length / 2
            
            #create multiple x,y values for the perpendicular line
            t = np.linspace(0,1)
            x = t* (x4-x3)+x3
            y = t* (y4-y3)+y3

            #find the middle
            offset = np.sqrt((x-x3)**2+ (y-y3)**2) - np.sqrt((x1-x3)**2+(y1-y3)**2)

            # Interpolate Z values from points based on the class
            z1_values = profile1.interp(offset)  # returns an array of Z values for profile 1
            z2_values = profile2.interp(offset)  # returns an array of Z values for profile 2
            
            # Create a perpendicular normalized line from the interpolated points
            line_geometry = ogr.Geometry(ogr.wkbLineString25D)
            # Create a perpendicular normalized line from the interpolated points
            perpendicular_points = []
            for i in range(len(t)):
                # Calculate coordinates and interpolated z-values for each point
                x_curr = t[i] * (x4-x3) + x3
                y_curr = t[i] * (y4-y3) + y3
                z_curr = (1-t[i]) * z1_values[i] + t[i] * z2_values[i]
                perpendicular_points.append((x_curr, y_curr, z_curr))

            if previous_perpendicular_line:  # Only draw lines if there was a previous line
                for j in range(len(t)):  
                    point1 = previous_perpendicular_line[j]
                    point2 = perpendicular_points[j]

                    line_geometry = ogr.Geometry(ogr.wkbLineString25D)
                    line_geometry.AddPoint(point1[0], point1[1], float(point1[2]))
                    line_geometry.AddPoint(point2[0], point2[1], float(point2[2]))

                    # Create output feature for crosssections
                    output_line_feature = ogr.Feature(output_lines_layer.GetLayerDefn())
                    output_line_feature.SetGeometry(line_geometry)
                    output_lines_layer.CreateFeature(output_line_feature)

            previous_perpendicular_line = perpendicular_points

    logging.info(f"processed {len(points)} points and created lines")

if __name__ == '__main__':
    main()

