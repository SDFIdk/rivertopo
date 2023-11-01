from rivertopo import topology
from rivertopo.topology import ConnectedPoints, get_layer_topology

from osgeo import ogr
import numpy as np

ogr.UseExceptions()

def test_get_layer_topology(centerline_layer):
    layer_defn = centerline_layer.GetLayerDefn()

    river_points = [
        [
            (0.0, 0.0, 10.0),
            (4.0, 3.0, 9.5),
            (5.0, 3.0, 9.0),
            (7.0, 2.0, 8.5),
            (8.0, 0.0, 8.0),
        ],
        [
            (1.0, 4.0, 10.0),
            (4.0, 3.0, 9.5),
        ],
        [
            (7.0, 2.0, 8.5),
            (9.0, 3.0, 8.0),
            (10.0, 3.0, 7.5),
        ],
    ]

    expected_connections = {
        river_points[0][0][:2]: ConnectedPoints(downstream=set([river_points[0][1][:2]])),
        river_points[0][1][:2]: ConnectedPoints(upstream=set([river_points[0][0][:2], river_points[1][0][:2]]), downstream=set([river_points[0][2][:2]])),
        river_points[0][2][:2]: ConnectedPoints(upstream=set([river_points[0][1][:2]]), downstream=set([river_points[0][3][:2]])),
        river_points[0][3][:2]: ConnectedPoints(upstream=set([river_points[0][2][:2]]), downstream=set([river_points[0][4][:2], river_points[2][1][:2]])),
        river_points[0][4][:2]: ConnectedPoints(upstream=set([river_points[0][3][:2]])),
        river_points[1][0][:2]: ConnectedPoints(downstream=set([river_points[0][1][:2]])),
        river_points[2][1][:2]: ConnectedPoints(upstream=set([river_points[0][3][:2]]), downstream=set([river_points[2][2][:2]])),
        river_points[2][2][:2]: ConnectedPoints(upstream=set([river_points[2][1][:2]])),
    }
    
    for coord_array in river_points:
        geometry = ogr.Geometry(ogr.wkbLineString25D)
        for coords in coord_array:
            geometry.AddPoint(coords[0], coords[1], coords[2])
        
        feature = ogr.Feature(layer_defn)
        feature.SetGeometry(geometry)

        centerline_layer.CreateFeature(feature)

    topology = get_layer_topology(centerline_layer)
    
    assert topology == expected_connections
