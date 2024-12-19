from osgeo import ogr
import numpy as np

ogr.UseExceptions()

class ConnectedPoints:
    """
    Class representing the connections of a point in a river network.

    For a given point, this structure contains a Python set of coordinates of
    the connected upstream points, and another set containing the coordinates
    of the connected downstream points.
    """

    def __init__(self, upstream=frozenset(), downstream=frozenset()):
        # Default arguments are frozensets, since mutable default arguments can
        # cause obscure errors
        self.upstream = set(upstream)
        self.downstream = set(downstream)
    
    def __repr__(self):
        return f'ConnectedPoints(upstream={self.upstream}, downstream={self.downstream})'
    
    def __eq__(self, other):
        return self.upstream == other.upstream and self.downstream == other.downstream

def get_layer_topology(layer):
    """
    For a layer of river network data, determine the connectedness of each vertex.

    Returns a dictionary of ConnectedPoints indexed by coordinates, i.e.:
    {
        (x0, y0): ConnectedPoints(upstream=set([]), downstream=set([(x1,y1)])),
        (x1, y1): ConnectedPoints(upstream=set([(x0,y0)]), downstream=set([(x2,y2)])),
        ...
    }

    :param layer: River network to analyze.
    :type layer: ogr.Layer containing LineString features
    """

    # dictionary of ConnectedPoint, indexed by coordinates
    point_connections = {}

    for feature in layer:
        geometry = feature.GetGeometryRef()

        # TODO flip array if faldretning == "Modsat"?
        geometry_xy = np.array(geometry.GetPoints())[:,0:2]
        geometry_xy_tuples = [tuple(xy) for xy in geometry_xy]

        # initialize
        for xy in geometry_xy_tuples:
            if xy not in point_connections:
                point_connections[xy] = ConnectedPoints()

        # Store upstream and downstream points
        for i in range(1, len(geometry_xy_tuples)):
            point_connections[geometry_xy_tuples[i]].upstream.add(geometry_xy_tuples[i-1])
        
        for i in range(0, len(geometry_xy_tuples)-1):
            point_connections[geometry_xy_tuples[i]].downstream.add(geometry_xy_tuples[i+1])
    
    return point_connections
