from osgeo import ogr
import numpy as np

ogr.UseExceptions()

class ConnectedPoints:
    def __init__(self, upstream=frozenset(), downstream=frozenset()):
        self.upstream = set(upstream)
        self.downstream = set(downstream)
    
    def __repr__(self):
        return f'ConnectedPoints(upstream={self.upstream}, downstream={self.downstream})'
    
    def __eq__(self, other):
        return self.upstream == other.upstream and self.downstream == other.downstream

def get_layer_topology(layer):
    # dictionary of ConnectedPoint, indexed by coordinates
    point_connections = {}

    for feature in layer:
        geometry = feature.GetGeometryRef()
        # geometry = feature.GetGeometryRef().GetGeometryRef(0)

        # TODO flip array if faldretning == "Modsat"?
        geometry_xy = np.array(geometry.GetPoints())[:,0:2]
        geometry_xy_tuples = [tuple(xy) for xy in geometry_xy]

        # initialize
        for xy in geometry_xy_tuples:
            if not xy in point_connections:
                point_connections[xy] = ConnectedPoints()

        # Store upstream and downstream points
        for i in range(1, len(geometry_xy_tuples)):
            point_connections[geometry_xy_tuples[i]].upstream.add(geometry_xy_tuples[i-1])
        
        for i in range(0, len(geometry_xy_tuples)-1):
            point_connections[geometry_xy_tuples[i]].downstream.add(geometry_xy_tuples[i+1])
    
    return point_connections
