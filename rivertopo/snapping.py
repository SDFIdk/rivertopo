from osgeo import ogr
import numpy as np
from collections import namedtuple

# import pdb

ogr.UseExceptions()

SnapResult = namedtuple('SnapResult', ['feature', 'segment', 'param', 'offset'])

def snap_points(points, feature_id, linestring):
    """
    For an array of points, find their nearest locations on a given linestring
    geometry.

    :param points: Array of point coordinates to snap
    :type points: N-by-2 array (further columns are acceptable)
    :param linestring: LineString geometry to snap to
    :type linestring: ogr.Geometry
    """

    linestring_points = np.array(linestring.GetPoints())[:,:2]
    linestring_startpoints = linestring_points[:-1,:]
    linestring_endpoints = linestring_points[1:,:]

    linestring_vectors = linestring_endpoints - linestring_startpoints

    snap_results = []

    for point in points[:,:2]:
        point_vectors = point[np.newaxis,:] - linestring_startpoints

        # Find vector projections and vector rejections for each line segment, see https://en.wikipedia.org/wiki/Vector_projection
        linestring_vector_projparams = np.sum(point_vectors*linestring_vectors, axis=1) / np.sum(linestring_vectors*linestring_vectors, axis=1)
        vector_projections = linestring_vector_projparams[:,np.newaxis] * linestring_vectors
        vector_rejections = point_vectors - vector_projections

        vector_projections_from_endpoints = linestring_vectors - vector_projections
        longitudinal_dists_from_startpoints = np.hypot(vector_projections[:,0], vector_projections[:,1])
        longitudinal_dists_from_endpoints = np.hypot(vector_projections_from_endpoints[:,0], vector_projections_from_endpoints[:,1])
        
        longitudinal_dists = np.minimum(longitudinal_dists_from_startpoints, longitudinal_dists_from_endpoints)
        # Set longitudinal distances to 0 where the vector projection falls within the segment
        longitudinal_dists[np.logical_and(linestring_vector_projparams >= 0.0, linestring_vector_projparams <= 1.0)] = 0.0

        transversal_dists = np.hypot(vector_rejections[:,0], vector_rejections[:,1])

        # This point's distances to the respective line segments
        point_dists = np.hypot(longitudinal_dists, transversal_dists)
        
        # The index of the best-fit line segment
        closest_segment_index = np.argmin(point_dists)

        # Horizontal signed offset (negative left, positive right, as seen in the direction of the line segment)
        offset = point_dists[closest_segment_index] * np.sign(np.cross(vector_rejections[closest_segment_index], linestring_vectors[closest_segment_index]))

        snap_results.append(SnapResult(
            feature=feature_id,
            segment=closest_segment_index,
            param=linestring_vector_projparams[closest_segment_index],
            offset=offset,
        ))

    return snap_results
