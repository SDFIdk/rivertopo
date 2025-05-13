from osgeo import ogr
import numpy as np
from collections import namedtuple

# import pdb

ogr.UseExceptions()

SnapResult = namedtuple('SnapResult', ['feature', 'segment', 'param', 'chainage', 'offset'])

def cross2d(x, y):
    """
    Scalar-value cross product of 2D vectors.

    This used to be possible with np.cross(), but this has been deprecated with
    NumPy 2.0+.
    """
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

def snap_points_to_linestring(points, feature_id, linestring):
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
    segment_lengths = np.hypot(linestring_vectors[:,0], linestring_vectors[:,1])
    cum_segment_lengths = np.hstack([0, np.cumsum(segment_lengths)])

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

        # How far along the matched segment we are
        param = linestring_vector_projparams[closest_segment_index]

        # Chainage (within feature) of best fit
        chainage = cum_segment_lengths[closest_segment_index] + param * segment_lengths[closest_segment_index]

        # Horizontal signed offset (negative left, positive right, as seen in the direction of the line segment)
        offset = point_dists[closest_segment_index] * np.sign(cross2d(vector_rejections[closest_segment_index], linestring_vectors[closest_segment_index]))

        snap_results.append(SnapResult(
            feature=feature_id,
            segment=closest_segment_index,
            param=param,
            chainage=chainage,
            offset=offset,
        ))

    return snap_results

def snap_points_to_linestring_layer(points_layer, linestrings_layer, buffer_dist=0.0):
    snap_results = {} # closest snap result per point
    for linestring_feature in linestrings_layer:
        # get linestring bbox
        linestring_geometry = linestring_feature.GetGeometryRef()
        x_min, x_max, y_min, y_max = linestring_geometry.GetEnvelope()

        points_layer.SetSpatialFilterRect(x_min - buffer_dist, y_min - buffer_dist, x_max + buffer_dist, y_max + buffer_dist)
        points_xy = np.array([point_feature.GetGeometryRef().GetPoint() for point_feature in points_layer])[:, :2]

        linestring_fid = linestring_feature.GetFID() # TODO should we use a particular field here?
        linestring_snap_results = snap_points_to_linestring(points_xy, linestring_fid, linestring_geometry)

        for point_xy, linestring_snap_result in zip(points_xy, linestring_snap_results):
            if point_xy in snap_results:
                prev_snap_result = snap_results[point_xy]
                # Replace SnapResult for this point if we are closer in terms of left/right distance
                if abs(linestring_snap_result.offset) < abs(prev_snap_result.offset):
                    snap_results[point_xy] = linestring_snap_result
            else:
                snap_results[point_xy] = linestring_snap_result

        points_layer.ResetReading()

        return snap_results
