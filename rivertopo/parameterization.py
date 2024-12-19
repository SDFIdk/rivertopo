from osgeo import ogr
import matplotlib.pyplot as plt
import numpy as np

ogr.UseExceptions()

FEATURE_ID_TYPE = np.int64
FEATURE_ID_NODATA = np.iinfo(FEATURE_ID_TYPE).min # Avoid colliding with valid index values
POINT_ID_TYPE = np.int64
POINT_ID_NODATA = np.iinfo(POINT_ID_TYPE).min # Avoid colliding with valid index values
SEGMENT_ID_TYPE = np.int64
SEGMENT_ID_NODATA = np.iinfo(SEGMENT_ID_TYPE).min # Avoid colliding with valid index values

def parameterize_grid(grid_xy, linestring):
    linestring_points = np.array(linestring.GetPoints())[:,:2]
    linestring_startpoints = linestring_points[:-1,:]
    linestring_endpoints = linestring_points[1:,:]

    linestring_vectors = linestring_endpoints - linestring_startpoints

    linestring_chainages = np.hstack([0, np.cumsum(np.hypot(linestring_vectors[:, 0], linestring_vectors[:, 1]))])

    grid_x, grid_y = grid_xy

    feature_id = 0

    dist_grid = np.full_like(grid_x, np.inf)
    feature_id_grid = np.full_like(grid_x, FEATURE_ID_NODATA, dtype=FEATURE_ID_TYPE)
    segment_id_grid = np.full_like(grid_x, SEGMENT_ID_NODATA, dtype=SEGMENT_ID_TYPE)
    chainage_grid = np.full_like(grid_x, np.nan)
    
    # Iterate over linestring segments
    for i in range(len(linestring_points)-1):
        startpoint = linestring_points[i]
        endpoint = linestring_points[i+1]
        segment_vector = endpoint - startpoint

        # Vectors from line segment start point to each pixel center of the raster
        raster_vectors = np.column_stack([grid_x.flatten(), grid_y.flatten()]) - startpoint[np.newaxis, :]
        raster_endpoint_vectors = raster_vectors - (endpoint - startpoint)[np.newaxis, :]
        
        # Vector projections onto this line segment ("params" is the relative distance along the segment, 0 at the start point and 1 at the endpoint)
        raster_projection_params = np.sum(raster_vectors * segment_vector[np.newaxis,:], axis=1) / np.sum(segment_vector * segment_vector)
        raster_projections = raster_projection_params[:,np.newaxis] * segment_vector[np.newaxis,:]
        raster_rejections = raster_vectors - raster_projections

        # determine if we are left (-1) or right (+1) of line segment, using cross product
        left_or_right = np.sign(raster_rejections[:,0]*segment_vector[np.newaxis,1] - raster_rejections[:,1]*segment_vector[np.newaxis,0])

        # As it is not acceptable to use 0 as a distance multiplier here, use +1 (right) as a tiebreaker
        left_or_right[left_or_right == 0] = 1

        # TODO grab sign from raster rejection
        startpoint_dists = np.hypot(raster_vectors[:,0], raster_vectors[:,1])
        endpoint_dists = np.hypot(raster_endpoint_vectors[:,0], raster_endpoint_vectors[:,1])
        perpendicular_dists = np.hypot(raster_rejections[:,0], raster_rejections[:,1])

        # Raster distances to this line segment
        segment_dists = np.full((len(raster_vectors),), np.inf)

        startpoint_is_nearest = startpoint_dists < segment_dists
        segment_dists[startpoint_is_nearest] = startpoint_dists[startpoint_is_nearest]

        endpoint_is_nearest = endpoint_dists < segment_dists
        segment_dists[endpoint_is_nearest] = endpoint_dists[endpoint_is_nearest]

        is_within_segment = np.logical_and(raster_projection_params >= 0.0, raster_projection_params <= 1.0)
        segment_dists[is_within_segment] = np.minimum(segment_dists[is_within_segment], perpendicular_dists[is_within_segment])

        segment_dists *= left_or_right

        segment_chainages = np.full((len(raster_vectors),), np.nan)
        segment_chainages[startpoint_is_nearest] = linestring_chainages[i]
        segment_chainages[endpoint_is_nearest] = linestring_chainages[i+1]
        segment_chainages[is_within_segment] = (1.0-raster_projection_params[is_within_segment])*linestring_chainages[i] + raster_projection_params[is_within_segment]*linestring_chainages[i+1]

        segment_dists_grid = segment_dists.reshape(grid_x.shape)
        segment_chainage_grid = segment_chainages.reshape(grid_x.shape)

        delta_x = grid_x[0,1] - grid_x[0,0]
        delta_y = grid_y[1,0] - grid_y[0,0]
        extent = (grid_x[0,0] - 0.5*delta_x, grid_x[0, -1] + 0.5*delta_x, grid_y[0,0] - 0.5*delta_y, grid_y[-1,0] + 0.5*delta_y)

        segment_is_closest = np.abs(segment_dists_grid) < np.abs(dist_grid)
        dist_grid[segment_is_closest] = segment_dists_grid[segment_is_closest]
        chainage_grid[segment_is_closest] = segment_chainage_grid[segment_is_closest]
        segment_id_grid[segment_is_closest] = i
        feature_id_grid[segment_is_closest] = feature_id

        dist_maxabs = np.max(np.abs(dist_grid))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    im1 = ax1.imshow(dist_grid, extent=extent, origin='lower', cmap='RdBu_r', vmin=-dist_maxabs, vmax=dist_maxabs)
    ax1.plot(linestring_points[:,0], linestring_points[:,1], '.-k')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    fig.colorbar(im1, ax=ax1, shrink=0.5)
    ax1.set_title('Left/right distance to centerline')
    im2 = ax2.imshow(chainage_grid, extent=extent, origin='lower')
    ax2.plot(linestring_points[:,0], linestring_points[:,1], '.-k')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Chainage')
    fig.colorbar(im2, ax=ax2, shrink=0.5)
    im3 = ax3.imshow(segment_id_grid, extent=extent, origin='lower')
    ax3.plot(linestring_points[:,0], linestring_points[:,1], '.-k')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Segment ID')
    fig.colorbar(im3, ax=ax3, shrink=0.5)
    im4 = ax4.imshow(feature_id_grid, extent=extent, origin='lower')
    ax4.plot(linestring_points[:,0], linestring_points[:,1], '.-k')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Feature ID')
    fig.colorbar(im4, ax=ax4, shrink=0.5)
    plt.show()

def _debug_example():
    grid_xy = np.meshgrid(np.linspace(0., 10., 250), np.linspace(0., 8., 200))

    linestring = ogr.Geometry(ogr.wkbLineString)
    linestring.AddPoint(2.1, 2.0)
    linestring.AddPoint(5.5, 4.0)
    linestring.AddPoint(7.0, 6.0)
    linestring.AddPoint(8.5, 6.2)

    parameterize_grid(grid_xy, linestring)
