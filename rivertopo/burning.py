from osgeo import gdal

def burn_lines(raster, lines):
    """
    Burn elevation of vector line segments into raster, modifying the raster
    dataset in-place.

    :param raster: DEM raster to burn lines into
    :type raster: GDAL Dataset object
    :param lines: line segments whose elevation should be burned in
    :type lines: OGR Layer object
    """

    # The "burn value" must be set to 0, resulting in 0 + the z value being
    # burned in. The default is 255 + z (yes, really).
    # See https://lists.osgeo.org/pipermail/gdal-dev/2015-August/042360.html
    gdal.RasterizeLayer(
        raster,
        [1], # band 1
        lines,
        burn_values=[0],
        options=[
            'BURN_VALUE_FROM=Z',
            'ALL_TOUCHED=TRUE', # ensure connectedness of resulting pixels
        ],
    )