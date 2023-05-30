from hydroadjust.burning import burn_lines

from osgeo import gdal, ogr
import argparse
import logging
import numpy as np

# Entry point for use in setup.py
def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('lines', type=str, help='linestring features with DEM-sampled Z')
    argument_parser.add_argument('input_raster', type=str, help='DEM input raster')
    argument_parser.add_argument('output_raster', type=str, help='DEM output raster with objects burned in')
    argument_parser.add_argument('--log-level', type=str)

    input_arguments = argument_parser.parse_args()

    lines_path = input_arguments.lines
    input_raster_path = input_arguments.input_raster
    output_raster_path = input_arguments.output_raster

    input_raster_dataset = gdal.Open(input_raster_path)

    band = input_raster_dataset.GetRasterBand(1)
    band_array = band.ReadAsArray()

    lines_datasrc = ogr.Open(lines_path)

    # Create an intermediate, in-memory dataset that the lines will be burned
    # into. This is done in order to prevent writing a premature output raster
    # in case something goes wrong during the line rasterization.
    intermediate_driver = gdal.GetDriverByName("MEM")
    intermediate_raster_dataset = intermediate_driver.CreateCopy(
        "temp", # the dataset has to have a name
        input_raster_dataset,
    )

    # Burn the line layers into the temp raster
    for layer in lines_datasrc:
        burn_lines(intermediate_raster_dataset, layer)
        logging.info(f"burned layer {layer.GetName()} into temporary raster")

    int_band = intermediate_raster_dataset.GetRasterBand(1)
    int_band_array = int_band.ReadAsArray()
    
    # only store the minimum values
    int_band_array = np.minimum(int_band_array, band_array)

    # replace values in the intermediate band array with original band if they are larger
    #int_band_array = np.where(int_band_array > band_array, band_array, int_band_array)
   
    # Write the updated array back to the band
    int_band.WriteArray(int_band_array)

    # Save changes to the file
    intermediate_raster_dataset.FlushCache()

    # Line rasterization is now complete, copy the temp raster to output file
    output_driver = gdal.GetDriverByName("GTiff")
    intermediate_raster_dataset = output_driver.CreateCopy(
        output_raster_path,
        intermediate_raster_dataset,
    )
    logging.info("output raster written")

# Allows executing this module with "python -m"
if __name__ == '__main__':
    main()