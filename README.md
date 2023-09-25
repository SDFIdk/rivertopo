# rivertopo
Tools to help combine river profile data and DEMs

## Installation
A Python 3 environment is required. A suitable Conda environment (here called
"rivertopo") can be created with:

```
conda env create -n rivertopo -f environment.yml
```

For now, the tools support editable installation using `pip`. To install the
tools this way, use the following command in the root directory:

```
pip install -e .
```

Test that everything works by calling `pytest` from the root directory:

```
pytest
```
## Usage

The scripts are still under development, and is only tested for for a specific usecase (see test data under tests) 

### Preparing line objects for burning using cross_lines_z_2.py and cross sectional data
This module is made for creating ready to burn lines from cross sectional data gathered from Dataforsyningen.dk (skikkelsesdata for vandløb). 

```
cross_lines_z_2.py input_points_simpel input_points_sammensat input_points_opmaalt input_polyline output_lines 
```
| Parameter | Description |
| --------- | ----------- |
| `input_points_simpel` | Path to gpkg file of regulatory simple profiles |
| `input_points_sammensat` | Path to gpkg file of regulatory sammensat profiles |
| `input_points_opmaalt` | Path to gpkg file of opmaalt profiles |
| `input_polyline` | Path to geoDK vandloebsmidte polyline |
| `output_lines` | Path to file to write output elevation-sampled 3D line objects to. Will be written in gpkg format |

### Burning the prepared vector objects into raster tile
This module is made for burning the cross sectional data created in the 2d lines the above script (cross_lines_z_2.py).
```
burn_line_z.py input_lines input_raster output_raster 
```
| Parameter | Description |
| --------- | ----------- |
| `input_lines` |  Path or connection string to OGR-readable datasource containing the cross_lines_z 2D line objects |
| `input_raster` | Path to GDAL-readable raster dataset for input tile |
| `output_raster` | Path to write output raster tile to. Will be written in GeoTIFF format |

### Profile extraction along polyline
This module is made to extract profiles along a polyline e.g. a polyline over a streamlines mid. The output is CSV files intended to use for plotting.
```
profile_extraction.py input_rasters input_line output_lines 
```
| Parameter | Description |
| --------- | ----------- |
| `input_rasters` |   Path to GDAL-readable raster dataset(s) for input tile |
| `input_line` | Path to geoDK vandloebsmidte polyline |
| `output_lines` | Path to file to write output elevation-sampled 3D line objects to. Will be written in gpkg format |

### Profile plotting using plotly module
This module is made for 5 specific cross section files created in profile_extraction.py for 4 locations. 
```
profile_plotting_v2.py ex_profiles1 ex_profiles2 ex_profiles3 ex_profiles4_line1 ex_profiles5_line2
```
| Parameter | Description |
| --------- | ----------- |
| `ex_profiles1` | Path to first (karup.csv) CSV file containing cross sectional data from profile_extraction.py |
| `ex_profiles2` | Path to second (fiskbæk.csv) CSV file containing cross sectional data from profile_extraction.py |
| `ex_profiles3` | Path to third (skive.csv) CSV file containing cross sectional data from profile_extraction.py |
| `ex_profiles4_line1` | Path to (indbrændt.csv) CSV file containing cross sectional data from profile_extraction.py |
| `ex_profiles5_line2` | Path to (indbrændings_eksempel.csv) CSV file containing cross sectional data from profile_extraction.py |

## Example workflow (creating and burning cross sectional data points)

As an example, the steps below illustrate preparing the relevant intermediate data and burning it into a raster tile.

| Example filename | Description |
| ---------------- | ----------- |
| test_sammensat.gpkg | Input cross sectional data of type sammensat |
| test_simpel.gpkg | Input cross sectional data of type simple |
| test_opmaalt.gpkg | Input cross sectional data of type opmaalt |
| cropped_vandloebsmidte.gpkg | Input vandloebsmidte polyline |
| LINES_WITH_Z.gpkg | Intermediate datasource of prepared 3D line objects |
| ORIGINAL_DTM/1km_NNNN_EEE.tif | Input raster tile for which corresponding output will be produced. |
| ADJUSTED_DTM/1km_NNNN_EEE.tif | Output raster tile, created from the input raster tile with 3D lines burned in |

Prepare lines with cross sectional data points:

```
cross_lines_z_2.py test_simpel.gpkg test_sammensat.gpkg test_opmaalt.gpkg cropped_vandloebsmidte.gpkg LINES_WITH_Z.gpkg 
```

Notice that the script as of right now will only handle cases where the points and segments are in order. It might be necesary to 'clean up' the lines manually in QGIS afterwards, for best burn results.

Create the adjusted DEM tile from the 3d lines and the original DEM tile:

```
burn_line_z.py LINES_WITH_Z.gpkg ORIGINAL_DTM/1km_NNNN_EEE.tif ADJUSTED_DTM/1km_NNNN_EEE.tif 
```

## Example workflow (extracting and plotting profiles from DEM tile)

Prepare and extract profiles to a CSV on polyline:

```
profile_extraction.py ORIGINAL_DTM/1km_NNNN_EEE.tif cropped_vandloebsmidte.gpkg 
```

Now the CSV file can be used in the plotting module. It is currently set up to plot the 5 example case files:

```
profile_plotting_v2.py karup.csv skive.csv fiskbæk.csv indbrændt.csv indbrændt_eksempel.csv
```



