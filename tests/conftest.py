from osgeo import ogr, osr
import pytest

ogr.UseExceptions()
osr.UseExceptions()

@pytest.fixture()
def srs():
    dk_srs = osr.SpatialReference()
    dk_srs.ImportFromEPSG(25832)
    return dk_srs

@pytest.fixture()
def driver():
    return ogr.GetDriverByName('Memory')

@pytest.fixture()
def centerline_datasrc(driver):
    return driver.CreateDataSource('river_centerlines')

@pytest.fixture()
def profile_datasrc(driver):
    return driver.CreateDataSource('vandloebsdata')

@pytest.fixture()
def centerline_layer(centerline_datasrc, srs):
    layer = centerline_datasrc.CreateLayer('vandloebsmidte', srs=srs, geom_type=ogr.wkbLineString25D)
    return layer

@pytest.fixture()
def regulativprofilsimpel_layer(profile_datasrc, srs):
    layer = profile_datasrc.CreateLayer('regulativprofilsimpel_nohist', srs=srs, geom_type=ogr.wkbPoint)
    layer.CreateField(ogr.FieldDefn('anlaeghoejre', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('anlaegvenstre', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('bundbredde', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('bundkote', ogr.OFTReal))
    return layer

@pytest.fixture()
def regulativprofilsammensat_layer(profile_datasrc, srs):
    layer = profile_datasrc.CreateLayer('regulativprofilsammens_nohist', srs=srs, geom_type=ogr.wkbPoint)
    layer.CreateField(ogr.FieldDefn('afsatbanketbreddehoejre', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('afsatbanketbreddevenstre', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('afsatsanlaeghoejre', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('afsatsanlaegvenstre', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('afsatskote', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('anlaeghoejre', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('anlaegvenstre', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('bundbredde', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('bundkote', ogr.OFTReal))
    return layer

@pytest.fixture()
def opmaaltprofil_layer(profile_datasrc, srs):
    layer = profile_datasrc.CreateLayer('opmaaltprofil_nohist', srs=srs, geom_type=ogr.wkbLineString25D)
    return layer
