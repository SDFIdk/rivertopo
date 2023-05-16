from rivertopo.profile import ProfilABC, RegulativProfilSimpel, RegulativProfilSammensat, OpmaaltProfil

from osgeo import ogr
import numpy as np

ogr.UseExceptions()

def assert_profile_interp_equal(x: np.array, actual_profile: ProfilABC, expected_coords: np.array):
    actual_coords = actual_profile.interp(x)
    assert np.allclose(actual_coords, expected_coords, equal_nan=True)

def test_regulativprofilsimpel(regulativprofilsimpel_layer):
    geometry = ogr.Geometry(ogr.wkbPoint)
    geometry.AddPoint(500000.0, 6200000.0)
    
    feature = ogr.Feature(regulativprofilsimpel_layer.GetLayerDefn())
    feature.SetGeometry(geometry)
    feature.SetField('anlaeghoejre', 1.0)
    feature.SetField('anlaegvenstre', 2.0)
    feature.SetField('bundbredde', 2.0)
    feature.SetField('bundkote', 42.0)
    
    regulativprofilsimpel_layer.CreateFeature(feature)

    profile = RegulativProfilSimpel(feature)

    x = np.array([-4.0, -3.0, -1.0, 0.0, 1.0, 1.5, 2.0, 2.5])
    expected_y = np.array([43.5, 43.0, 42.0, 42.0, 42.0, 42.5, 43.0, 43.5])

    assert_profile_interp_equal(x, profile, expected_y)

def test_regulativprofilsammensat(regulativprofilsammensat_layer):
    geometry = ogr.Geometry(ogr.wkbPoint)
    geometry.AddPoint(500000.0, 6200000.0)

    feature = ogr.Feature(regulativprofilsammensat_layer.GetLayerDefn())
    feature.SetGeometry(geometry)
    feature.SetField('afsatbanketbreddehoejre', 3.0)
    feature.SetField('afsatbanketbreddevenstre', 2.0)
    feature.SetField('afsatsanlaeghoejre', 1.0)
    feature.SetField('afsatsanlaegvenstre', 2.0)
    feature.SetField('afsatskote', 43.0)
    feature.SetField('anlaeghoejre', 2.0)
    feature.SetField('anlaegvenstre', 1.0)
    feature.SetField('bundbredde', 3.0)
    feature.SetField('bundkote', 42.0)

    regulativprofilsammensat_layer.CreateFeature(feature)

    profile = RegulativProfilSammensat(feature)

    x = np.array([-6.5, -4.5, -3.5, -2.5, -2.0, -1.5, 0.0, 1.5, 2.5, 3.5, 5.0, 6.5, 7.5])
    expected_y = np.array([44.0, 43.0, 43.0, 43.0, 42.5, 42.0, 42.0, 42.0, 42.5, 43.0, 43.0, 43.0, 44.0])

    assert_profile_interp_equal(x, profile, expected_y)

def test_opmaaltprofil(opmaaltprofil_layer):
    geometry = ogr.Geometry(ogr.wkbLineString25D)
    geometry.AddPoint(500000.0 + 0.4*(-2), 6200000.0 + 0.3*(-2), 43.5)
    geometry.AddPoint(500000.0 + 0.4*(-1), 6200000.0 + 0.3*(-1), 42.5)
    geometry.AddPoint(500000.0, 6200000.0, 42.0) # thalweg (lowest z)
    geometry.AddPoint(500000.0 + 0.4*1, 6200000.0 + 0.3*1, 43.0)
    geometry.AddPoint(500000.0 + 0.4*3, 6200000.0 + 0.3*3, 43.5)
    geometry.AddPoint(500000.0 + 0.4*4, 6200000.0 + 0.3*4, 44.0)

    feature = ogr.Feature(opmaaltprofil_layer.GetLayerDefn())
    feature.SetGeometry(geometry)

    opmaaltprofil_layer.CreateFeature(feature)

    profile = OpmaaltProfil(feature)

    # avoid testing at endpoints, as rounding may cause them to be in NaN area
    x = 0.5*np.array([-2.5, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 3.5, 4.5])
    expected_y = np.array([np.nan, 43.0, 42.5, 42.25, 42.0, 42.5, 43.0, 43.25, 43.5, 43.75, np.nan])

    assert_profile_interp_equal(x, profile, expected_y)
