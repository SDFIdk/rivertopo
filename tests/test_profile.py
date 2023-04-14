from rivertopo.profile import ProfilABC, RegulativProfilSimpel

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
