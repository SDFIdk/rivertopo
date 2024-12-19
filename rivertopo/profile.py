"""
Implementation of river profile geometry.

This module provides a number of classes to represent profile data of various
types. This comprises the Danish categorizations like "regulativprofil",
"opmålt profil" etc. The classes in this module allow abstraction of the
underlying profile type, allowing their geometry to be queried without regard
to profile type.
"""

from osgeo import ogr
import numpy as np

from abc import ABCMeta, abstractmethod

ogr.UseExceptions()

# TODO Add center coords, azimuth?
class ProfilABC(metaclass=ABCMeta):
    """
    Abstract base class representing a river profile.

    This is intended to be subclassed with "regulativprofil", "opmålt profil"
    etc. classes. This base class provides a constructor method to construct a
    profile directly from an OGR Feature with appropriate geometry and
    attributes, as well as an "interp" method to query the profile geometry
    anywhere.
    """

    @abstractmethod
    def __init__(self, feature):
        pass
    
    @abstractmethod
    def interp(self, x):
        pass

class RegulativProfilSimpel(ProfilABC):
    def __init__(self, feature):
        self._anlaeghoejre = feature.GetFieldAsDouble('anlaeghoejre')
        self._anlaegvenstre = feature.GetFieldAsDouble('anlaegvenstre')
        self._bundbredde = feature.GetFieldAsDouble('bundbredde')
        self._bundkote = feature.GetFieldAsDouble('bundkote')

        self._interp = lambda x: np.piecewise(
            x,
            [
                x < -0.5*self._bundbredde,
                np.logical_and(-0.5*self._bundbredde <= x, x <= 0.5*self._bundbredde),
                0.5*self._bundbredde < x,
            ],
            [
                lambda x: self._bundkote + (-0.5*self._bundbredde - x)/self._anlaegvenstre if self._anlaegvenstre != 0.0 else np.nan,
                lambda x: self._bundkote,
                lambda x: self._bundkote + (x - 0.5*self._bundbredde)/self._anlaeghoejre if self._anlaeghoejre != 0.0 else np.nan,
            ]
        )

    def interp(self, x):
        return self._interp(x)

class RegulativProfilSammensat(ProfilABC):
    def __init__(self, feature):
        # TODO check interpretation of these two
        self._afsatbanketbreddehoejre = feature.GetFieldAsDouble('afsatbanketbreddehoejre')
        self._afsatbanketbreddevenstre = feature.GetFieldAsDouble('afsatbanketbreddevenstre')

        self._afsatsanlaeghoejre = feature.GetFieldAsDouble('afsatsanlaeghoejre')
        self._afsatsanlaegvenstre = feature.GetFieldAsDouble('afsatsanlaegvenstre')
        self._afsatskote = feature.GetFieldAsDouble('afsatskote')
        self._anlaeghoejre = feature.GetFieldAsDouble('anlaeghoejre')
        self._anlaegvenstre = feature.GetFieldAsDouble('anlaegvenstre')
        self._bundbredde = feature.GetFieldAsDouble('bundbredde')
        self._bundkote = feature.GetFieldAsDouble('bundkote')

        # intermediate helper coordinates
        bund_left = -0.5*self._bundbredde
        afsatsvenstre_inner = bund_left - self._anlaegvenstre*(self._afsatskote - self._bundkote)
        afsatsvenstre_outer = afsatsvenstre_inner - self._afsatbanketbreddevenstre # TODO check interpretation

        bund_right = 0.5*self._bundbredde
        afsatshoejre_inner = bund_right + self._anlaeghoejre*(self._afsatskote - self._bundkote)
        afsatshoejre_outer = afsatshoejre_inner + self._afsatbanketbreddehoejre # TODO check interpretation

        self._interp = lambda x: np.piecewise(
            x,
            [
                x < afsatsvenstre_outer,
                np.logical_and(afsatsvenstre_outer <= x, x <= afsatsvenstre_inner),
                np.logical_and(afsatsvenstre_inner < x, x < bund_left),
                np.logical_and(bund_left <= x, x <= bund_right),
                np.logical_and(bund_right < x, x < afsatshoejre_inner),
                np.logical_and(afsatshoejre_inner <= x, x <= afsatshoejre_outer),
                afsatshoejre_outer < x,
            ],
            [
                lambda x: self._afsatskote + (afsatsvenstre_outer - x)/self._afsatsanlaegvenstre,
                lambda x: self._afsatskote,
                lambda x: self._bundkote + (bund_left - x)/self._anlaegvenstre,
                lambda x: self._bundkote,
                lambda x: self._bundkote + (x - bund_right)/self._anlaeghoejre,
                lambda x: self._afsatskote,
                lambda x: self._afsatskote + (x - afsatshoejre_outer)/self._afsatsanlaeghoejre,
            ]
        )

    def interp(self, x):
        return self._interp(x)

class OpmaaltProfil(ProfilABC):
    def __init__(self, feature):
        geometry_ref = feature.GetGeometryRef()
        geometry_coords = np.array(geometry_ref.GetPoints())

        # Determine thalweg position, to be considered the profile "center"
        z_min_indices = np.argmin(geometry_coords[:,2])
        z_min_coords = geometry_coords[z_min_indices,:].reshape(-1, 3) # the reshape is necessary when z_min_indices is a scalar
        thalweg_coord = np.mean(z_min_coords, axis=0)

        endpoint_left = geometry_coords[0]
        # endpoint_right = geometry_coords[-1]

        # distances between endpoints
        # delta_x = endpoint_right[0] - endpoint_left[0]
        # delta_y = endpoint_right[1] - endpoint_left[1]

        # azimuth = np.arctan2(-delta_y, delta_x) # TODO generalize to all profile types

        dists_from_left = np.hypot(
            geometry_coords[:,0] - endpoint_left[0],
            geometry_coords[:,1] - endpoint_left[1]
        )
        thalweg_dist_from_left = np.hypot(thalweg_coord[0] - endpoint_left[0], thalweg_coord[1] - endpoint_left[1])

        # along-profile x coordinate
        profile_x = dists_from_left - thalweg_dist_from_left

        self._interp = lambda x: np.interp(x, profile_x, geometry_coords[:,2], left=np.nan, right=np.nan)

    def interp(self, x):
        return self._interp(x)
