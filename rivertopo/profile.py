from osgeo import ogr
import scipy
import numpy as np

from abc import ABCMeta, abstractmethod, abstractproperty

ogr.UseExceptions()

class ProfilABC(metaclass=ABCMeta):
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
    
    def interp(self, x):
        return np.piecewise(
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

    def interp(self, x):
        bund_left = -0.5*self._bundbredde
        afsatsvenstre_inner = bund_left - self._anlaegvenstre*(self._afsatskote - self._bundkote)
        afsatsvenstre_outer = afsatsvenstre_inner - self._afsatbanketbreddevenstre # TODO check interpretation

        bund_right = 0.5*self._bundbredde
        afsatshoejre_inner = bund_right + self._anlaeghoejre*(self._afsatskote - self._bundkote)
        afsatshoejre_outer = afsatshoejre_inner + self._afsatbanketbreddehoejre # TODO check interpretation

        return np.piecewise(
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
