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
