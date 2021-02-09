"""
class with physical characteristics of satellite (used mostly in the force model):
    *Drag Area [m^2]
    *SRP Area [m^2]
    *mass [kg]

In my application these are all constant (but, if needed they may be time dependent)


IMPORTANT NOTE: In the constructor variables should be provided in SI units
"""

class SatelliteSpecs():
    attrs = {
        "m": "mass",
        "a_drag": "drag_area",
        "a_srp": "area_SRP",
        "CD": "CD"
    }

    def __init__(self, name,**kwargs):
        """
        Generic class for the description of physical characteristics of propagated satellite
        name of satellite [str]
        Drag_Area [m^2]
        Mass [kg]
        """

        self.name = name

        for k, v in kwargs.items():
            setattr(self, self.__class__.attrs[k], v)
    def __repr__(self):
        return "<Satellite '%s'>" % self.name

    def __getattr__(self, name):
        try:
            return getattr(self, __class__.attrs[name])
        except KeyError:
            raise AttributeError(name)

    @property
    def drag_area (self):
        return self._drag_area
    
    @drag_area.setter
    def drag_area(self,value):
        self._drag_area = value / (1000**2)
        """value is stored in [km^2] """
