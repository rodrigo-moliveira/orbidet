class SatelliteSpecs():
    """
    class with physical characteristics of satellite:
        *Drag Area [m^2]
        *mass [kg]

    In my application these values are constant, but, if needed, they may be time dependent
    IMPORTANT NOTE: In the constructor variables should be provided in SI units
    """

    attrs = {
        "m": "mass",
        "a_drag": "drag_area",
        "CD": "CD"
    }

    def __init__(self, name,CD,mass,area,**kwargs):
        """
        Generic class for the description of physical characteristics of satellite
        name of satellite [str]
        Drag_Area [m^2]
        Mass [kg]
        """

        self.name = name
        self.CD = CD
        self.mass = mass
        self.area = area

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
    def area (self):
        return self._area

    @area.setter
    def area(self,value):
        self._area = value / (1000**2)
        """value is stored in [km^2] """
