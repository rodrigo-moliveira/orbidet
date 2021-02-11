import numpy as np

from beyond.beyond.frames import frames,center,orient
from beyond.beyond.constants import Earth


"""This classes are from beyond, but I needed to change them in order to implement my own ground station frame
The rotation matrices and angular definitions are different
"""
def create_station(
    name, latlonalt, parent_frame=frames.PEF
):
    """Create a ground station instance

    Args:
        name (str): Name of the station

        latlonalt (tuple of float): coordinates of the station, as follow:

            * Latitude in degrees
            * Longitude in degrees
            * Altitude to sea level in kilometers

        parent_frame (Frame): Planetocentric rotating frame of reference of
            coordinates.
    Return:
        TopocentricFrame
    """

    latlonalt = list(latlonalt)
    latlonalt[:2] = np.radians(latlonalt[:2])
    latlonalt[2] /= 1000 # transform altitude to kilometers
    coordinates = TopocentricFrame._geodetic_to_cartesian(*latlonalt)


    c = center.Center(name, body=parent_frame.center.body)
    c.add_link(
        parent_frame.center,
        parent_frame.orientation,
        coordinates,
    )


    o = TopocentricOrientation(
        name, latlonalt, parent=parent_frame.orientation
    )
    mtd = "{}_to_{}".format(name, parent_frame.orientation.name)
    setattr(orient.Orientation, mtd, o._to_parent)
    o + parent_frame.orientation

    return TopocentricFrame(name, o, c, mask=None)



class TopocentricOrientation(orient.Orientation):
    """Orientation for handling topocentric frames i.e. ground stations"""

    def __init__(self, name, latlonalt, parent=frames.PEF):
        super().__init__(name)
        self.parent = parent
        self.latlonalt = latlonalt

        mtd = "{}_to_{}".format(name, parent.name)
        setattr(self, mtd, self._to_parent)

        self.parent + self


    def _to_parent(self, date):
        lat, lon, _ = self.latlonalt
        # the 'rot3(np.pi)' is here to place the X axis along the north direction
        m = np.array([[-np.sin(lon),np.cos(lon),0],
                [-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)],
                [np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)]]).T
        return m, None


class TopocentricFrame(frames.Frame):
    """Base class for ground station"""

    def __init__(self, name, orientation, center, mask=None):
        self.mask = np.asarray(mask) if mask else None
        super().__init__(name, orientation, center)

    @property
    def latlonalt(self):
        return self.orientation.latlonalt

    def visibility(self, orb, **kwargs):
        """Visibility from a topocentric frame

        see :py:meth:`Propagator.iter() <beyond.propagators.base.Propagator.iter>`
        for description of arguments handling.

        Args:
            orb (Orbit): Orbit to compute visibility from the station with

        Keyword Args:
            start (Date): starting date of the visibility search
            stop (Date or datetime.timedelta) end of the visibility search
            step (datetime.timedelta): step of the computation
            events (bool, Listener or list): If evaluate to True, compute
                AOS, LOS and MAX elevation for each pass on this station.
                If 'events' is a Listener or an iterable of Listeners, they
                will be added to the computation

        Any other keyword arguments are passed to the propagator.

        Yield:
            Orbit: In-visibility point of the orbit. This Orbit is already
                in the frame of the station and in spherical form.
        """

        from ..propagators.listeners import stations_listeners, Listener

        listeners = kwargs.setdefault("listeners", [])
        events = kwargs.pop("events", None)
        event_classes = tuple()

        if events:
            # Handling of the listeners passed in the 'events' kwarg
            # and merging them with the `listeners` kwarg
            if isinstance(events, Listener):
                listeners.append(events)
            elif isinstance(events, (list, tuple)):
                listeners.extend(events)

            sta_list = stations_listeners(self)
            listeners.extend(sta_list)

            # Only the events present in the `event_classes` list will be yielded
            # outside of visibility. This list was created in order to force
            # the yield of AOS and LOS.

            event_classes = tuple(listener.event for listener in sta_list)

        for point in orb.iter(**kwargs):
            point.frame = self
            point.form = "spherical"

            # Not very clean !
            if point.phi < 0 and not isinstance(point.event, event_classes):
                continue

            yield point



    @classmethod
    def _geodetic_to_cartesian(cls, lat, lon, alt):
        """Conversion from latitude, longitude and altitude coordinates to
        cartesian with respect to an ellipsoid (Earth)

        Args:
            lat (float): Latitude in radians
            lon (float): Longitude in radians
            alt (float): Altitude to sea level in kilometers

        Return:
            numpy.array: 3D element (in kilometers)
        """
        #Equation 5.83 of [Montenbruck]
        N = Earth.r / (np.sqrt(1 - Earth.f*(2 - Earth.f)*np.sin(lat)**2 ))
        x = (N + alt) * np.cos(lat) * np.cos(lon)
        y = (N + alt) * np.cos(lat) * np.sin(lon)
        z = ((1 - Earth.f)**2 * N + alt) * np.sin(lat)

        return np.array([x,y,z,0,0,0])


    def get_mask(self, azim):
        """Linear interpolation between two points of the mask"""

        if self.mask is None:
            raise ValueError("No mask defined for the station {}".format(self.name))

        azim %= 2 * np.pi

        if azim in self.mask[0, :]:
            return self.mask[1, np.where(azim == self.mask[0, :])[0][0]]

        for next_i, mask_azim in enumerate(self.mask[0, :]):
            if mask_azim > azim:
                break
        else:
            next_i = 0

        x0, y0 = self.mask[:, next_i - 1]
        x1, y1 = self.mask[:, next_i]

        if next_i - 1 == -1:
            x0 = 0

        return y0 + (y1 - y0) * (azim - x0) / (x1 - x0)
