import numpy as np


from beyond.beyond.frames import frames
from beyond.beyond.constants import Earth
from beyond.beyond.orbits import Orbit
from beyond.beyond.dates import Date


from orbidet.propagators.utils import _framedct
from .stationUtils import create_station,TopocentricFrame



class GroundStation():
    _sensors = ["Range","Range Rate","Azimuth","Elevation"]

    def __init__(self,name,latlonalt,sensors,dict_std,parent_frame):
        """
        args:
            name (str) - name of the ground station
            latlonalt (tuple of floats (lat,lon,alt)) in degrees and meters respectively
            sensors (list of sensors to use). The available measurements are:
                *azimuth [rad]
                *elevation [rad]
                *range[km]
                *range rate[km/s]
                All these values are computed in the topocentric frame (however they may not all be provided and used)

        """
        if not isinstance(latlonalt,tuple):
            msg = ("*latlonalt* argument should be a tuple with the station coordinates\n"
                   "(latitude [deg], longitude[deg],altiude[m])")
            raise Exception(msg)

        if not isinstance(sensors,list):
            msg = ("The *sensors* argument must be a list of strings containing these sensors",
                   "(not all are necessary): \"Range\", \"Range Rate\", \"Azimuth\", \"Elevation\"")
            raise Exception(msg)

        for sensor in sensors:
            if sensor not in __class__._sensors:
                msg = ("Error in the definition of the sensors argument. Unknown sensor: %s" %sensor)
                raise Exception(msg)


        self.station = create_station(name,latlonalt,frames.get_frame(parent_frame))
        self._orbit = None
        self.parent_frame = parent_frame
        self.name = name
        self.sensors = sensors

        self.R_default = self._get_R(sensors,dict_std) #default noise matrix
        self.dict_std = dict_std

        self.coordinates = TopocentricFrame._geodetic_to_cartesian(*self.station.latlonalt)[0:3]




    def __repr__(self):
        lat,lon,alt = self.station.latlonalt
        return ("Ground Station: {}\nGeodetic Coordinates: Lat = {}[deg], Lon = {}[deg], alt = {}[m]".format(
            self.name, np.rad2deg(lat),np.rad2deg(lon),alt*1000))


    def h(self,orbit,date=None,frame="EME2000"):
        """implementation of algorithm 3.4.2 from [Vallado] - TOPOCENTRIC EQUATORIAL MEASUREMENTS
        args:
            *orbit (Orbit) with the current instantaneous orbit to observe

        returns:
            *list with the measurements (depending on the sensors provided). The order of the output
            respects the one provided by *sensors*
        """
        if not isinstance(orbit,Orbit):
            orbit = Orbit(date,orbit,"cartesian",frame,None)


        orbit.frame = self.station
        s = orbit[0:3]
        s_dot = orbit[3:]

        range = np.linalg.norm(s)
        obs = []
        for sensor in self.sensors:
            if sensor == "Range":
                obs.append(range)
            elif sensor == "Azimuth":
                azimuth = np.arctan2(s[0],s[1])
                obs.append(azimuth)
            elif sensor == "Elevation":
                elevation = np.arctan2( s[2] , np.sqrt(s[0]**2+s[1]**2))
                obs.append(elevation)
            elif sensor == "Range Rate":
                range_rate = np.dot(s,s_dot) / range
                obs.append(range_rate)
        return np.array(obs)


#     def grad_h(self,orbit,t=None,frame="EME2000"):
#         """computes the jacobian matrix of the measurement model
#         args:
#             *orbit(Orbit)
#         return:
#             *Jacobian matrix(numpy.ndarray)
#         """
#         if not isinstance(orbit,Orbit):
#             orbit = Orbit(t,orbit,"cartesian",frame,None)
#         self.orbit = orbit
#
#         range = self._dict["range"]
#         s = self._dict["s"]
#         s_dot = self._dict["s_dot"]
#
#         # get matrix ECI to topocentric
#         rot_ECI_horizon = self.ITRF_to_TOP @ self.ECI_ECEF_rot(orbit.date)
#         grad = np.zeros((len(self.sensors),6))
#         i = 0
#         for sensor in self.sensors:
#             if sensor == "Range":
#                 grad[i,0:3] = s/range @ rot_ECI_horizon
#                 i+=1
#             elif sensor == "Azimuth":
#                 grad[i,0:3] = [s[1] / (s[0]**2+s[1]**2) , -s[0]/(s[0]**2+s[1]**2), 0] @ rot_ECI_horizon
#                 i+=1
#             elif sensor == "Elevation":
#                 aux = np.sqrt((s[0])**2 + (s[1])**2) * (range)**2
#                 grad[i,0:3] = [-s[0]*s[2]/aux , -s[1]*s[2]/aux , np.sqrt((s[0])**2 + (s[1])**2) / (range)**2 ] @ rot_ECI_horizon
#                 i+=1
#             elif sensor == "Range Rate":
#                 s_dot = self._dict["s_dot"]
#                 range_rate = self._dict.setdefault("range rate", (np.dot(s,s_dot))/range)
#                 grad[i,0:3] = (((s_dot*range - s*range_rate)/range**2) @ rot_ECI_horizon)
#                 grad[i,3:] = (s/range) @ rot_ECI_horizon
#                 i+=1
#         del(self.orbit)
#         return grad
#
    @classmethod
    def _get_R(cls, sensors, dict_std):
        # Defining the noise matrices
        R = []
        for sensor in sensors:
            if sensor == 'Range':
                R.append(dict_std['range']**2)
            elif sensor == 'Azimuth' or sensor == 'Elevation':
                R.append(dict_std['angles']**2)
            elif sensor == 'Range Rate':
                R.append(dict_std['range rate']**2)
        return np.diag(R)


    def getStationECICoordinates(self,date,ECI_frame):
        orb = Orbit(np.concatenate((self.coordinates,[0,0,0])),date,"cartesian",self.parent_frame,None)
        orb.frame = ECI_frame
        return np.array(orb[0:3])
