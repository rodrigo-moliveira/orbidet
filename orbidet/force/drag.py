import numpy as np
from math import exp
import warnings
from pathlib import Path

from .acceleration import Acceleration
from beyond.beyond.constants import Earth
from orbidet.errors import MissingDbValue,ConfigError


class ExponentialDragDb:
    """Implementation of Exponential Atmospheric Drag database, cf. implemented in Vallado


    Expected table format:
        **
        0: HEADER (h_sat_min | h_sat_max | h0 | rho0 | H)
        1: Data
        2: ...                            ...
        **
    """
    DEFAULT_PATH = Path("orbidet/data/atmosphere.txt")
    PASS = 'pass'
    WARN = 'warn'
    ERROR = 'error'

    def __init__(self,path=DEFAULT_PATH,policy=WARN):

        # Create data dictionary
        self._db = {}
        # policy configuration for this instance
        if policy not in (__class__.PASS, __class__.WARN, __class__.ERROR):
            raise ConfigError("Unknown policy")
        self.policy = policy

        # Data reading
        self._db = self.readfile(path)

    def readfile(self,path):
        db = {}
        with open(path,'r') as f:
            next(f)
            for line in f:
                try:
                    _data = line[0:-1].split(',')
                    h_min = int(_data[0])
                    h_max = int(_data[1])
                    h0 = int(_data[2])
                    rho0 = float(_data[3]) * 10**9 #[kg/km^3]
                    H = float(_data[4])

                    db[(h_min,h_max)] = (h0,rho0,H)

                except:
                    warnings.warn("Warning: Problem encountered while reading atmosphere "
                                  "in line: %s"%line)
        if not db:
            raise Exception("Drag database was not read successfully")
        return db

    def __getitem__(self, h):
        """
        given the satellite altitude h, chooses the correct database line and return it
        """
        for keys,vals in self._db.items():
            if keys[0] <= h < keys[1]:
                return vals
        # in case no match is found returns None
        raise MissingDbValue("No match was found for an altitude of %f"%h)


    def get_rho(self,r):
        """
        Retrieve rho for the drag exponential model
        INPUT:
            radius r of orbit in [km]
        OUTPUT:
            rho [kg/km^3]
        if 0 is returned, then in the acceleration function, drag is excluded
        """
        h = r - Earth.equatorial_radius
        try:
            line = self[h]
            h0,rho0,H = line

        except MissingDbValue:
            if self.policy is __class__.ERROR:
                raise Exception("Missing policy (ERROR) is terminating the process due to failure in DB ")
            else:
                if self.policy is __class__.WARN:
                    warnings.warn("Warning: Missing policy (WARN) is warning about failure in DB read."
                                  "assigning rho to 0")
                return 0
        return rho0 * exp(-(h - h0) / H)




class AtmosphericDrag(Acceleration):

    def __init__(self,sat):
        super().__init__("Atmospheric Drag")
        self.sat = sat

    def acceleration(self, DensityHandler, r, v, rot_vector):
        """
        computes drag acceleration
        DensityHandler - Density Handler (I only implemented an Exponential Model)
        (r,v) satellite position and velocity vectors in ECI
        sat - Satellite instance
        rot_vector - rotational speed vector to convert between ECI and ECEF (omega vector)
        """
        rho = DensityHandler.get_rho(np.linalg.norm(r))

        v_r = v - np.cross(rot_vector,r)
        v_abs = np.linalg.norm(v_r)
        a = -1/2 * rho * self.sat.CD * self.sat.area / self.sat.m * v_abs * v_r

        return a
