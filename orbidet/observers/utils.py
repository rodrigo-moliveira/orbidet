"""
this file provides ideal measurements the possible measurements for the problem given the observer
    (GroundStation or SatelliteObserver)
"""
import numpy as np

from beyond.beyond.constants import Earth

_dct = {
    "range": "range",
    "azimuth": "angles",
    "elevation": "angles",
    "elevation":"angles",
    "declination":"angles",
    "range rate": "range rate"
}

def get_obs(orbit,observer,dict_std,apply_noise=False,do_LOS=True):
    sensors = observer.sensors
    date = orbit.date

    r = np.array(orbit[0:3])
    v = np.array(orbit[3:])

    if do_LOS:
        LOS = _line_of_sight(r,observer.getStationECICoordinates(date,orbit.frame),True)
        if not LOS:
            return None

    obs_i = observer.h(orbit)

    if apply_noise:
        for i,sensor in enumerate(sensors):
            obs_i[i] += np.random.normal(0,dict_std[_dct[sensor.lower()]])
    return obs_i


def _line_of_sight(r1,r2,geoid=False):
    """
    this function checks whether there is line of sight between the observatory and the satellite
    applies algorithm SIGHT of Vallado (section 3.9 page 217)
    INPUTS
        -r1 and r2 are the two position vectors (observatory and satellite)
        -geoid is a boolean variable to model the Earth as a geoid
    OUPUT
        True if there is LOS and false otherwise
    """
    if geoid == True:
        r1 = _scaling_factor(r1.copy())
        r2 = _scaling_factor(r2.copy())


    tau = (np.linalg.norm(r1)**2 - np.dot(r1,r2)) / (np.linalg.norm(r1)**2 + np.linalg.norm(r2)**2 - 2*np.dot(r1,r2))

    LOS = False
    if (tau < 0) or (tau > 1):
        LOS = True
    elif (1 - tau) * np.linalg.norm(r1)**2 + np.dot(r1,r2) * tau >= Earth.r**2:
        LOS = True
    return LOS

def _scaling_factor(r):
    """
    this function applies a scaling factor to Earth centered position vectors
    (only in the Z-coordinate) creating an equivalent position vector r that lives
    in a perfect sphere Earth model
    """
    scale = 1 / np.sqrt(1 - Earth.e**2)
    r[2] = r[2] * scale
    return r



def matrix_RSW_to_ECI(pos,vel):
    """
    matrix to rotate RSW to ECI frame
    required orbital elements:
        -ECI postion and velocity vectors (np.array)
    """
    q = pos / np.linalg.norm(pos)
    w = np.cross(pos, vel) / (np.linalg.norm(pos) * np.linalg.norm(vel))
    s = np.cross(w, q)
    R_RSW_to_ECI = np.array([q, s, w]).T
    return R_RSW_to_ECI
