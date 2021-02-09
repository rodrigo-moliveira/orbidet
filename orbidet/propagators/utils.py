"""Utility functions for propagation/integration
    -differential state functions
    -frame rotation handlers
"""
import numpy as np

from beyond.beyond.dates import Date
from beyond.beyond.frames.orient import TEME,PEF,TOD,MOD,EME2000,G50,ITRF,TIRF,CIRF,GCRF

_framedct = {"TEME":TEME,"PEF":PEF,"TOD":TOD,"MOD":MOD,
            "EME2000":EME2000, "G50":G50,"ITRF":ITRF, "TIRF":TIRF,
            "CIRF":CIRF,"GCRF":GCRF}



def cartesian_osculating_state_fct(t,X,force):
    """
    returns differential equation dx/dt = f(x),
        where x = (r,v) -> osculating state vector
    """
    # unpacking state
    date = Date(t/86400)

    # get rotation matrix from ECI (integration frame) to ECEF (gravity frame)
    T = _framedct[force.integrationFrame].convert_to(date,force.gravityFrame)

    X_trans = T @ X
    # R = np.array(X_trans[0:3])
    # V = np.array(X_trans[3:])

    # apply force model and compute acceleration in ECEF
    a = np.zeros(3)
    for accel in force.getForceList():
        a += accel.acceleration(X_trans)

    # rotate acceleration to ECI
    T = T[0:3,0:3]
    a = T.T @ a
    return np.concatenate((X[3:],a))
