"""
Initialization of the orbidet package.
Orbidet is dependent on Beyond, which is initialized here
"""
import numpy as np
from numpy import cos, arccos, sin, arcsin, arctan2, sqrt, arctanh, sinh, cosh, tan

# User configuration of Beyond Package
from beyond.beyond.config import config
from pathlib import Path


# Earth Orientation Parameters
path = Path(__file__).parent.absolute()
config.update({
        "eop": {
            "missing_policy": "error",
            'folder':  path / "data/pole/",
            'type': "all",
        }
    })



# change Earth constant
import beyond.beyond.constants
beyond.beyond.constants.G = 6.6743015e-11* 10**-9
# del beyond.beyond.constants.Body.mu #set mu as input constant rather than a computed variable
beyond.beyond.constants.Earth = beyond.beyond.constants.Body(
    name="Earth",
    mass=5.972167147378643e+24, # so that mu becomes mu = 3.986004415e5,  # [km^3 / s^2]
    equatorial_radius=6378.137, # [km]
    polar_radius = 6356.7523    ,#[km]
    flattening= 1 / 298.257222101,
    J2=0.108262693e-2,
    J3 = -0.253230782e-5,
    J4 = -0.162042999e-5,
    J5 = -0.227071104e-6,
    J6 = 0.540843616e-6,
    rot_vector = [0,0,7.292115e-5], #angular velocity vector
)



from beyond.beyond.frames.iau1980 import _tab
from math import sin, cos, radians

# I define a new _nutation function which is not cached (memoized), since in long term Orbit Determination
# simulations it would eventually consume all the available memory
# The alternative to this is to simply comment the @memoize decorator on the iau1980.py file
def _new_nutation(date, eop_correction=True, terms=106):
    """Model 1980 of nutation as described in Vallado p. 224

    Args:
        date (beyond.utils.date.Date)
        eop_correction (bool): set to ``True`` to include model correction
            from 'finals' files.
        terms (int)
    Return:
        tuple : 3-elements, all floats in degrees
            1. ̄ε
            2. Δψ
            3. Δε

    Warning:
        The good version of the nutation model can be found in the **errata**
        of the 4th edition of *Fundamentals of Astrodynamics and Applications*
        by Vallado.
    """

    ttt = date.change_scale("TT").julian_century

    r = 360.0

    # in arcsecond
    epsilon_bar = 84381.448 - 46.8150 * ttt - 5.9e-4 * ttt ** 2 + 1.813e-3 * ttt ** 3

    # Conversion to degrees
    epsilon_bar /= 3600.0

    # mean anomaly of the moon
    m_m = (
        134.96298139
        + (1325 * r + 198.8673981) * ttt
        + 0.0086972 * ttt ** 2
        + 1.78e-5 * ttt ** 3
    )

    # mean anomaly of the sun
    m_s = (
        357.52772333
        + (99 * r + 359.0503400) * ttt
        - 0.0001603 * ttt ** 2
        - 3.3e-6 * ttt ** 3
    )

    # L - Omega
    u_m_m = (
        93.27191028
        + (1342 * r + 82.0175381) * ttt
        - 0.0036825 * ttt ** 2
        + 3.1e-6 * ttt ** 3
    )

    # Mean elongation of the moon from the sun
    d_s = (
        297.85036306
        + (1236 * r + 307.11148) * ttt
        - 0.0019142 * ttt ** 2
        + 5.3e-6 * ttt ** 3
    )

    # Mean longitude of the ascending node of the moon
    om_m = (
        125.04452222
        - (5 * r + 134.1362608) * ttt
        + 0.0020708 * ttt ** 2
        + 2.2e-6 * ttt ** 3
    )

    delta_psi = 0.0
    delta_eps = 0.0
    for integers, reals in _tab(terms):
        a1, a2, a3, a4, a5 = integers
        A, B, C, D = reals

        a_p = a1 * m_m + a2 * m_s + a3 * u_m_m + a4 * d_s + a5 * om_m

        delta_psi += (A + B * ttt) * sin(radians(a_p)) / 36000000.0
        delta_eps += (C + D * ttt) * cos(radians(a_p)) / 36000000.0

    if eop_correction:
        delta_eps += date.eop.deps / 3600000.0
        delta_psi += date.eop.dpsi / 3600000.0

    return epsilon_bar, delta_psi, delta_eps
import beyond.beyond.frames.iau1980
beyond.beyond.frames.iau1980._nutation = _new_nutation











#Equinoctial form
import beyond.beyond.orbits.forms

EQUI_M = beyond.beyond.orbits.forms.Form("equinoctial_mean", ["a","h","k","p","q","lmb"])
"""Equinoctial form
    * a : semimajor axis [km]
    * h : []
    * k : []
    * p : []
    * q : []
    * lambda : longitude [rad]
"""
beyond.beyond.orbits.forms._cache["equinoctial_mean"] = EQUI_M
beyond.beyond.orbits.forms.KEPL_M + EQUI_M



def _keplerian_mean_to_equinoctial_mean(cls,coord,center):

    a, e, i, RAAN, w, M = coord

    h = e*sin(w + RAAN)
    k = e*cos(w + RAAN)
    p = tan(i/2) * sin(RAAN)
    q = tan(i/2) * cos(RAAN)
    lmb = (M + w + RAAN) % (np.pi * 2)

    return np.array([a,h,k,p,q,lmb], dtype=float)


def _equinoctial_mean_to_keplerian_mean(cls,coord,center):
    a,h,k,p,q,lmb = coord

    #auxiliary angle eta
    sin_eta = h / (sqrt(h**2 + k**2))
    cos_eta = k / (sqrt(h**2 + k**2))
    eta = arctan2(sin_eta,cos_eta)
    if np.isnan(eta):
        eta = 0

    e = sqrt(h**2 + k**2)
    i = (2*np.arctan(sqrt(p**2 + q**2))) % (2*np.pi)
    sin_RAAN = p / (sqrt(p**2 + q**2))
    cos_RAAN = q / (sqrt(p**2 + q**2))
    RAAN = arctan2(sin_RAAN,cos_RAAN)
    w = (eta - RAAN) % (np.pi * 2)
    M = (lmb - eta) % (np.pi * 2)

    return np.array([a, e, i, RAAN, w, M], dtype=float)


setattr(beyond.beyond.orbits.forms.Form, "_keplerian_mean_to_equinoctial_mean", _keplerian_mean_to_equinoctial_mean)
setattr(beyond.beyond.orbits.forms.Form, "_equinoctial_mean_to_keplerian_mean", _equinoctial_mean_to_keplerian_mean)




print("Beyond Package was successfully setup")
