"""
Initialization of the orbidet package.
Orbidet is dependent on Beyond, which is initialized here
"""

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
del beyond.beyond.constants.Body.mu #set mu as input constant rather than a computed variable
beyond.beyond.constants.Earth = beyond.beyond.constants.Body(
    name="Earth",
    mu = 3.986004415e5,  # [km^3 / s^2]
    mass=5.97237e24,
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

print("Beyond Package was successfully setup")
