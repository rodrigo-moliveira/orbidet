"""Utility functions for propagation/integration
    -differential state functions
    -frame rotation handlers
"""
import numpy as np
from scipy.integrate import fixed_quad

from beyond.beyond.dates import Date
from beyond.beyond.frames.orient import TEME,PEF,TOD,MOD,EME2000,G50,ITRF,TIRF,CIRF,GCRF

_framedct = {"TEME":TEME,"PEF":PEF,"TOD":TOD,"MOD":MOD,
            "EME2000":EME2000, "G50":G50,"ITRF":ITRF, "TIRF":TIRF,
            "CIRF":CIRF,"GCRF":GCRF}


################################################################################
#Cowell propagator functions
################################################################################

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






################################################################################
#Semianalytical propagator functions
################################################################################

# State function in equinoctial elements (GAUSS VOP)
def equinoctial_mean_state_fct(t,X,*args):
    """
    returns differential equation dx/dt = f(x),
        where x = equinoctial mean orbit
    """
    force = args[0]
    n = args[1]
    print("corrigir state functions")
    exit()
    # reconstruct the orbit object
    mean_orb = Orbit(Date(t/86400), X,"equinoctial",force.integrationFrame,None)
    lmb = X[5]

    # quadratures (mean dynamics)
    f_mean,_ =  fixed_quad(wrapp_VOP_quad, lmb, lmb+2*np.pi,args=(mean_orb,force),n=n)
    f_mean = f_mean / 2 / np.pi

    _n = np.sqrt(mu / (mean_orb.a)**3)
    return np.array ( [ f_mean[0],
                       f_mean[1],
                       f_mean[2],
                       f_mean[3],
                       f_mean[4],
                       _n + f_mean[5]])


def wrapp_VOP_quad(x,orbit,force,_types):
    """
    tesserals do not come into play (non-resonant case only)
    """
    a,h,k,p,q = orbit[0:5]
    lmb = x % (2*np.pi)

    orb_arr = []
    for lmb_k in lmb:
        orb = orbit.copy()
        orb[5] = lmb_k
        orb_arr.append(orb)

    Fs = np.array([np.array(VOP_partials(orb,force)) for orb in orb_arr])
    return Fs.T




def perturbing_osculating_state_fct(R,V,force,date,T,types):
    # rotate cartesian vectors (rotation is only needed for types = "double" -> tesserals need to be evaluated in ECEF
                #, for types = "single" (zonals, drag) we evaluate the accelerations directly in ECI
    R = T @ R
    V = T @ V
    r = np.linalg.norm(R)
    a_pert = np.zeros(3)

    if "single" in types:
        for accel in force.getForceList():
            if ("gravity" in accel.type):
                a_pert += accel.acceleration(R,n_start=2,m_start=0)
            elif ("drag" in accel.type):
                a_pert += accel.acceleration(np.concatenate([R,V]))


    if "double" in types:
        pass

    return T.T @ a_pert


def VOP_partials(orbit,force,T,types):
    """
    Computes the perturbing part of the VOP equinoctial state function. Eq. (2.3-5) of Cefola
        orb - orbit in equinoctial form
        a_pert - perturbing acceleration in ECI cartesian form (without the 2-body dynamics)
    """
    orb = orbit.copy(form = "equinoctial")
    a,h,k,p,q,lmb = orb[0:6]

    orb.form = 'cartesian'
    r = np.array(orb[0:3]); v = np.array(orb[3:])
    a_pert = perturbing_osculating_state_fct(r,v,force,orb.date,T,types)
    
    f = 1/(1+p**2+q**2)*np.array([1-p**2+q**2,2*p*q,-2*p])
    g = 1/(1+p**2+q**2)*np.array([2*p*q,1+p**2-q**2,2*q])
    w = 1/(1+p**2+q**2)*np.array([2*p,-2*q,1-p**2-q**2])

    X = np.dot(r,f) ; Y = np.dot(r,g)
    Xd = np.dot(v,f) ; Yd = np.dot(v,g)

    # auxiliary quantities
    n = np.sqrt(mu/a**3)
    A = n*a**2 #np.sqrt(mu*a)
    B = np.sqrt(1-h**2-k**2)
    C = 1 + p**2 + q**2

    # perturbing equinoctial partials
    dh_dv = ((2*Xd*Y-X*Yd)*f-X*Xd*g)/mu + k*(q*Y-p*X)*w/A/B
    dk_dv = ((2*X*Yd-Xd*Y)*g-Y*Yd*f)/mu - h*(q*Y-p*X)*w/A/B
    dalpha_dv = np.array([2*a**2*v/mu,
                          dh_dv,
                          dk_dv,
                          C*Y*w/2/A/B,
                          C*X*w/2/A/B,
                          -2*r/A + (k*dh_dv-h*dk_dv)/(1+B) + (q*Y-p*X)*w/A])


    # wrap gauss VOP differentials
    VOP = dalpha_dv @ a_pert
    return VOP
