cimport cython

from libc.math cimport sqrt,abs,pi,sin,cos,atan2,atan#sin, sqrt,cos,atan2,asin,acos, tan
import numpy as np
cimport numpy as np
from scipy.integrate import fixed_quad

from beyond.frames.iau1980 import precesion,nutation,sideral,earth_orientation
from beyond.dates import Date
from beyond.orbits import Orbit

from diff_eqs cimport osculating_state_fct,_drag_accel,_earth_grav_accel

DTYPE = np.double
ctypedef np.float64_t np_float_t

# defining physical constants
cdef double mu = 3.986004415e5

@cython.cdivision(True)
cdef double eccentric_longitude(double lmb, double h, double k):
      """
      solve eccentric longitude F using equinoctial form of Kepler's Eq. (Newton's Method):
      lmb = F + h cos(F) - k sin(F)
      [Cefola]
      """
      cdef double error = 1e-12, ratio = 1.0
      cdef int NMAX = 20, i = 0

      #start point
      cdef double Fi = lmb
      while (abs(ratio) > error) and i < NMAX:
          ratio = (Fi + h*cos(Fi) - k*sin(Fi) - lmb) / (1 - h*sin(Fi) - k*cos(Fi))
          Fi = Fi - ratio
          i += 1
      return Fi % (2*pi)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
cdef double[:] equinoctial_to_cartesian(double a, double h, double k, double lmb, double [:] f, double [:] g,
                                        double n):
      """
      convert equinoctial state directly to cartesian, cf. section 2.4.1 of Cefola
      equinoctial state: (a,h,k,p,q,lmb)
      cartesian state: (x,y,z,vx,vy,vz)
      """

      # step 1.
      cdef double F = eccentric_longitude(lmb, h, k)
      cdef double b = 1 / (1 + sqrt(1-h*h-k*k))

       # step 2.
      aux = 1 - h*sin(F) - k*cos(F)
      cdef double sin_L = ((1-k*k*b)*sin(F) + h*k*b*cos(F) - h) / (aux)
      cdef double cos_L = ((1-h*h*b)*cos(F) + h*k*b*sin(F) - k) / (aux)
      # cdef double L = atan2(sin_L,cos_L)

      # step 3
      cdef double r = a*(1 - h*sin(F) - k*cos(F))
      cdef double X = r*cos_L
      cdef double Y = r*sin_L
      cdef double X_dot = -(n*a*(h + sin_L)) / (sqrt(1 - h*h - k*k))
      cdef double Y_dot = (n*a*(k + cos_L)) / (sqrt(1 - h*h - k*k))

      # filling up return vector
      cdef double ret[10]
      cdef int i
      for i in range(3):
          ret[i] = X * f[i] + Y * g[i]
          ret[i+3] = X_dot * f[i] + Y_dot * g[i]
      ret[6] = X
      ret[7] = Y
      ret[8] = X_dot
      ret[9] = Y_dot
      return ret

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef np.ndarray[np_float_t] _perturbing_osculating_state_fct(np.ndarray[np_float_t] R, np.ndarray[np_float_t] V,
                                      np.ndarray[np_float_t,ndim=2] T, force,_types=""):



    cdef double[:] _R = T @ R
    cdef double[:] _V = T @ V
    cdef double r = sqrt(_R[0]*_R[0] + _R[1]*_R[1] + _R[2]*_R[2])
    cdef np.ndarray[np_float_t] a_pert = np.zeros(3, dtype=DTYPE)
    cdef double [:] aux

    # Gravitational Field
    if "zonals" in _types:
        aux = _earth_grav_accel(force.Cnm_,force.Snm_,_R,r,force.Grav_n,0,n_start=2,m_start=0)
        for i in range(3):
            a_pert[i] += aux[i]
    if "tesserals" in _types:
        aux = _earth_grav_accel(force.Cnm_,force.Snm_,_R,r,force.Grav_n,force.Grav_m,n_start=2,m_start=1)
        for i in range(3):
            a_pert[i] += aux[i]
    # Drag
    if "drag" in _types:
        aux = _drag_accel(_R,_V,r,force)
        for i in range(3):
            a_pert[i] += aux[i]

    return T.T @ a_pert

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
cpdef np.ndarray[np_float_t] VOP_partials(orbit,force, np.ndarray[np_float_t,ndim=2] T,_types=("",)):
    """
    Computes the perturbing part of the VOP equinoctial state function. Eq. (2.3-5) of Cefola
        orb - orbit in equinoctial form
        a_pert - perturbing acceleration in ECI cartesian form (without the 2-body dynamics)
        _types (str): "zonals", "drag", "tesserals"
    """
    orb = orbit.copy(form = "equinoctial")
    cdef double a=orb[0],h=orb[1],k=orb[2],p=orb[3],q=orb[4],lmb=orb[5]
    cdef double n = sqrt(mu / (a*a*a))

    cdef double f[3]
    cdef double g[3]
    cdef double w[3]

    cdef double aux = 1 / (1 + p*p + q*q)
    f[:] = [(1.0-p*p+q*q)*aux,
           2.0*p*q*aux,
           -2.0*p*aux]
    g[:] = [2.0*p*q*aux,
           (1.0+p*p-q*q)*aux,
           2.0*q*aux]
    w[:] = [2.0*p*aux,
           -2.0*q*aux,
           (1.0-p*p-q*q)*aux]

    # cartesian form and perturbing acceleration
    cdef double[:] info = equinoctial_to_cartesian(a, h, k, lmb, f, g, n)
    cdef np.ndarray[np_float_t] r = np.array([info[0],info[1],info[2]], dtype=DTYPE)
    cdef np.ndarray[np_float_t] v = np.array([info[3],info[4],info[5]], dtype=DTYPE)

    cdef double X = info[6]
    cdef double Y = info[7]
    cdef double Xd = info[8]
    cdef double Yd = info[9]

    cdef np.ndarray[np_float_t] a_pert = _perturbing_osculating_state_fct(r,v,T,force,_types)

    # auxiliary quantities
    cdef double A = n*a*a
    cdef double B = sqrt(1-h*h-k*k)
    cdef double C = 1 + p*p + q*q

    # perturbing equinoctial partials
    cdef double dh_dv[3]
    cdef double dk_dv[3]
    # cdef double [6][3]
    cdef np.ndarray[np_float_t,ndim=2] dalpha_dv = np.zeros((6,3), dtype=DTYPE)
    cdef int i
    for i in range(3):
        dalpha_dv[0][i] = 2*v[i]/n/n/a
        dalpha_dv[1][i] = ((2*Xd*Y-X*Yd)*f[i]-X*Xd*g[i])/mu + k*(q*Y-p*X)*w[i]/A/B
        dalpha_dv[2][i] = ((2*X*Yd-Xd*Y)*g[i]-Y*Yd*f[i])/mu - h*(q*Y-p*X)*w[i]/A/B
        dalpha_dv[3][i] = C*Y*w[i]/2/A/B
        dalpha_dv[4][i] = C*X*w[i]/2/A/B
        dalpha_dv[5][i] = -2*r[i]/A + (k*dalpha_dv[1][i]-h*dalpha_dv[2][i])/(1+B) + (q*Y-p*X)*w[i]/A

    # wrap gauss VOP differentials
    cdef np.ndarray[np_float_t] VOP = dalpha_dv @ a_pert
    return VOP



################################################################################
#######################-Semi-analytical State Function-#########################
################################################################################
def equinoctial_state_fct(t, np.ndarray[np_float_t] X,*args):
    """
    n - order of the fixed quadrature
    NOTA (14/AGOSTO/20): USA ISTO PARA TENTAR CORRER OS QUADS MAIS RAPIDO
	https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html#quad-callbacks
	https://saoghal.net/articles/2020/faster-numerical-integration-in-scipy/#top

    """
    force = args[0]
    _types = args[1]
    n = args[2]
    if isinstance(t,float):
        t = Date(t/86400)

    # reconstruct the orbit object
    mean_orb = Orbit(t, X,"equinoctial",force.integration_frame,None)
    cdef double lmb = X[5]

    # quadratures (mean dynamics)
    f_mean,_ =  fixed_quad(wrapp_VOP_quad, lmb, lmb+2*pi,args=(mean_orb,force,_types),n=n)
    f_mean = f_mean / 2 / pi

    _n = sqrt(mu / (mean_orb.a)**3)
    return np.array ( [ f_mean[0],
                       f_mean[1],
                       f_mean[2],
                       f_mean[3],
                       f_mean[4],
                       _n + f_mean[5]])

cpdef np.ndarray[np_float_t,ndim=2] wrapp_VOP_quad(double[:] x, orbit,force,_types):
    """
    _types = ("zonals","drag")
    tesserals do not come into play for a double quadrature (cf. ELY)
    """
    cdef int n = len(x), i
    orb_arr = []
    for i in range (n):
        orb = orbit.copy()
        orb[5] = x[i] % (2*pi)
        orb_arr.append(orb)

    T = _J2000_to_ITRF_rot(orbit.date) if not force.TOD_PEF_rot else np.eye(3)
    Fs = np.array([np.array(VOP_partials(orb,force,T,_types)) for orb in orb_arr])
    return Fs.T



cdef _J2000_to_ITRF_rot(date):
    """utility function to get rotation matrix from J2000 to ITRF
    This is just to avoid defining an Orbit object to rotate the accelerations
    (which is less efficient)
    usage:
        X_ITRF = T @ X_J2000
    rotate state from J2000 to ITRF
    """
    return (
        earth_orientation(date).T @ sideral(date,model="apparent", eop_correction=False).T @
        nutation(date,eop_correction=False).T @ precesion(date).T
    )
