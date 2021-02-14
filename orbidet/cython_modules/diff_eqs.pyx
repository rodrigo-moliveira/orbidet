# urls
# https://blog.paperspace.com/faster-numpy-array-processing-ndarray-cython/


# numerical integration scipy / cython ler:
# https://saoghal.net/articles/2020/faster-numerical-integration-in-scipy/#top
# https://stackoverflow.com/questions/32895553/passing-a-cython-function-vs-a-cython-method-to-scipy-integrate
# https://scipy.github.io/devdocs/dev/contributor/cython.html

cimport cython

from libc.math cimport sin, sqrt,cos,atan2,asin,acos, tan
import numpy as np
# from numpy.math import factorial
cimport numpy as np

from beyond.dates import Date
from beyond.frames.iau1980 import precesion,nutation,sideral,earth_orientation

import time
DTYPE = np.double
ctypedef np.float64_t np_float_t

# defining physical constants
cdef double mu = 3.986004415e5
cdef double J2 = 0.108262693e-2
cdef double J3 = -0.253230782e-5
cdef double J4 = -0.162042999e-5
cdef double J5 = -0.227071104e-6
cdef double J6 = 0.540843616e-6
cdef double Re = 6378.137
cdef np_float_t[:] rot_vector_aux = np.array([0,0,7.292115e-5],dtype=DTYPE)

# trick to initialize the rot_vector (may not be the smartest or efficient way though)
def init_arrays(np_float_t[:] rot_vector_aux):
    cdef np.ndarray[np_float_t] aux = np.array(rot_vector_aux, dtype=DTYPE)
    global rot_vector
    rot_vector = aux.copy()
init_arrays(rot_vector_aux)




################################################################################
#########################Osculating State Function##############################
################################################################################


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double[:] _drag_accel(double [:] R, double [:] V,double r,force):
    """
    computes drag acceleration
    (R,V) position and velocity vectors (either in ECI or ECEF)
    r = np.linalg.norm(R)
    force - force model object
    """
    rho = force.get_rho(r)

    v_r = V - np.cross(rot_vector,R)
    v_abs = np.linalg.norm(v_r)
    a_drag = -1/2 * rho * force.sat.CD * force.sat.a_drag / force.sat.m * v_abs * v_r

    cdef double[3] a_ret
    a_ret[0] = a_drag[0];a_ret[1] = a_drag[1];a_ret[2] = a_drag[2]
    return a_ret



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef osculating_state_fct(t, np.ndarray[np_float_t] X, force):
    """
    d/dt(y) = f(y,t),   y=(r,v)
    """

    # Defining variables
    cdef int i
    cdef double r = sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2])
    cdef double r_3 = r * r * r
    cdef np.ndarray[np_float_t] a = np.zeros(3, dtype=DTYPE)
    cdef np.ndarray[np_float_t] a_pert = np.zeros(3, dtype=DTYPE)
    cdef double [:] a_aux, R, V
    cdef np.ndarray[np_float_t,ndim=2] T



    # get rotation matrix
    if force.no_rotations:
        # no rotation (integration and evaluation in TOD frame)
        R = X[0:3]
        V = X[3:]
    else:
      date = Date(t/86400)
      if force.TOD_PEF_rot:
          # integration in TOD, evaluation in PEF
          T = _TOD_to_PEF_date(date)
          R = T @ X[0:3]
          V = T @ X[3:]
      else:
          # integration in EME2000, evaluation in ITRF
          T = _J2000_to_ITRF_rot(date)
          R = T @ X[0:3]
          V = T @ X[3:]


    # Keplerian 2body dynamics
    for i in range(3):
        a[i] = - R[i] * mu / r_3


    # Gravitational Field
    if force.Grav_zonals:
        a_aux =  _zonal_harmonics_accel(R,r,force.Grav_n)
        for i in range(3):
            a_pert[i] += a_aux[i]
    elif force.Grav_complete:
        a_aux = _earth_grav_accel(force.Cnm_,force.Snm_,R,r,force.Grav_n,force.Grav_m,2,0)
        for i in range(3):
            a_pert[i] += a_aux[i]
    # Drag
    if force.DRAG:
        a_aux = _drag_accel(R,V,r,force)
        for i in range(3):
            a_pert[i] += a_aux[i]



    if not force.no_rotations:
          a = T.T @ (a + a_pert)
    else:
          a += a_pert

    # if not force.no_rotations:
    #     a_pert = T.T @ a_pert
    # for i in range(3):
    #     a[i] += a_pert[i]
    return np.concatenate((X[3:],a))





#Zonal Harmonics (Hard-Coded)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
cdef double[:] _zonal_harmonics_accel(double [:] R, double r,int n):
    """
    compute acceleration due to zonal harmonics.
    The harmonics coded are the low-order ones (the equations are hard-coded) and
    the general harmonic series is not applied
    R - np array with position
    r - np.linalg.norm(R)
    n - order of the zonal harmonic. from n=2 (J2) to n=6(J2,J3,J4,J5,J6)
    """
    # initializations
    cdef double x, y, z, ax, ay, az, aux, aux1, r_4
    x = R[0]; y = R[1]; z = R[2]
    ax = ay = az = 0
    cdef double r_2 = r*r
    cdef double[3] a_ret

    if n >= 2:
        #J2
        aux = -3 * J2 * mu * Re**2 / (2 * r **5)
        aux1 = aux*(1 - 5*z**2 / r_2)*x
        ax += aux1
        ay += aux1*y/x
        az += aux*(3 - 5*z**2 / r_2)*z
    if n >= 3:
        #J3
        aux = -5*J3*mu*Re**3/(2*r**7)
        aux1 = aux * (3*z - 7*z**3/r_2)*x
        ax += aux1
        ay += aux1*y/x
        az += aux * (6*z**2 - 7*z**4/r_2 - 3*r_2/5)
    if n >= 4:
        r_4 = r*r*r*r
        #J4
        aux = 15*J4*mu*Re**4/(8*r**7)
        aux1 = aux*(1-14*z**2/r_2+21*z**4/r_4)*x
        ax += aux1
        ay += aux1*y/x
        az += aux * (5 - 70*z**2/(3*r_2) + 21*z**4/r_4)*z
    if n >= 5:
        #J5
        aux=3*J5*mu*Re**5/(8*r**9)
        aux1 = aux * x*y * (35-210*z**2/r_2 + 231*z**4/r_4)
        ax += aux1; ay += aux1
        az += aux * z**2 * (105 - 315*z**2/r_2 + 231*z**4/r_4) + 15*J5*mu*Re**5/(8*r**7)
    if n >= 6:
        aux = -J6*mu*Re**6/(16*r**9)
        aux1 = aux*x*(35 - 945*z**2/r_2 + 3465*z**4/r_4 - 3003*z**6/r**6)
        ax += aux1; ay += aux1*y/x
        az += aux * (245 - 2205*z**2/r_2 + 4851*z**4/r_4 - 3003*z**6/r**6)

    a_ret[:] = [ax,ay,az]
    return a_ret


cdef long double factorial (int x):
    cdef long double y = 1
    cdef int i
    for i in range (1,x+1):
        y*=i
    return y


@cython.cdivision(True)
cdef inline double normalize_P(int n,int m):
    cdef int k = 1 if m == 0 else 2
    return sqrt(  (factorial(n-m) * k * (2*n + 1)) / factorial(n+m) )


@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double [:,:] Legendre_coeff(int N, int M, double phi):
    """
    computes the Legendre normalized coefficients for the function sin(phi), phi is the lattitude
    """
    #1 - calculate unnormalized coeff.

    # initialize
    cdef double [:,:] P_ = np.zeros((N+1,N+1), dtype=DTYPE)
    cdef double cos_phi = cos(phi)
    cdef double sin_phi = sin(phi)
    cdef double P_aux,test
    cdef int n,m

    P_[0][0] = 1.0
    P_[1][0] = sin_phi
    P_[1][1] = cos_phi

    #recursive alg
    for n in range(2,N+1):
        #diagonal:
        P_[n][n] = (2*n-1)*cos_phi*P_[n-1][n-1]
        test = (2*n - 1) * cos_phi * P_[n-1][n-1]

        #horizontal first step coefficients
        P_[n][0] = ((2*n-1) * sin_phi * P_[n-1][0] - (n-1)*P_[n-2][0]) / n

        #horizontal second step coefficients
        for m in range(1,n):
            if m > n - 2:
                P_aux = 0
            else:
                P_aux = P_[n-2][m]
            P_[n][m] = P_aux + (2*n-1)*cos_phi*P_[n-1][m-1]

    #2 - normalize the coefficients
    for n in range(N+1):
        for m in range(n+1):
            P_[n][m] = P_[n][m] * normalize_P(n,m)
    return P_

@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double[:] _earth_grav_accel(double [:,:] Cnm, double [:,:] Snm, double [:]R,double r,int N,int M,
                                 int n_start=2,int m_start=0):
    """
    Cnm and Snm are the Earth coefficients

    computes the gravitational earth acceleration with coefficients NxM
    R is the position vector in the provided frame (should be ITRF)
    r = np.linalg.norm(R)
    The calculations are made in the ECEF ITRF frame and then uppon return transformed back to ECI
    n_start -> initial n value to start the summation (default is 2: excludes the central force term
            (1st order terms all vanish))
    """

    # initialize local variables
    cdef int n,m
    cdef double[3] a_ret

    # get spherical coordinates
    cdef double latgc = asin(R[2]/r)
    cdef double lon = atan2(R[1],R[0])

    # Legendre coefficients
    cdef double [:,:] P = Legendre_coeff(N+2,N+2,latgc)

    # partial potential derivatives (equation 7.21 VALLADO)
    cdef double dU_dr=0.0, dU_dlat=0.0, dU_dlon = 0.0
    cdef double q2=0.0, q1=0.0, q0 = 0.0
    cdef double b0, b1, b2, N0, N1, y, aux
    cdef double ax,ay,az

    for n in range(n_start,N+1):
        b0 = (-mu / r**2) * (Re / r)**n * (n+1)
        b1 = b2 = (mu / r) * (Re / r)**n

        for m in range(m_start,n+1):
            if m > M: break
            aux = Cnm[n][m]*cos(m*lon) + Snm[n][m]*sin(m*lon)
            N0 = normalize_P(n,m)

            if n != m:
                N1 = normalize_P(n,m+1)
                y = P[n][m+1]
            else:
                N1=1
                y = 0
            q0 += P[n][m] * aux
            q1 += (y*N0/N1 - (m)*tan(latgc) * P[n][m]) * aux
            q2 += m * P[n][m] * (Snm[n][m]*cos(m*lon) - Cnm[n][m]*sin(m*lon))

        dU_dr += q0*b0
        dU_dlat += q1*b1
        dU_dlon += q2*b2
        q2=0.0; q1=0.0; q0 = 0.0

    #ECEF acceleration components (equation 7.23 VALLADO)
    cdef double r2xy = R[0]**2 + R[1]**2
    cdef double aux_0 = (1/r*dU_dr-R[2]/(r**2*sqrt(r2xy))*dU_dlat)
    cdef double aux_1 = (1/r2xy*dU_dlon)

    ax = aux_0*R[0] - aux_1*R[1]
    ay = aux_0*R[1] + aux_1*R[0]
    az =  1/r*dU_dr*R[2]+sqrt(r2xy)/r**2*dU_dlat
    a_ret[:] = [ax,ay,az]

    return a_ret





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

cdef _TOD_to_PEF_date(date):
    return sideral(date,model="apparent", eop_correction=False).T
