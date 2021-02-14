import numpy as np
cimport numpy as np

cpdef osculating_state_fct(t, np.ndarray[np.float64_t] X, force)
cdef double[:] _drag_accel(double [:] R, double [:] V,double r,force)
cdef double[:] _earth_grav_accel(double [:,:] Cnm, double [:,:] Snm, double [:]R,double r,int N,int M,
                                 int n_start=*,int m_start=*)
