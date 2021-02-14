cimport cython

from libc.math cimport sqrt#sin, sqrt,cos,atan2,asin,acos, tan
import numpy as np
cimport numpy as np

from diff_eqs cimport osculating_state_fct

DTYPE = np.double
ctypedef np.float64_t np_float_t

# defining physical constants
cdef double mu = 3.986004415e5
cdef double J2 = 0.108262693e-2
cdef double Re = 6378.137

################################################################################
########################-Extended Kalman Filter-################################
################################################################################

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def EKF_LS_diff_eq(double t,np.ndarray[np_float_t] Y ,force):
    """
    wrapper that contains the state and phi diff eqs.
    These two functions are packed in a system to feed the ODE solver
    This system is used for the EKF and LS
    """
    #unpack the variables
    cdef np.ndarray[np_float_t, ndim=2] Y_matrix = np.reshape(Y,(7,6))

    cdef np.ndarray[np_float_t] y = Y_matrix[0]
    cdef np.ndarray[np_float_t,ndim=2] phi = Y_matrix[1:]

    #apply differential functions
    dydt = osculating_state_fct(t,y,force) #derivative of position and velocity
    F = jacobian_state_fct(y,force)
    dphidt = F @ phi

    #pack the variables
    return np.append(dydt,dphidt)


@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef jacobian_state_fct(np.ndarray[np_float_t] X,force=None):
    """
    evaluates the jacobian of state function
    force object has the information regarding the force model (ForceModel Object)
    A simplistic approach considering only an hard-coded J2 harmonic is used for simplicity
    """
    cdef np.ndarray[np_float_t, ndim=2] grad = np.zeros((6,6), dtype=DTYPE)
    cdef np.ndarray[np_float_t, ndim=2] grad_oblat = np.zeros((6,6), dtype=DTYPE)

    cdef double x=X[0], y=X[1], z=X[2]
    cdef double vx=X[3], vy=X[4], vz=X[5]

    cdef double r = sqrt(x*x+y*y+z*z)
    cdef double r_5 = r * r * r * r * r
    cdef double r_3 = r * r * r

    cdef double r_7,r_9,aux,ax,dax_dx,dax_dy,dax_dz,day_dx,day_dy,day_dz,daz_dx,daz_dy,daz_dz

    #two-body dynamics hardcoded
    grad[0,3] = 1
    grad[1,4] = 1
    grad[2,5] = 1
    grad[3,0] = mu*3*x*x/r_5-mu/r_3
    grad[3,1] = mu*3*x*y/r_5
    grad[3,2] = mu*3*x*z/r_5
    grad[4,0] = mu*3*x*y/r_5
    grad[4,1] = mu*3*y*y/r_5-mu/r_3
    grad[4,2] = mu*3*y*z/r_5
    grad[5,0] = mu*3*x*z/r_5
    grad[5,1] = mu*3*y*z/r_5
    grad[5,2] = mu*3*z*z/r_5-mu/r_3
    # grad = np.array([[0,0,0,1,0,0],
    #                 [0,0,0,0,1,0],
    #                 [0,0,0,0,0,1],
    #                 [mu*3*x**2/r**5-mu/r**3, mu*3*x*y/r**5, mu*3*x*z/r**5,0,0,0],
    #                 [mu*3*x*y/r**5, mu*3*y**2/r**5-mu/r**3, mu*3*y*z/r**5,0,0,0],
    #                 [mu*3*x*z/r**5, mu*3*y*z/r**5, mu*3*z**2/r**5-mu/r**3,0,0,0]])

    if (force != None) and (force.Grav_zonals or force.Grav_complete):
            #analytical (hardcoded) jacobian
            r_7 = r_5 * r * r
            r_9 = r_7 * r * r

            aux = -3*J2*mu*Re*Re / (2*r_5)
            ax = aux*(1 - 5*z*z / (r*r))*x

            dax_dx = aux * (1/r_5 - 5*x*x/r_7 - 5*z*z/r_7 + 35*z*z*x*x/r_9)
            dax_dy = aux * (-5*x*y/r_7 + 35*z*z*x*y/r_9)
            dax_dz = aux * (-5*x*z/r_7 + 35*z*z*z*x/r_9)

            day_dx = dax_dy
            day_dy = y/x * dax_dy + ax/x
            day_dz = y/x * dax_dz

            daz_dx = aux * (-15*x*z/r_7 + 35*z*z*z*x/r_9)
            daz_dy = aux * (-15*y*z/r_7 + 35*z*z*z*y/r_9)
            daz_dz = aux * (3/r_5 - 30*z*z/r_7 + 35*z*z*z*z/r_9)

            grad_oblat[3,0] = dax_dx
            grad_oblat[3,1] = dax_dy
            grad_oblat[3,2] = dax_dz
            grad_oblat[4,0] = day_dx
            grad_oblat[4,1] = day_dy
            grad_oblat[4,2] = day_dz
            grad_oblat[5,0] = daz_dx
            grad_oblat[5,1] = daz_dy
            grad_oblat[5,2] = daz_dz

            # grad_oblat = np.array([[0,0,0,0,0,0],
            #                         [0,0,0,0,0,0],
            #                         [0,0,0,0,0,0],
            #                         [dax_dx,dax_dy,dax_dz,0,0,0],
            #                         [day_dx,day_dy,day_dz,0,0,0],
            #                         [daz_dx,daz_dy,daz_dz,0,0,0]])
            grad += grad_oblat
    return grad





################################################################################
########################-Unscented Kalman Filter-###############################
################################################################################

#Discretization of the provided continuous (state) function with RK4
def discrete_function(np.ndarray[np_float_t] x0,double t0,double t1,double dt, f_continuous):
    """
    function that applies discretization of the state function
    we can say x(t_k+1) = f_discrete(x(t_k))
    discretization is made from t0(t_k) to t1(t_k+1)
    INPUTS:
        x0 - state at t0
        t1 - final integration time
        dt - stepsize of integration (t0 + dt + ... + dt = t1)
        f_continuous is the continuous function to discretize
    """
    cdef Py_ssize_t n = x0.shape[0]
    cdef np.ndarray[np_float_t] x = x0
    while (t0 < t1):
        x = rk4vec ( t0, n, x, dt, f_continuous)
        t0 += dt
    return x

#Vectorized function
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
cdef np.ndarray[np_float_t] rk4vec ( double t0,Py_ssize_t m, np.ndarray[np_float_t] u0,double dt, f ):
    """
    applies one stepsize runge kutta integration
    """
    cdef int i = 0
    #  Get four sample values of the derivative.
    cdef np.ndarray[np_float_t] f0 = f ( t0, u0 )

    cdef double t1 = t0 + dt / 2.0
    cdef np.ndarray[np_float_t] u1 = np.zeros(m)
    for i in range(m):
        u1[i] = u0[i] + dt * f0[i] / 2.0
    cdef np.ndarray[np_float_t] f1 = f ( t1, u1 )

    cdef double t2 = t0 + dt / 2.0
    cdef np.ndarray[np_float_t] u2 = np.zeros(m)
    for i in range(m):
        u2[i] = u0[i] + dt * f1[i] / 2.0
    cdef np.ndarray[np_float_t] f2 = f ( t2, u2 )

    cdef double t3 = t0 + dt
    cdef np.ndarray[np_float_t] u3 = np.zeros(m)
    for i in range(m):
        u3[i] = u0[i] + dt * f2[i]
    cdef np.ndarray[np_float_t] f3 = f ( t3, u3 )


    #  Combine them to estimate the solution U at time T1.
    cdef np.ndarray[np_float_t] u = np.zeros(m)
    for i in range(m):
        u[i] = u0[i] + ( dt / 6.0 ) * (f0[i]  + 2.0 * f1[i] + 2.0 * f2[i] +  f3[i] )

    return u




@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
def predic_dif_eq_sqroot(double t, np.ndarray[np_float_t] X,f, np.ndarray[np_float_t] wm,
                         np.ndarray[np_float_t,ndim=2]W,np.ndarray[np_float_t,ndim=2]Q,int n,double lamb):
    """
    This function applies the square root version of the continuous predict step
    equation (35)
    This fuction does not rely on the UT_matrix() function because the implementation is a bit different
    (it would get a bit messy to try an unified solution - plus this way it's less computations)
    """
    #unpack the sigma Matrix
    cdef np.ndarray[np_float_t,ndim=2]sigmaX = np.reshape(X,(n,2*n+1))

    cdef double sqrt_c = sqrt(n+lamb)

    #recover A from sigmaX and get inv(A)
    cdef np.ndarray[np_float_t]m = sigmaX[0:,0]
    cdef np.ndarray[np_float_t,ndim=2]mMatrix = np.full((n,n),m).T

    A = (sigmaX[0:,1:7] - mMatrix) / sqrt_c
    A_inv = np.linalg.inv(A)

    #apply state function to sigmaX
    fun = lambda _x: f(t,_x)
    cdef np.ndarray[np_float_t,ndim=2] sigmaY = np.apply_along_axis(fun, 0, sigmaX)


    #get M and phi(M)
    cdef np.ndarray[np_float_t,ndim=2]M = A_inv @ ( sigmaX @ W @ sigmaY.T + sigmaY @ W @ sigmaX.T + Q) @ A_inv.T
    cdef np.ndarray[np_float_t,ndim=2]phi = np.tril(np.ones((n,n)))
    np.fill_diagonal(phi, 0.5)
    cdef np.ndarray[np_float_t,ndim=2]phi_M = phi * M              #element-wise multiplication. eq (33)

    #finally, get dX_dt (column by column) and pack result
    cdef np.ndarray[np_float_t,ndim=2]mean_sigY = np.full((2*n+1,n),sigmaY@wm).T

    cdef np.ndarray[np_float_t,ndim=2]block = mean_sigY + sqrt_c*np.block([np.zeros((n,1)),A @ phi_M,-A @ phi_M])
    dX_dt = block.flatten()
    return dX_dt
